from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs._utils import flat_action
from sekai.envs.finance._common import (
    check_cost_inputs,
    check_number,
    check_sources,
    episode_over,
)
from sekai.error import InvalidAction, ResetRequired
from sekai.finance.account import AccountConfig, project_weights, settle
from sekai.finance.costs import CostModel, as_cost_model
from sekai.finance.data import MarketData
from sekai.finance.rewards import LogReturn, RewardFunction, StepOutcome
from sekai.finance.sampling import EpisodeSampler, as_sampler, resolve_start, sample_start
from sekai.finance.simulators import GBM, ListingProcess, MarketModel, RegimeSwitching
from sekai.finance.universe import SlotMap
from sekai.spaces.box import Box
from sekai.spaces.dict import Dict
from sekai.spaces.masked_box import MaskedBox


def default_market(n_assets: int = 10) -> MarketModel:
    """The simulated market of Market-v0: a two-regime market whose assets list and delist.

    A calm regime (8% drift, 15% volatility, mild correlation) and a stressed one
    (-20% drift, 40% volatility, high correlation) alternate as a hidden Markov
    chain with persistent regimes (about 2 years calm, 6 months stressed on
    average). Each asset lists and delists over time, so the universe changes
    within an episode.
    """
    calm_corr = np.full((n_assets, n_assets), 0.2) + 0.8 * np.eye(n_assets)
    stress_corr = np.full((n_assets, n_assets), 0.7) + 0.3 * np.eye(n_assets)
    regimes = [
        GBM(mu=0.08, sigma=0.15, corr=calm_corr, n_assets=n_assets),
        GBM(mu=-0.2, sigma=0.4, corr=stress_corr, n_assets=n_assets),
    ]
    p_calm, p_stress = 1 / 504, 1 / 126
    transition = [[1 - p_calm, p_calm], [p_stress, 1 - p_stress]]
    return ListingProcess(
        RegimeSwitching(regimes, transition),
        initial_listed=0.7,
        listing_rate=1.0,
        delisting_rate=0.3,
        min_listed=2,
    )


class MarketEnv(Env[dict[str, Any], NDArray[np.float32]]):
    """Multi-asset trading over a changing universe, with features, outside data and costs.

    The general-purpose finance environment, assembled from `sekai.finance`
    components. Assets list and delist (NaN prices in a `MarketData`, or a
    `ListingProcess` market model), so the number of tradable assets changes
    from step to step; the observation is a fixed-size padded set (`MaskedBox`)
    with a validity mask, and the action mask marks the tradable slots.

    **Data.** Either a historical `MarketData` as `data` (episodes are windows of it,
    chosen by `sampler` or `reset(options={"start": ...})`) or a `model` that
    simulates a fresh market each episode (default: `default_market()`, two
    hidden regimes and listings/delistings), not both.

    **Slots.** With `max_assets=None` each asset has its own slot. With a smaller
    `max_assets` (a large historical universe, few names alive at once) assets
    are mapped to slots by a `SlotMap`: an asset keeps its slot while it is
    tradable, and newcomers take free slots (incumbents first, then in column
    order, if more are tradable than there are slots). `env.slot_assets` names
    the asset in each slot.

    **Observation** (`Dict`):

    - `"assets"`: `MaskedBox` of `max_assets` slots, each a `(lookback, F)` window
      of per-asset features: the log return, an availability flag (1 where the
      return is defined), then each data field named in `features`. Missing
      values are 0.
    - `"portfolio"`: `Box (max_assets + 1,)`, current weights per slot, cash last.
    - `"account"`: `Box (2,)`: log equity and gross exposure.
    - `"market"` (only with `exogenous`): `Box (lookback, E)` window of the
      market-wide series named in `exogenous` (MarketData exogenous series, or,
      in simulation, model state such as the regime, which makes it observable).

    **Action.** Long-only (default): `Box(0, 1, (max_assets + 1,))` scores, cash
    last, normalised into weights (scaled to `max_leverage`). Long/short:
    `Box(-1, 1, (max_assets,))` target weights, scaled down if gross exposure
    exceeds `max_leverage`. Untradable slots are ignored.

    **Dynamics.** At each decision the account rebalances to the target weights,
    pays `transaction_cost` (any `CostModel`), then holds them over the next
    period: assets earn their returns, cash earns `cash_rate` (or pays
    `margin_rate` when borrowed), shorts pay `borrow_rate`. An asset that delists
    during the period returns `delisting_return` and its position becomes cash.

    **Reward**: `reward(outcome)` for a `sekai.finance.RewardFunction`, by default
    the log growth of equity. **Termination**: equity wiped out.
    **Truncation**: after `episode_length` steps. An episode is a window of an ongoing
    market, so its end is a truncation and the observation holds no clock; wrap the env
    in `TimeAwareObservation(env, max_episode_steps=episode_length)` to observe one.

    **info**: `equity`, `turnover`, `cost`, `financing`, `gross_exposure`,
    `n_assets` (tradable), `weights` (per slot), `timestamp` (the data's timestamp,
    or the step index in simulation), `dt`; plus `state`
    (the simulation's latent state at the new decision time) when
    `privileged_state=True`, for asymmetric actor-critic methods.

    `step()` after the episode has ended raises `ResetRequired`.

    Args:
        data: Historical data (NaN prices = not listed).
        model: Market model for simulation when no data is given.
        max_assets: Slots in the observation (default: every asset).
        features: Names of per-asset data fields to observe.
        exogenous: Names of market-wide series to observe.
        lookback: Rows in each observation window.
        episode_length: Decisions per episode.
        sampler: Placement of episodes in the history ("random", "sequential", an
            int, or a `sekai.finance.EpisodeSampler`).
        transaction_cost: Proportional rate or `CostModel`, applied per asset (so
            per-asset parameters follow the assets, whatever their slots). Impact and
            spread models read the `volume` (shares) and `spread` (fraction of the
            price) data fields when present.
        reward: Reward function (default `LogReturn()`). The env keeps a copy (as it
            does of a `reward_fn` set later), so a stateful reward is never shared.
        allow_short: Long/short action space.
        max_leverage: Cap on gross exposure.
        cash_rate: Annual interest earned on positive cash.
        margin_rate: Annual interest paid on borrowed cash (default: cash_rate).
        borrow_rate: Annual fee on the value of short positions.
        capital: Account size in currency (for fixed and impact costs).
        dt: Years per step (default: from the data's timestamps, the model, or 1/252).
        delisting_return: Return of an asset over the period in which it delists.
        privileged_state: Add the simulation's latent state to info.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 10}

    def __init__(
        self,
        data: MarketData | None = None,
        *,
        model: MarketModel | None = None,
        max_assets: int | None = None,
        features: Sequence[str] = (),
        exogenous: Sequence[str] = (),
        lookback: int = 20,
        episode_length: int = 252,
        sampler: str | int | EpisodeSampler = "random",
        transaction_cost: float | CostModel = 1e-3,
        reward: RewardFunction | None = None,
        allow_short: bool = False,
        max_leverage: float = 1.0,
        cash_rate: float = 0.0,
        margin_rate: float | None = None,
        borrow_rate: float = 0.0,
        capital: float = 1_000_000.0,
        dt: float | None = None,
        delisting_return: float = 0.0,
        privileged_state: bool = False,
    ) -> None:
        check_sources("MarketEnv", data=data, model=model, takes_prices=False)
        if lookback < 1 or episode_length < 1:
            raise ValueError("lookback and episode_length must be >= 1.")
        if dt is not None:
            dt = check_number("dt", dt, ">0")
        if data is None and model is None:
            model = default_market()
        self.data = data
        self.model = model
        self._lookback = lookback
        self._episode_length = episode_length
        self._window = lookback + episode_length + 1
        self.features = tuple(features)
        self.exogenous = tuple(exogenous)
        self.sampler = as_sampler(sampler)
        self.cost_model = as_cost_model(transaction_cost)
        # A copy (the setter copies): stateful rewards must not be shared between envs.
        self.reward_fn = reward if reward is not None else LogReturn()
        self.config = AccountConfig(
            allow_short=allow_short,
            max_leverage=check_number("max_leverage", max_leverage, ">0"),
            cash_rate=check_number("cash_rate", cash_rate),
            margin_rate=None if margin_rate is None else check_number("margin_rate", margin_rate),
            borrow_rate=check_number("borrow_rate", borrow_rate, ">=0"),
        )
        self.capital = check_number("capital", capital, ">0")
        self.delisting_return = check_number("delisting_return", delisting_return)
        if self.delisting_return < -1.0:
            raise ValueError(
                f"delisting_return must be >= -1 (a total loss), got {delisting_return!r}."
            )
        self.privileged_state = privileged_state

        exo_widths: list[int] = []
        if data is not None:
            if data.n_steps < self._window:
                raise ValueError(
                    f"The data has {data.n_steps} rows but an episode needs {self._window} "
                    "(lookback + episode_length + 1)."
                )
            for name in self.features:
                data.field(name)  # KeyError for unknown fields
            exo_widths = [data.exogenous_width(name) for name in self.exogenous]
            n_total = data.n_assets
            ppy = data.periods_per_year
            self.dt = dt if dt is not None else (1.0 / ppy if ppy else 1 / 252)
        else:
            assert model is not None
            if self.features:
                raise ValueError(
                    "features= names data fields, so it needs data=. To add features to a "
                    "simulated market, simulate a history into MarketData and add fields."
                )
            spec = model.state_spec()
            for name in self.exogenous:
                if name not in spec:
                    raise KeyError(f"The model has no state {name!r}; it has {sorted(spec)}.")
                exo_widths.append(int(np.prod(spec[name])) if spec[name] else 1)
            n_total = model.n_assets
            self.dt = dt if dt is not None else model.dt
        check_cost_inputs(self.cost_model, data, simulated=data is None, lookback=lookback)
        self.n_total = n_total
        self.max_assets = n_total if max_assets is None else int(max_assets)
        if self.max_assets < 1:
            raise ValueError("max_assets must be >= 1.")
        self._dynamic_slots = self.max_assets < n_total
        self._slots = SlotMap(self.max_assets)
        self._n_features = 2 + len(self.features)
        self._n_exo = sum(exo_widths)

        S = self.max_assets
        spaces: dict[str, Any] = {
            "assets": MaskedBox(
                -np.inf, np.inf, shape=(lookback, self._n_features), max_size=S, dtype=np.float32
            ),
            "portfolio": Box(-np.inf, np.inf, shape=(S + 1,), dtype=np.float32),
            "account": Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        }
        if self.exogenous:
            spaces["market"] = Box(-np.inf, np.inf, shape=(lookback, self._n_exo), dtype=np.float32)
        self.observation_space = Dict(spaces)
        if allow_short:
            self.action_space = Box(-1.0, 1.0, shape=(S,), dtype=np.float32)
        else:
            self.action_space = Box(0.0, 1.0, shape=(S + 1,), dtype=np.float32)

        self._started = False
        self._over = False
        self._t = 0
        self._start = 0
        self.equity = 1.0
        self.weights = np.zeros(S)
        self._slot_cols = np.full(S, -1, dtype=np.intp)

    @property
    def reward_fn(self) -> RewardFunction:
        """The reward function. Setting it stores a copy, as `reward=` does, so a stateful
        reward set on several envs at once (e.g. with a vector env's `set_attr`) is not shared."""
        return self._reward_fn

    @reward_fn.setter
    def reward_fn(self, value: RewardFunction) -> None:
        self._reward_fn = copy.deepcopy(value)

    # ------------------------------------------------------------------
    # Episode data
    # ------------------------------------------------------------------

    def _load_episode(self, options: dict[str, Any] | None) -> None:
        if self.data is not None:
            n_starts = self.data.n_steps - self._window + 1
            if options and options.get("start") is not None:
                start = resolve_start(
                    options["start"],
                    timestamps=self.data.timestamps,
                    lookback=self.lookback,
                    n_starts=n_starts,
                    warmup=self.data.warmup,
                )
            else:  # no decision on the warm-up rows of a split
                start = sample_start(
                    self.sampler,
                    self.rng,
                    n_starts=n_starts,
                    episode_length=self.episode_length,
                    lookback=self.lookback,
                    warmup=self.data.warmup,
                )
            rows = slice(start, start + self._window)
            prices = np.asarray(self.data.prices[rows], dtype=np.float64)
            fields = [np.asarray(self.data.field(f)[rows], dtype=np.float64) for f in self.features]
            exo = [
                self.data.exogenous_series(e)[rows].reshape(self._window, -1)
                for e in self.exogenous
            ]
            self._volume = self._optional_field("volume", rows)
            self._spread = self._optional_field("spread", rows)
            self._state: dict[str, NDArray[np.float64]] = {}
            self._times = np.asarray(self.data.timestamps[rows])
            self._start = start
        else:
            assert self.model is not None
            if options and options.get("start") is not None:
                raise ValueError("options['start'] needs historical data (data=...).")
            sim = self.model.simulate(self.rng, self._window - 1)
            prices = np.asarray(sim.prices, dtype=np.float64)
            fields = []
            self._state = {k: np.asarray(v) for k, v in sim.state.items()}
            exo = [self._state[e].reshape(self._window, -1) for e in self.exogenous]
            self._volume = self._spread = None
            self._times = np.arange(self._window) - self.lookback  # the step index at decisions
            self._start = 0

        valid = np.isfinite(prices)
        self._prices = np.where(valid, prices, np.nan)
        self._valid = valid
        with np.errstate(invalid="ignore", divide="ignore"):
            log_ret = np.log(prices[1:] / prices[:-1])
        log_ret = np.vstack([np.full((1, prices.shape[1]), np.nan), log_ret])
        available = np.isfinite(log_ret)
        self._log_returns = np.where(available, log_ret, 0.0)
        # (T, N, F) feature tensor: return, availability, fields; missing values are 0.
        channels = [self._log_returns, available.astype(np.float64)]
        channels += [np.nan_to_num(f, nan=0.0) for f in fields]
        self._features = np.stack(channels, axis=-1).astype(np.float32)
        self._available = available
        self._exo = (
            np.nan_to_num(np.concatenate(exo, axis=1), nan=0.0).astype(np.float32) if exo else None
        )

    def _optional_field(self, name: str, rows: slice) -> NDArray[np.float64] | None:
        if self.data is None or name not in self.data.fields:
            return None
        return np.asarray(self.data.field(name)[rows], dtype=np.float64)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _assign_slots(self, row: int) -> None:
        tradable = np.flatnonzero(self._valid[row])
        if not self._dynamic_slots:  # one slot per asset (any extra slots stay empty)
            cols = np.full(self.max_assets, -1, dtype=np.intp)
            cols[: self.n_total] = np.arange(self.n_total)
            self._slot_cols = cols
            return
        incumbents = [int(c) for c in tradable if self._slots.slot_of(int(c)) is not None]
        newcomers = [int(c) for c in tradable if self._slots.slot_of(int(c)) is None]
        present = incumbents + newcomers[: self.max_assets - len(incumbents)]
        self._slots.update(present)
        cols = np.full(self.max_assets, -1, dtype=np.intp)
        for slot, asset in enumerate(self._slots.assets):
            if asset is not None:
                cols[slot] = int(asset)
        changed = self._slots.changed
        self.weights[changed] = 0.0  # a slot that changed hands holds nothing yet
        self._slot_cols = cols

    @property
    def slot_assets(self) -> tuple[Any, ...]:
        """The asset in each slot (MarketData asset ids, or column indices), None if empty."""
        names = self.data.assets if self.data is not None else tuple(range(self.n_total))
        return tuple(None if c < 0 else names[c] for c in self._slot_cols)

    def _tradable(self, row: int) -> NDArray[np.bool_]:
        cols = self._slot_cols
        ok = cols >= 0
        out = np.zeros(self.max_assets, dtype=bool)
        out[ok] = self._valid[row, cols[ok]]
        return out

    def action_mask(self) -> NDArray[np.int8]:
        """1 for slots that can be traded now (and for cash, in the long-only action)."""
        mask = self._tradable(self.lookback + self._t).astype(np.int8)
        return mask if self.config.allow_short else np.append(mask, np.int8(1))

    # ------------------------------------------------------------------
    # Env API
    # ------------------------------------------------------------------

    @property
    def episode_length(self) -> int:
        """Decisions per episode (read-only: it sizes the episode's data window)."""
        return self._episode_length

    @episode_length.setter
    def episode_length(self, value: int) -> None:
        raise AttributeError("episode_length is fixed at construction; make a new env.")

    @property
    def lookback(self) -> int:
        """Rows in each observation window (read-only: it sizes the observation)."""
        return self._lookback

    @lookback.setter
    def lookback(self, value: int) -> None:
        raise AttributeError("lookback is fixed at construction; make a new env.")

    @property
    def elapsed_steps(self) -> int:
        return self._t

    def _decision_row(self) -> int:
        if not self._started:
            raise ResetRequired()
        return min(self.lookback + self._t, self._window - 1)

    @property
    def timestamp(self) -> Any:
        """The time of the current decision, as in `info["timestamp"]` (the step when simulated)."""
        row = self._decision_row()
        return self._times[row]

    @property
    def latent_state(self) -> dict[str, NDArray[np.float64]]:
        """The simulation's latent state at the current decision (e.g. `{"regime": ...}`), as
        `info["state"]` carries it with `privileged_state=True`; empty for historical data."""
        row = self._decision_row()
        return {k: np.asarray(v[row]).copy() for k, v in self._state.items()}

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[dict[str, Any]]:
        self._set_rng(seed)
        self.reward_fn.reset()
        self._slots.reset()
        self._load_episode(options)
        self._started = True
        self._over = False
        self._t = 0
        self.equity = 1.0
        self.weights = np.zeros(self.max_assets)
        self._assign_slots(self.lookback)
        info = self._info(turnover=0.0, cost=0.0, financing=0.0)
        return ResetResult(observation=self._get_obs(), info=info)

    def _cost_context(self, row: int) -> dict[str, Any]:
        """Per-ASSET context for the cost model (costs are computed in asset space)."""
        needs = self.cost_model.requires
        ctx: dict[str, Any] = {}
        if "equity" in needs:
            ctx["equity"] = self.capital * self.equity
        if "volatility" in needs:
            window = self._log_returns[row - self.lookback + 1 : row + 1]
            avail = self._available[row - self.lookback + 1 : row + 1]
            n = avail.sum(axis=0)
            mean = window.sum(axis=0) / np.maximum(n, 1)
            var = (np.where(avail, window - mean, 0.0) ** 2).sum(axis=0) / np.maximum(n, 1)
            # Fewer than two returns: no estimate (the cost model substitutes a typical one).
            ctx["volatility"] = np.where(n >= 2, np.sqrt(var), np.nan)
        if "volume" in needs and self._volume is not None:
            ctx["volume"] = self._volume[row] * self._prices[row]  # shares x price; NaN if missing
        if "spread" in needs and self._spread is not None:
            ctx["spread"] = self._spread[row]
        return ctx

    def _asset_trades(self, trades: NDArray[np.float64]) -> NDArray[np.float64]:
        """Slot weight changes mapped onto asset columns."""
        cols = self._slot_cols
        ok = cols >= 0
        out = np.zeros(self.n_total)
        out[cols[ok]] = trades[ok]
        return out

    def target_weights(self, action: Any) -> NDArray[np.float64]:
        """The per-slot weights an action asks for in the current state."""
        a = flat_action(action).astype(np.float64)
        if a.shape != self.action_space.shape or not np.all(np.isfinite(a)):
            raise InvalidAction(action, self.action_space)
        tradable = self._tradable(self.lookback + self._t)
        if self.config.allow_short:
            return project_weights(a, tradable, self.config)
        return project_weights(a[:-1], tradable, self.config, cash_score=a[-1])

    def step(self, action: NDArray[np.float32]) -> StepResult[dict[str, Any]]:
        if not self._started:
            raise ResetRequired()
        if self._over:
            raise episode_over()
        row = self.lookback + self._t
        target = self.target_weights(action)
        trades = target - self.weights
        turnover = float(np.abs(trades).sum())
        cols, ok = self._slot_cols, self._slot_cols >= 0
        cost = float(self.cost_model(self._asset_trades(trades), **self._cost_context(row)))

        # Returns over (row, row + 1]; delisting assets return delisting_return.
        safe = np.where(ok, cols, 0)
        now_valid = ok & self._valid[row, safe]
        next_valid = ok & self._valid[row + 1, safe]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = self._prices[row + 1, safe] / self._prices[row, safe] - 1.0
        returns = np.where(now_valid & next_valid, r, 0.0)
        returns = np.where(now_valid & ~next_valid, self.delisting_return, returns)

        s = settle(target, returns, cost=cost, dt=self.dt, config=self.config)
        growth = float(s.growth)
        drifted = np.where(now_valid & ~next_valid, 0.0, s.drifted)  # delisted -> cash
        previous_equity, self.equity = self.equity, max(self.equity * growth, 0.0)
        self.weights = drifted
        self._t += 1
        terminated = bool(s.wiped_out)
        truncated = not terminated and self._t >= self.episode_length
        self._over = terminated or truncated
        if not (terminated or truncated):
            self._assign_slots(self.lookback + self._t)
        outcome = StepOutcome.from_growth(
            growth,
            equity=self.equity,
            turnover=turnover,
            cost=cost,
            previous_equity=previous_equity,
        )
        reward = float(self.reward_fn(outcome))
        info = self._info(turnover=turnover, cost=cost, financing=float(s.financing))
        return StepResult(self._get_obs(), reward, terminated, truncated, info)

    # ------------------------------------------------------------------
    # Observation and info
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, Any]:
        row = min(self.lookback + self._t, self._window - 1)
        rows = slice(row - self.lookback + 1, row + 1)
        cols, ok = self._slot_cols, self._slot_cols >= 0
        window = self._features[rows][:, np.where(ok, cols, 0)]  # (L, S, F)
        values = np.ascontiguousarray(np.transpose(window, (1, 0, 2)))
        mask = (ok & self._valid[row, np.where(ok, cols, 0)]).astype(np.int8)
        values[mask == 0] = 0.0
        cash = 1.0 - self.weights.sum()
        obs: dict[str, Any] = {
            "assets": {"values": values, "mask": mask},
            "portfolio": np.append(self.weights, cash).astype(np.float32),
            "account": np.array(
                [np.log(max(self.equity, 1e-12)), np.abs(self.weights).sum()], dtype=np.float32
            ),
        }
        if self._exo is not None:
            obs["market"] = self._exo[rows].copy()
        return obs

    def _info(self, *, turnover: float, cost: float, financing: float) -> dict[str, Any]:
        row = self._decision_row()
        info: dict[str, Any] = {
            "equity": self.equity,
            "turnover": turnover,
            "cost": cost,
            "financing": financing,
            "gross_exposure": float(np.abs(self.weights).sum()),
            "n_assets": int(self._tradable(row).sum()),
            "weights": self.weights.copy(),
            "timestamp": self.timestamp,
            "dt": self.dt,
        }
        if self.privileged_state:
            info["state"] = self.latent_state
        return info


__all__ = ["MarketEnv", "default_market"]
