"""`BatchMarketEnv`: the general finance environment, batched, on NumPy or torch.

One implementation serves every way of running it: `Market-v1` as a single env
(`sekai.make`), a native vector env (`sekai.make_vec`), and a batch of lanes on a
GPU whose rewards are differentiable in the actions, for pathwise policy
optimisation. It is assembled from `sekai.finance` components:

- data: a `MarketData` history (episodes are windows of it), a market model
  (`MarketModel`), or any `ScenarioGenerator`, e.g. a generative model of markets
  (a fresh scenario per episode);
- a changing universe: assets list and delist, and are mapped onto `max_assets`
  stable slots (`MaskedBox` observations);
- an `ActionMap` (long-only scores, target weights, or scores and leverage
  projected onto hard `Constraints`), an `ObservationBuilder`, a `CostModel` and
  `HoldingCost`, a `RewardFunction` and an optional `FactorRiskModel`;
- explicit execution timing (`execution=`): when a decision made on the data at time
  t is filled.

Execution:

- `"close"`: at the close of t, the price just observed. The convention of
  `Market-v0`, and of simulated MDPs; on real data it is optimistic (no order can
  fill at the close that has already been observed), and agents learn to exploit it.
- `"next_close"` (default): at the close of t + 1. The order is pending for one
  period, during which the previous book is held; the observation includes the
  pending order, so it stays Markov. Conservative, and needs no extra data.
- `"next_open"` / `"next_vwap"`: at the next period's open (or VWAP), with the data
  fields `"open"` (or `"vwap"`) and the close prices (`MarketData`'s price field, the
  `"close"` field of a panel). The previous book earns the overnight move, the new
  one the move from the fill to the close. The realistic choice for daily data that
  has opening prices.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai.backend import Array, Backend, get_backend
from sekai.backend._base import _Check, _collect_checks
from sekai.envs.finance._common import check_number
from sekai.envs.finance.market_env import default_market
from sekai.error import InvalidAction
from sekai.finance.account import AccountConfig, settle
from sekai.finance.actions import ActionContext, ActionMap, LongOnlyScores, TargetWeights
from sekai.finance.costs import CostModel, HoldingCost, as_cost_model
from sekai.finance.data import MarketData
from sekai.finance.observations import MarketObservation, MarketView, ObservationBuilder
from sekai.finance.panel import MarketPanel, ModelScenarios, ScenarioGenerator
from sekai.finance.rewards import LogReturn, RewardFunction, StepOutcome
from sekai.finance.risk import FactorRiskModel, factor_exposures, portfolio_variance
from sekai.finance.sampling import EpisodeSampler, as_sampler, resolve_start, sample_start
from sekai.finance.simulators import MarketModel
from sekai.vector.batch import BatchEnv, BatchVecEnv, UnbatchedEnv
from sekai.vector.native import shared

EXECUTIONS = ("close", "next_close", "next_open", "next_vwap")
_FILL_FIELD = {"next_open": "open", "next_vwap": "vwap"}


class BatchMarketEnv(BatchEnv[dict[str, Any], Any]):
    """Multi-asset trading over a changing universe, batched over `num_envs` lanes.

    Each lane is an independent account trading its own episode: a window of a
    history, or a fresh scenario. At each decision the lane's action becomes target
    weights (`action`), the account trades to them when the order fills
    (`execution`), pays `transaction_cost`, then holds them over the period: assets
    earn their returns, cash earns `cash_rate` (or pays `margin_rate` when borrowed),
    shorts pay `borrow_rate`, and `holding_cost` charges anything else (per-name
    borrow fees, financing on gross). An asset that delists returns
    `delisting_return` and its position becomes cash.

    **Observation**: built by `observation` (default `MarketObservation(features,
    exogenous)`, the layout of `Market-v0` plus the pending order with
    `execution="next_close"`). **Reward**: `reward(outcome)` (default `LogReturn()`).
    **Termination**: equity wiped out, or a drawdown of `max_drawdown`. **Truncation**:
    after `episode_length` decisions.

    **info** (arrays with a leading `num_envs` axis): `equity`, `return` (the step's net
    simple return, after `step()`), `turnover`, `cost`,
    `holding_cost`, `financing`, `gross_exposure`, `net_exposure`, `drawdown`,
    `n_assets` (tradable), `weights` (per slot, after the step), `timestamp`, `dt`;
    `variance` and `exposures` with a risk model; `state` (the simulation's latent
    state) with `privileged_state=True`.

    Without autoreset, a lane that finishes is frozen until it is reset (see
    `BatchEnv`). With the torch backend and `differentiable = True`, the rewards are
    differentiable in the actions through the action map, costs, accounting and
    reward; `detach_state()` truncates the graph.

    Args:
        num_envs: Number of lanes.
        data: A history (NaN prices = not listed); episodes are windows of it.
        scenarios: A `ScenarioGenerator`; each episode is a fresh path (its first
            `lookback + episode_length + 1` times).
        model: A `MarketModel` to simulate (default, when no data or scenarios are
            given: `Market-v0`'s regime-switching market with listings).
        simulate_on_device: Simulate the model's paths with torch on `device`
            (`MarketModel.simulate_torch`) instead of with NumPy on the host, so that a
            reset copies no data to the device (`backend="torch"` only; a seed then gives
            other paths).
        max_assets: Slots per lane (default: every asset).
        features: Per-asset data fields to observe (default observation builder).
        exogenous: Market-wide series to observe (default observation builder); with
            a model, names of its state series (e.g. "regime").
        lookback: Rows in each observation window.
        episode_length: Decisions per episode.
        sampler: Placement of episodes in a history ("random", "sequential", an int,
            or an `EpisodeSampler`); each lane gets `sampler.spawn()`.
        execution: When orders fill: "close", "next_close", "next_open", "next_vwap".
        action: An `ActionMap` (default: `LongOnlyScores(max_leverage)`, or
            `TargetWeights(max_leverage)` with `allow_short`).
        observation: An `ObservationBuilder` (replaces `features` / `exogenous`).
        transaction_cost: Proportional rate or `CostModel`. Its context: `volume`
            (traded value: the data field `"dollar_volume"`, else `"volume"` shares
            times the close), `spread`, `volatility` (of the observed window) and
            `equity` (`capital` times the equity).
        holding_cost: A `HoldingCost` charged every period (e.g. `BorrowFee()` from
            the `"borrow_rate"` field, `GrossFinancing(rate)`).
        reward: Reward function (default `LogReturn()`; the env keeps a copy).
        risk: A `FactorRiskModel` (adds the ex-ante `variance` to the outcome and info).
        shortable: Name of a per-asset field, nonzero where the asset may be shorted
            (default: every asset may be).
        allow_short: Long/short default action (`TargetWeights`).
        max_leverage: Gross exposure cap of the default action.
        cash_rate: Annual interest earned on positive cash.
        margin_rate: Annual interest paid on borrowed cash (default: cash_rate).
        borrow_rate: Annual fee on the value of short positions (every asset).
        capital: Account size in currency (for fixed and impact costs).
        dt: Years per step (default: from the data's timestamps, the model, or 1/252).
        delisting_return: Return of an asset over the period in which it delists.
        max_drawdown: Terminate a lane when its drawdown reaches this (in (0, 1]).
        flat_overnight: Hold no position from one trading session into the next: at the
            last decision of a session the book is sold to cash at that close (costs
            charged). Sessions come from `MarketData.sessions()` (calendar dates of
            datetime timestamps, or a `session` series), or a panel's `session` series.
            Needs `execution="close"` or `"next_close"`.
        initial_margin: Equity required per unit of gross exposure to open positions (e.g.
            0.5: at most 2x gross): orders above `1 / initial_margin` gross are scaled down.
        maintenance_margin: Equity required per unit of gross exposure to keep positions
            (e.g. 0.25): when price moves push gross exposure above
            `1 / maintenance_margin`, the broker cuts every position in proportion, at that
            close and at the cost of the trade, back to `1 / initial_margin` (default
            initial margin: twice the maintenance margin, at most 1). `info["margin_call"]`
            flags it.
        privileged_state: Add the simulation's latent state to info.
        backend: "numpy" or "torch".
        device: torch device.
        dtype: torch float dtype of the data and state (default: torch's default).
        autoreset: Reset finished lanes within `step()`.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 10}
    # A scenario generator draws from the seed it is given, so snapshots share it (a
    # generative model, perhaps on a GPU and still training, is never copied).
    _snapshot_shared = ("scenarios",)

    def __init__(
        self,
        num_envs: int = 1,
        data: MarketData | None = None,
        *,
        scenarios: ScenarioGenerator | None = None,
        model: MarketModel | None = None,
        simulate_on_device: bool = False,
        max_assets: int | None = None,
        features: Sequence[str] = (),
        exogenous: Sequence[str] = (),
        lookback: int = 20,
        episode_length: int = 252,
        sampler: str | int | EpisodeSampler = "random",
        execution: str = "next_close",
        action: ActionMap | None = None,
        observation: ObservationBuilder | None = None,
        transaction_cost: float | CostModel = 1e-3,
        holding_cost: HoldingCost | None = None,
        reward: RewardFunction | None = None,
        risk: FactorRiskModel | None = None,
        shortable: str | None = None,
        allow_short: bool = False,
        max_leverage: float = 1.0,
        cash_rate: float = 0.0,
        margin_rate: float | None = None,
        borrow_rate: float = 0.0,
        capital: float = 1_000_000.0,
        dt: float | None = None,
        delisting_return: float = 0.0,
        max_drawdown: float | None = None,
        flat_overnight: bool = False,
        initial_margin: float | None = None,
        maintenance_margin: float | None = None,
        privileged_state: bool = False,
        backend: str | Backend = "numpy",
        device: Any = None,
        dtype: Any = None,
        autoreset: bool = True,
    ) -> None:
        sources = [
            name
            for name, v in (("data", data), ("scenarios", scenarios), ("model", model))
            if v is not None
        ]
        if len(sources) > 1:
            raise ValueError(
                f"Give one of data=, scenarios= or model=, not {' and '.join(sources)}."
            )
        if data is not None and not isinstance(data, MarketData):
            raise TypeError(f"data must be a MarketData, got {type(data).__name__}.")
        if not sources:
            model = default_market()
        if lookback < 1 or episode_length < 1:
            raise ValueError("lookback and episode_length must be >= 1.")
        if execution not in EXECUTIONS:
            raise ValueError(f"execution must be one of {EXECUTIONS}, got {execution!r}.")
        if observation is not None and (features or exogenous):
            raise ValueError(
                "features= and exogenous= configure the default observation; with "
                "observation= list the fields in the builder instead."
            )
        if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
            raise ValueError(f"max_drawdown must be in (0, 1], got {max_drawdown!r}.")
        if flat_overnight and execution not in ("close", "next_close"):
            raise ValueError(
                "flat_overnight needs execution='close' or 'next_close' (with next-open or "
                "VWAP fills, positions are held overnight by construction)."
            )
        if maintenance_margin is not None and initial_margin is None:
            initial_margin = min(1.0, 2.0 * maintenance_margin)
        for name, value in (
            ("initial_margin", initial_margin),
            ("maintenance_margin", maintenance_margin),
        ):
            if value is not None and not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}.")
        if (
            initial_margin is not None
            and maintenance_margin is not None
            and maintenance_margin > initial_margin
        ):
            raise ValueError("maintenance_margin cannot exceed initial_margin.")
        B = get_backend(backend)
        if simulate_on_device and model is None:
            raise ValueError(
                "simulate_on_device=True simulates a market model (model=, or the default "
                f"market) on the device; it does not apply to {sources[0]}=."
            )
        if simulate_on_device and B.name != "torch":
            raise ValueError(
                "simulate_on_device=True needs backend='torch' (the NumPy backend simulates "
                "on the host)."
            )
        self._lookback = int(lookback)
        self._episode_length = int(episode_length)
        self._window = self._lookback + self._episode_length + 1
        self.execution = execution
        self._features = tuple(features)
        self._exogenous = tuple(exogenous)
        self.builder = (
            observation
            if observation is not None
            else MarketObservation(self._features, self._exogenous)
        )
        self.action_map = (
            action
            if action is not None
            else (TargetWeights(max_leverage) if allow_short else LongOnlyScores(max_leverage))
        )
        self.cost_model = as_cost_model(transaction_cost)
        needs_window = lookback < 2 and "volatility" in self.cost_model.requires
        if (
            needs_window
            and scenarios is None
            and not (data is not None and "volatility" in data.fields)
        ):
            raise ValueError(
                f"{self.cost_model!r} needs a per-asset volatility estimate, which takes at least "
                f"2 returns, but lookback={lookback} observes only {lookback}; every trade would "
                "be unpriced. Use lookback >= 2."
            )
        self._cost_takes_assets = _accepts(self.cost_model, "assets")
        self.holding_cost = holding_cost
        self.reward_fn = reward if reward is not None else LogReturn()
        self.risk = risk
        self.shortable = shortable
        self.config = AccountConfig(
            allow_short=True,
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
        self.max_drawdown = max_drawdown
        self.flat_overnight = flat_overnight
        self.initial_margin = initial_margin
        self.maintenance_margin = maintenance_margin
        self.privileged_state = privileged_state
        self.data = data
        self.scenarios: ScenarioGenerator | None = None
        self.sampler = as_sampler(sampler)

        asset_fields, market_fields = self._needed_fields(data)
        self._dtype = dtype
        if data is not None:
            for name in asset_fields:
                if name != data.price_field:
                    data.field(name)  # KeyError for a missing field
            for name in market_fields:
                data.exogenous_series(name)
            if data.n_steps < self._window:
                raise ValueError(
                    f"The data has {data.n_steps} rows but an episode needs {self._window} "
                    "(lookback + episode_length + 1)."
                )
            self.scenarios = None
            panel = MarketPanel.from_market_data(
                data,
                fields=sorted(asset_fields),
                exogenous=sorted(market_fields),
                backend=B,
                device=device,
                dtype=dtype,
            )
            if flat_overnight:
                sessions = np.asarray(data.sessions(), dtype=np.float64)[None]
                panel = MarketPanel(
                    returns=panel.returns,
                    valid=panel.valid,
                    fields=panel.fields,
                    exogenous={**panel.exogenous, "session": B.asfloat(sessions, panel.returns)},
                    timestamps=panel.timestamps,
                    assets=panel.assets,
                    corporate_actions=panel.corporate_actions,
                )
            ppy = data.periods_per_year
            self.dt = dt if dt is not None else (1.0 / ppy if ppy else 1 / 252)
        else:
            if model is not None:
                if simulate_on_device:  # on the env's device (BatchEnv's default: the CPU)
                    where = "cpu" if device is None else device
                    scenarios = ModelScenarios(model, self._window - 1, device=where, dtype=dtype)
                else:
                    scenarios = ModelScenarios(model, self._window - 1)
                self.dt = dt if dt is not None else model.dt
            else:
                self.dt = dt if dt is not None else 1 / 252
            assert scenarios is not None
            self.scenarios = scenarios
            panel = self._sample(1, seed=0, backend=B, device=device)  # shapes for the spaces
            if needs_window and "volatility" not in panel.fields:
                raise ValueError(
                    f"{self.cost_model!r} needs a per-asset volatility: give the scenarios a "
                    "'volatility' field, or use lookback >= 2 to estimate it from returns."
                )
        self.dt = check_number("dt", self.dt, ">0")
        self.model = model
        self._asset_fields, self._market_fields = asset_fields, market_fields
        self.n_total = panel.n_assets
        self.max_assets = self.n_total if max_assets is None else int(max_assets)
        if self.max_assets < 1:
            raise ValueError("max_assets must be >= 1.")
        self._dynamic = self.max_assets < self.n_total
        self._field_shapes = {k: tuple(v.shape[3:]) for k, v in panel.fields.items()}
        self._market_shapes = {k: tuple(v.shape[2:]) for k, v in panel.exogenous.items()}
        self._panel = panel
        super().__init__(
            num_envs,
            self.builder.space(self),
            self.action_map.space(self.max_assets),
            backend=B,
            device=device,
            autoreset=autoreset,
        )
        self._like = self.backend.zeros((), like=self._panel.returns)
        # Whether a step reads its checks back together (see `_checkpoint`); None: when
        # the env is on a device. (Tests set it to run that path on the CPU.)
        self._defer: bool | None = None
        self._init_state()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _needed_fields(self, data: MarketData | None) -> tuple[set[str], set[str]]:
        """The per-asset fields and market series every component reads."""
        assets = set(self.builder.fields) | set(self.action_map.fields)
        market = set(self.builder.market_fields)
        close = data.price_field if data is not None else "close"
        self._close_field = close
        needs = self.cost_model.requires
        if "spread" in needs:
            assets.add("spread")
        if "volatility" in needs and data is not None and "volatility" in data.fields:
            assets.add("volatility")
        if "volume" in needs:
            has_dollar = data is not None and "dollar_volume" in data.fields
            assets |= {"dollar_volume"} if has_dollar or data is None else {"volume", close}
        if self.holding_cost is not None and "borrow_rate" in self.holding_cost.requires:
            assets.add("borrow_rate")
        if self.risk is not None:
            assets |= set(self.risk.asset_fields)
            market |= set(self.risk.market_fields)
        if self.shortable is not None:
            assets.add(self.shortable)
        if self.execution in _FILL_FIELD:
            assets |= {_FILL_FIELD[self.execution], close}
            # Corporate actions on the fill day adjust the overnight move (if the data has them).
            if data is not None and data.has_corporate_actions:
                assets |= {"split", "dividend"} & set(data.fields)
        if self.flat_overnight and data is None:
            market.add("session")  # from MarketData, computed from the timestamps
        return assets, market

    def _sample(
        self, n: int, *, seed: int, backend: Backend | None = None, device: Any = None
    ) -> MarketPanel:
        """`n` fresh paths from the scenario generator, cut to one episode's window."""
        assert self.scenarios is not None
        B = backend if backend is not None else self.backend
        dev = device if device is not None else getattr(self, "device", None)
        panel = self.scenarios.sample(n, seed=seed)
        if not isinstance(panel, MarketPanel):
            raise TypeError(
                f"{self.scenarios!r}.sample() must return a MarketPanel, "
                f"got {type(panel).__name__}."
            )
        if panel.n_paths != n:
            raise ValueError(f"{self.scenarios!r} returned {panel.n_paths} paths for n_paths={n}.")
        if panel.n_steps < self._window:
            raise ValueError(
                f"{self.scenarios!r} returned paths of {panel.n_steps} times, but an episode needs "
                f"{self._window} (lookback + episode_length + 1)."
            )
        asset_fields, market_fields = (
            getattr(self, "_asset_fields", None),
            getattr(self, "_market_fields", None),
        )
        if asset_fields is None:
            asset_fields, market_fields = self._needed_fields(None)
        missing = sorted(set(asset_fields) - set(panel.fields))
        missing_market = sorted(set(market_fields or ()) - set(panel.exogenous))
        if missing or missing_market:
            raise KeyError(
                f"The scenarios lack the fields {missing} and market series {missing_market} "
                "that this environment's components read."
            )
        W = self._window
        cut = MarketPanel(
            returns=panel.returns[:, :W],
            valid=panel.valid[:, :W],
            fields={
                k: v[:, :W]
                for k, v in panel.fields.items()
                if k in asset_fields
                or (
                    k in ("split", "dividend")
                    and panel.corporate_actions
                    and self.execution in _FILL_FIELD
                )
                or (k == "volatility" and "volatility" in self.cost_model.requires)
            },
            exogenous={k: v[:, :W] for k, v in panel.exogenous.items()},
            timestamps=None if panel.timestamps is None else np.asarray(panel.timestamps)[..., :W],
            assets=panel.assets,
            state={k: v[:, :W] for k, v in panel.state.items()},
            corporate_actions=panel.corporate_actions,
        )
        return cut.to(B, device=dev, dtype=self._dtype)

    def _init_state(self) -> None:
        B, like = self.backend, self._like
        n, S = self.num_envs, self.max_assets
        # Each lane's path, episode start and decisions taken, on the host and (the `_dev`
        # copies) as backend arrays, so a step on a device never copies them across.
        history = self.data is not None
        self._paths_host = np.zeros(n, dtype=np.int64) if history else np.arange(n, dtype=np.int64)
        self._paths = B.ints((n,), 0, like=like) if history else B.arange(n, like=like)
        self._start = np.zeros(n, dtype=np.int64)
        self._t = np.zeros(n, dtype=np.int64)
        self._start_dev = B.ints((n,), 0, like=like)
        self._t_dev = B.ints((n,), 0, like=like)
        self._samplers = [self.sampler.spawn() for _ in range(n)]
        self._slots = B.ints((n, S), -1, like=like)
        self._weights = B.zeros((n, S), like=like)
        self._equity = B.full((n,), 1.0, like=like)
        self._peak = B.full((n,), 1.0, like=like)
        self._pending = B.zeros((n, S), like=like) if self.execution == "next_close" else None

    # ------------------------------------------------------------------
    # Introspection (used by observation builders, renderers and users)
    # ------------------------------------------------------------------

    @property
    def lookback(self) -> int:
        """Rows in each observation window (fixed at construction)."""
        return self._lookback

    @property
    def episode_length(self) -> int:
        """Decisions per episode (fixed at construction)."""
        return self._episode_length

    @property
    def n_slots(self) -> int:
        """Asset slots per lane (`max_assets`)."""
        return self.max_assets

    @property
    @shared
    def features(self) -> tuple[str, ...]:
        """The per-asset fields given as `features=` (shown by the default observation)."""
        return self._features

    @property
    @shared
    def exogenous(self) -> tuple[str, ...]:
        """The market-wide series given as `exogenous=` (shown by the default observation)."""
        return self._exogenous

    @shared
    def field_shape(self, name: str) -> tuple[int, ...]:
        """The trailing shape of a per-asset field (() for one value per asset and time)."""
        return self._field_shapes[name]

    @shared
    def market_shape(self, name: str) -> tuple[int, ...]:
        """The trailing shape of a market-wide series."""
        return self._market_shapes[name]

    @property
    def panel(self) -> MarketPanel:
        """The data the lanes are trading (one path for a history, one per lane otherwise)."""
        return self._panel

    @property
    def weights(self) -> Array:
        """`(num_envs, S)` current weights per slot."""
        return self._weights

    @property
    def equity(self) -> Array:
        """`(num_envs,)` equity relative to the start of each lane's episode."""
        return self._equity

    @property
    def elapsed_steps(self) -> NDArray[np.int64]:
        """Decisions taken in each lane's episode."""
        out: NDArray[np.int64] = self._t.copy()
        return out

    def _rows(self, room: int = 0) -> Array:
        """Each lane's decision time in the panel, at most `room` rows before its window ends."""
        rows = self._start_dev + self._lookback + self._t_dev
        return self.backend.minimum(rows, self._start_dev + (self._window - 1 - room))

    def _cols(self) -> tuple[Array, Array]:
        occ = self._slots >= 0
        return self.backend.where(occ, self._slots, 0), occ

    def _asset_at(self, arr: Array, rows: Array) -> Array:
        cols, _ = self._cols()
        return arr[self._paths[:, None], rows[:, None], cols]

    def _tradable(self, rows: Array) -> Array:
        cols, occ = self._cols()
        return occ & self._panel.valid[self._paths[:, None], rows[:, None], cols]

    def action_mask(self) -> Array:
        """`(num_envs, A)` int8: 1 for the slots that can be traded now (and for any extra
        action entries, such as cash or a leverage logit)."""
        B = self.backend
        mask = B.asint8(self._tradable(self._rows()))
        extra = int(self.action_space.shape[0]) - self.max_assets  # type: ignore[index]
        if extra > 0:
            mask = B.concat(
                [mask, B.asint8(B.bools((self.num_envs, extra), True, like=self._like))], axis=-1
            )
        return mask

    @property
    def slot_assets(self) -> NDArray[np.object_]:
        """`(num_envs,)` tuples: the asset (id, or column) in each slot, None if empty."""
        slots = self.backend.to_numpy(self._slots)
        names = self._panel.assets if self._panel.assets is not None else tuple(range(self.n_total))
        out = np.empty(self.num_envs, dtype=object)
        for i, row in enumerate(slots):
            out[i] = tuple(None if c < 0 else names[c] for c in row)
        return out

    @property
    def timestamp(self) -> NDArray[Any]:
        """`(num_envs,)` time of each lane's current decision (the step index when
        the data has no timestamps)."""
        rows = np.minimum(self._start + self._lookback + self._t, self._start + self._window - 1)
        ts = self._panel.timestamps
        if ts is None:
            return np.asarray(rows - self._start - self._lookback)
        ts = np.asarray(ts)
        if ts.ndim == 1:
            return np.asarray(ts[rows])
        return np.asarray(ts[self._paths_host, rows])

    @property
    def latent_state(self) -> dict[str, Array]:
        """The simulation's latent state at each lane's decision (empty for a history)."""
        rows = self._rows()
        return {k: v[self._paths, rows] for k, v in self._panel.state.items()}

    @property
    def reward_fn(self) -> RewardFunction:
        """The reward function; setting it stores a copy (as `reward=` does)."""
        return self._reward_fn

    @reward_fn.setter
    def reward_fn(self, value: RewardFunction) -> None:
        self._reward_fn = copy.deepcopy(value)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _assign_slots(self, lanes: Array, rows: Array) -> None:
        """Map tradable assets onto slots for `lanes` at `rows`: incumbents keep their
        slot, newcomers take the lowest free slots in column order."""
        B, S, N = self.backend, self.max_assets, self.n_total
        old = self._slots
        if not self._dynamic:
            static = B.where(B.arange(S, like=self._like) < N, B.arange(S, like=self._like), -1)
            new = B.where(lanes[:, None], B.broadcast_to(static, (self.num_envs, S)), old)
        else:
            valid = self._panel.valid[self._paths, rows]  # (n, N)
            occ = old >= 0
            cols = B.where(occ, old, 0)
            keep = occ & B.take_along_axis(valid, cols, axis=1)
            kept = B.where(keep, old, -1)
            held = B.put_along_axis(
                B.bools((self.num_envs, N + 1), False, like=self._like),
                B.where(keep, old, N),
                B.bools((self.num_envs, S), True, like=self._like),
                axis=1,
            )[:, :N]
            newcomer = valid & ~held
            free = kept < 0
            free_order = B.argsort(B.asint(~free, self._like), axis=1)
            new_order = B.argsort(B.asint(~newcomer, self._like), axis=1)[:, :S]
            n_fill = B.minimum(B.sum(free, axis=1), B.sum(newcomer, axis=1))
            fill = B.arange(S, like=self._like)[None, :] < n_fill[:, None]
            current = B.take_along_axis(kept, free_order, axis=1)
            placed = B.put_along_axis(kept, free_order, B.where(fill, new_order, current), axis=1)
            new = B.where(lanes[:, None], placed, old)
        changed = new != old
        self._weights = B.where(changed, 0.0, self._weights)
        if self._pending is not None:
            self._pending = B.where(changed, 0.0, self._pending)
        self._slots = new

    # ------------------------------------------------------------------
    # BatchEnv hooks
    # ------------------------------------------------------------------

    def _reset_lanes(
        self, indices: Array, options: Sequence[dict[str, Any] | None]
    ) -> dict[str, Any]:
        B, like = self.backend, self._like
        idx = self._lanes_on_host(indices)
        n = self.num_envs
        everyone = len(idx) == n and np.array_equal(np.sort(idx), np.arange(n))
        if self.data is not None:
            n_starts = self.data.n_steps - self._window + 1
            for lane, opts in zip(idx.tolist(), options):
                if opts and opts.get("start") is not None:
                    start = resolve_start(
                        opts["start"],
                        timestamps=self.data.timestamps,
                        lookback=self._lookback,
                        n_starts=n_starts,
                        warmup=self.data.warmup,
                    )
                else:
                    start = sample_start(
                        self._samplers[lane],
                        self.rng,
                        n_starts=n_starts,
                        episode_length=self._episode_length,
                        lookback=self._lookback,
                        warmup=self.data.warmup,
                    )
                self._start[lane] = start
        else:
            if any(opts and opts.get("start") is not None for opts in options):
                raise ValueError("options['start'] needs historical data (data=...).")
            fresh = self._sample(len(idx), seed=int(self.rng.integers(2**31 - 1)))
            if everyone and np.array_equal(idx, np.arange(n)):
                self._panel = fresh
            else:
                self._write_paths(idx, indices, fresh)
            self._start[idx] = 0
        self._t[idx] = 0
        lanes = B.put_along_axis(
            B.bools((n,), False, like=like),
            indices,
            B.bools(tuple(indices.shape), True, like=like),
            0,
        )
        self._t_dev = B.where(lanes, 0, self._t_dev)
        if self.data is not None:  # new starts, drawn on the host
            self._start_dev = B.asint(self._start, like)
        else:
            self._start_dev = B.where(lanes, 0, self._start_dev)
        self._weights = B.where(lanes[:, None], 0.0, self._weights)
        self._equity = B.where(lanes, 1.0, self._equity)
        self._peak = B.where(lanes, 1.0, self._peak)
        if self._pending is not None:
            self._pending = B.where(lanes[:, None], 0.0, self._pending)
        self._slots = B.where(lanes[:, None], -1, self._slots)
        self._assign_slots(lanes, self._rows())
        self.reward_fn.reset(None if everyone else lanes)
        zero = B.zeros((n,), like=like)
        info = self._info(turnover=zero, cost=zero, holding=zero, financing=zero, exec_weights=None)
        return {k: _take(v, idx, indices, B) for k, v in info.items()}

    def _write_paths(self, idx: NDArray[np.int64], indices: Array, fresh: MarketPanel) -> None:
        """Replace the paths of lanes `idx` (`indices` on the backend) in new arrays: a
        panel never changes once built, so snapshots and observations already returned
        can share it."""
        B, panel = self.backend, self._panel

        def put(arr: Array, new: Array) -> Array:
            out = B.copy(arr)
            out[indices] = new
            return out

        timestamps = panel.timestamps
        if timestamps is not None and fresh.timestamps is not None and np.ndim(timestamps) == 2:
            timestamps = np.array(timestamps, copy=True)
            timestamps[idx] = np.asarray(fresh.timestamps)
        self._panel = MarketPanel(
            returns=put(panel.returns, fresh.returns),
            valid=put(panel.valid, fresh.valid),
            fields={k: put(v, fresh.fields[k]) for k, v in panel.fields.items()},
            exogenous={k: put(v, fresh.exogenous[k]) for k, v in panel.exogenous.items()},
            timestamps=timestamps,
            assets=panel.assets,
            state={k: put(v, fresh.state[k]) for k, v in panel.state.items()},
            corporate_actions=panel.corporate_actions,
        )

    def _observe(self) -> dict[str, Any]:
        view = MarketView(
            backend=self.backend,
            panel=self._panel,
            paths=self._paths,
            rows=self._rows(),
            slots=self._slots,
            lookback=self._lookback,
            episode_length=self._episode_length,
            elapsed=self._t.copy(),
            weights=self._weights,
            equity=self._equity,
            peak=self._peak,
            pending=self._pending,
        )
        return self.builder(view)  # type: ignore[no-any-return]

    def _context(self, rows: Array, decision: Array, needs: frozenset[str]) -> dict[str, Any]:
        """Inputs of cost and holding models at the execution `rows`, per slot."""
        return cost_context(
            backend=self.backend,
            panel=self._panel,
            paths=self._paths,
            slots=self._slots,
            rows=rows,
            decision=decision,
            lookback=self._lookback,
            needs=needs,
            equity=self.capital * self._equity,
            close_field=self._close_field,
        )

    def _exposures(self, names: Sequence[str], rows: Array) -> dict[str, Array]:
        B = self.backend
        _, occ = self._cols()
        out = {}
        for name in names:
            values = self._asset_at(self._panel.fields[name], rows)
            mask = occ.reshape(tuple(occ.shape) + (1,) * (values.ndim - 2))
            out[name] = B.where(mask & B.isfinite(values), values, 0.0)
        return out

    def _step(self, actions: Any) -> tuple[Array, Array, Array, dict[str, Any]]:
        B, like, panel = self.backend, self._like, self._panel
        n = self.num_envs
        a = B.cast(actions, like)
        shape = (n, *self.action_space.shape)  # type: ignore[misc]
        if tuple(a.shape) != shape:
            raise InvalidAction(actions, self.action_space)
        # On a device, the checks of the action, of its projection and of the cost models'
        # inputs are read back once, with `finished`, before the step changes any state
        # (see `_checkpoint`).
        deferred = self._deferring()
        finite = B.all(B.isfinite(a))
        if not deferred and not B.item(finite):
            raise InvalidAction(actions, self.action_space)
        checks: list[_Check] = []
        collect: Callable[[], contextlib.AbstractContextManager[Any]] = (
            functools.partial(_collect_checks, checks) if deferred else contextlib.nullcontext
        )
        frozen = self.frozen
        r = self._rows(room=1)  # decision rows; frozen lanes stay inside their window
        nxt = r + 1
        cols, occ = self._cols()
        p2 = self._paths[:, None]
        valid_now = occ & panel.valid[p2, r[:, None], cols]
        valid_next = occ & panel.valid[p2, nxt[:, None], cols]
        shortable = None
        if self.shortable is not None:
            flag = self._asset_at(panel.fields[self.shortable], r)
            shortable = B.isfinite(flag) & (flag != 0)
        ctx = ActionContext(
            backend=B,
            tradable=valid_now,
            weights=self._weights,
            shortable=shortable,
            exposures=self._exposures(self.action_map.fields, r)
            if self.action_map.fields
            else None,
        )
        with collect():
            target = self.action_map(a, ctx)
        w = self._weights
        delisting = valid_now & ~valid_next
        with B.errstate(), collect():
            if self.execution in ("close", "next_close"):
                held = (
                    target if self.execution == "close" else B.where(valid_now, self._pending, 0.0)
                )
                start = w
                exec_rows = r
                ret = B.where(valid_now & valid_next, panel.returns[p2, nxt[:, None], cols], 0.0)
                ret = B.where(delisting, self.delisting_return, ret)
                overnight = None
            else:
                fill = panel.fields[_FILL_FIELD[self.execution]]
                close = panel.fields[self._close_field]
                both = valid_now & valid_next
                fill_next = fill[p2, nxt[:, None], cols]
                close_now = close[p2, r[:, None], cols]
                close_next = close[p2, nxt[:, None], cols]
                priced = both & B.isfinite(fill_next) & B.isfinite(close_now)
                value_next = fill_next  # a share's value at the fill, dividends and splits in
                if panel.corporate_actions and "dividend" in panel.fields:
                    cash = panel.fields["dividend"][p2, nxt[:, None], cols]
                    value_next = value_next + B.where(B.isfinite(cash), cash, 0.0)
                if panel.corporate_actions and "split" in panel.fields:
                    ratio = panel.fields["split"][p2, nxt[:, None], cols]
                    value_next = value_next * B.where(B.isfinite(ratio), ratio, 1.0)
                move = B.where(priced, value_next / B.where(priced, close_now, 1.0) - 1.0, 0.0)
                move = B.where(delisting, self.delisting_return, move)
                overnight = 1.0 + B.sum(w * move, axis=-1)
                alive = overnight > 0
                start = B.where(
                    alive[:, None] & ~delisting,
                    w * (1.0 + move) / B.where(alive, overnight, 1.0)[:, None],
                    0.0,
                )
                held = B.where(valid_next, target, 0.0)
                exec_rows = nxt
                inside = valid_next & B.isfinite(fill_next) & B.isfinite(close_next)
                ret = B.where(inside, close_next / B.where(inside, fill_next, 1.0) - 1.0, 0.0)
            crossing = None
            if self.flat_overnight:  # the next period crosses into a new session: go to cash
                session = panel.exogenous["session"]
                crossing = session[self._paths, nxt] != session[self._paths, r]
                held = B.where(crossing[:, None], 0.0, held)
            if self.initial_margin is not None:
                gross = B.sum(B.abs(held), axis=-1, keepdims=True)
                most = 1.0 / self.initial_margin
                held = held * B.where(gross > most, most / B.where(gross > most, gross, 1.0), 1.0)
            trades = held - start
            turnover = B.sum(B.abs(trades), axis=-1)
            cost_ctx = self._context(exec_rows, r, self.cost_model.requires)
            if not self._cost_takes_assets:
                cost_ctx.pop("assets")
            cost = B.asfloat(self.cost_model(trades, **cost_ctx), like)
            holding = None
            if self.holding_cost is not None:
                hctx = self._context(exec_rows, r, self.holding_cost.requires)
                holding = self.holding_cost(
                    held, dt=self.dt, borrow_rate=hctx.get("borrow_rate"), assets=hctx["assets"]
                )
            s = settle(held, ret, cost=cost, dt=self.dt, config=self.config, holding=holding)
            growth = s.growth
            drifted = s.drifted
            if self.execution in ("close", "next_close"):
                drifted = B.where(delisting, 0.0, drifted)
            if overnight is not None:
                growth = B.where(overnight > 0, overnight * growth, 0.0)
            margin_call = None
            if self.maintenance_margin is not None:
                assert self.initial_margin is not None
                gross = B.sum(B.abs(drifted), axis=-1)
                margin_call = (gross * self.maintenance_margin > 1.0) & (growth > 0.0)
                scale = B.where(
                    margin_call, (1.0 / self.initial_margin) / B.where(margin_call, gross, 1.0), 1.0
                )
                forced = drifted * scale[:, None]
                forced_ctx = self._context(nxt, r, self.cost_model.requires)
                if not self._cost_takes_assets:
                    forced_ctx.pop("assets")
                forced_cost = B.asfloat(self.cost_model(forced - drifted, **forced_ctx), like)
                forced_cost = B.where(margin_call, forced_cost, 0.0)
                growth = B.where(forced_cost >= 1.0, 0.0, growth * (1.0 - forced_cost))
                cost = cost + forced_cost
                turnover = turnover + B.where(
                    margin_call, B.sum(B.abs(forced - drifted), axis=-1), 0.0
                )
                drifted = B.where(margin_call[:, None], forced, drifted)
        wiped = growth <= 0.0
        previous = self._equity
        equity = B.where(frozen, previous, B.maximum(previous * growth, 0.0))
        peak = B.maximum(self._peak, equity)
        drawdown = B.where(peak > 0, 1.0 - equity / B.where(peak > 0, peak, 1.0), 1.0)
        variance = exposures = None
        if self.risk is not None:
            variance, exposures = self._risk(held, r)
        zero = B.zeros((n,), like=like)
        outcome = StepOutcome.from_growth(
            B.where(frozen, 1.0, growth),
            equity=equity,
            turnover=B.where(frozen, 0.0, turnover),
            cost=B.where(frozen, 0.0, cost),
            previous_equity=previous,
            weights=held,
            variance=variance,
        )
        t = self._t + (~self._frozen_host).astype(np.int64)
        t_dev = self._t_dev + B.asint(~frozen, like)
        stopped = wiped
        if self.max_drawdown is not None:
            stopped = stopped | (drawdown >= self.max_drawdown)
        terminated = ~frozen & stopped
        truncated = ~frozen & ~terminated & (t_dev >= self._episode_length)
        self._checkpoint(actions, finite, checks, terminated | truncated, deferred)
        reward = self.reward_fn(outcome)
        self._weights = B.where(frozen[:, None], w, drifted)
        self._equity = equity
        self._peak = B.where(frozen, self._peak, peak)
        if self._pending is not None:
            self._pending = B.where(frozen[:, None], self._pending, target)
        self._t, self._t_dev = t, t_dev
        live = ~frozen & ~(terminated | truncated)
        # Reassigning slots is a no-op for lanes that are not live, so on a device it runs
        # without first reading whether any is.
        if self._dynamic and (deferred or B.item(B.any(live))):
            self._assign_slots(live, self._rows())
        info = self._info(
            turnover=B.where(frozen, 0.0, turnover),
            cost=B.where(frozen, 0.0, cost),
            holding=zero if holding is None else B.where(frozen, 0.0, holding),
            financing=B.where(frozen, 0.0, s.financing),
            exec_weights=held,
        )
        info["return"] = B.where(frozen, 0.0, growth - 1.0)
        if variance is not None:
            info["variance"] = variance
            if exposures is not None:
                info["exposures"] = exposures
        if crossing is not None:
            info["session_end"] = B.where(frozen, False, crossing)
        if margin_call is not None:
            info["margin_call"] = B.where(frozen, False, margin_call)
        return reward, terminated, truncated, info

    def _fill_derived(self, state: Mapping[str, Any]) -> None:
        super()._fill_derived(state)
        if "_t_dev" not in state:
            B, like = self.backend, self._like
            self._t_dev = B.asint(self._t, like)
            self._start_dev = B.asint(self._start, like)
            self._paths_host = np.array(B.to_numpy(self._paths), dtype=np.int64)
        self.__dict__.setdefault("_defer", None)

    def _deferring(self) -> bool:
        """Whether steps read their checks back together: on a device, where each read
        waits for the device to finish its queued work."""
        if self._defer is not None:
            return self._defer
        return not self.backend.on_host(self._like)

    def _checkpoint(
        self,
        actions: Any,
        finite: Array,
        checks: list[_Check],
        finished: Array,
        deferred: bool,
    ) -> None:
        """With deferred checks, read them back in one transfer with `finished` (which the
        base class then uses), and raise the first that failed before the step changes any
        state: `InvalidAction` for a non-finite action, else the error the check would have
        raised (a cost model's `ValueError`, the `LinAlgError` of a singular projection)."""
        if not deferred:
            return
        B = self.backend
        failed = B.stack([~finite] + [flag for flag, _ in checks])
        host = B.to_numpy(B.concat([failed, finished]))
        if host[0]:
            raise InvalidAction(actions, self.action_space)
        for (_, error), flag in zip(checks, host[1 : len(checks) + 1]):
            if flag:
                raise error()
        self._finished_host = host[len(checks) + 1 :]

    def _risk(self, weights: Array, rows: Array) -> tuple[Array, Array | None]:
        risk = self.risk
        assert risk is not None
        B = self.backend
        fields = self._exposures(risk.asset_fields, rows)
        loadings = None if risk.loadings is None else fields[risk.loadings]
        idio = None if risk.idio_var is None else fields[risk.idio_var]
        cov = None
        if risk.factor_cov is not None:
            series = self._panel.exogenous[risk.factor_cov]
            cov = B.nan_to_num(series[self._paths, rows])
        variance = portfolio_variance(weights, loadings=loadings, factor_cov=cov, idio_var=idio)
        exposures = None if loadings is None else factor_exposures(weights, loadings)
        return variance, exposures

    def _info(
        self,
        *,
        turnover: Array,
        cost: Array,
        holding: Array,
        financing: Array,
        exec_weights: Array | None,
    ) -> dict[str, Any]:
        B = self.backend
        rows = self._rows()
        w = self._weights
        info: dict[str, Any] = {
            "equity": self._equity,
            "turnover": turnover,
            "cost": cost,
            "holding_cost": holding,
            "financing": financing,
            "gross_exposure": B.sum(B.abs(w), axis=-1),
            "net_exposure": B.sum(w, axis=-1),
            "drawdown": B.where(self._peak > 0, 1.0 - self._equity / self._peak, 1.0),
            "n_assets": B.sum(self._tradable(rows), axis=-1),
            "weights": w,
            "timestamp": self.timestamp,
            "dt": np.full(self.num_envs, self.dt),
        }
        if self.privileged_state:
            info["state"] = self.latent_state
        return info

    def _detach(self) -> None:
        B = self.backend
        self._weights = B.detach(self._weights)
        self._equity = B.detach(self._equity)
        self._peak = B.detach(self._peak)
        if self._pending is not None:
            self._pending = B.detach(self._pending)
        self.reward_fn.detach_state()

    def __repr__(self) -> str:
        source = "data" if self.data is not None else repr(self.scenarios)
        return (
            f"<BatchMarketEnv(num_envs={self.num_envs}, {source}, slots={self.max_assets}, "
            f"execution={self.execution!r}, backend={self.backend.name!r})>"
        )


def cost_context(
    *,
    backend: Backend,
    panel: MarketPanel,
    paths: Array,
    slots: Array,
    rows: Array,
    decision: Array,
    lookback: int,
    needs: frozenset[str],
    equity: Array,
    close_field: str = "close",
) -> dict[str, Any]:
    """The inputs of cost and holding models, per slot, that `needs` names.

    Orders fill at `rows` (whose volume, spread and borrow fees apply); the volatility is
    the data's `volatility` field at those rows if there is one (a per-step forecast, such
    as a risk model's), else it is estimated over the `lookback` returns observed at the
    `decision` rows (the standard deviation of log returns, NaN with fewer than two).
    `volume` is the traded value: the `dollar_volume` field, else `volume` shares times the
    close. Shared by `BatchMarketEnv` and `LiveMarketEnv`, so live and simulated trading
    price costs the same way.
    """
    B = backend
    occupied = slots >= 0
    cols = B.where(occupied, slots, 0)

    def at(arr: Array) -> Array:
        return arr[paths[:, None], rows[:, None], cols]

    ctx: dict[str, Any] = {"assets": cols}
    if "equity" in needs:
        ctx["equity"] = equity
    if "volatility" in needs and "volatility" in panel.fields:  # e.g. a risk model's forecast
        ctx["volatility"] = at(panel.fields["volatility"])
    elif "volatility" in needs:
        view = MarketView(
            backend=B,
            panel=panel,
            paths=paths,
            rows=decision,
            slots=slots,
            lookback=lookback,
            episode_length=0,
            elapsed=None,
            weights=B.zeros(tuple(slots.shape), like=panel.returns),
            equity=equity,
            peak=equity,
            pending=None,
        )
        avail = view.available(lookback)
        like = panel.returns
        window = B.where(avail, B.log1p(view.returns(lookback, fill=0.0)), 0.0)
        count = B.sum(B.asfloat(avail, like), axis=-1)
        mean = B.sum(window, axis=-1) / B.maximum(count, 1.0)
        var = B.sum(B.where(avail, window - mean[..., None], 0.0) ** 2, axis=-1) / B.maximum(
            count, 1.0
        )
        ctx["volatility"] = B.where(count >= 2, B.sqrt(B.maximum(var, 0.0)), float("nan"))
    if "volume" in needs:
        if "dollar_volume" in panel.fields:
            ctx["volume"] = at(panel.fields["dollar_volume"])
        else:
            ctx["volume"] = at(panel.fields["volume"]) * at(panel.fields[close_field])
    if "spread" in needs:
        ctx["spread"] = at(panel.fields["spread"])
    if "borrow_rate" in needs:
        ctx["borrow_rate"] = at(panel.fields["borrow_rate"])
    return ctx


def _accepts(fn: Any, name: str) -> bool:
    """Whether calling `fn` with the keyword argument `name` is allowed."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def _take(value: Any, idx: NDArray[np.int64], indices: Array, B: Backend) -> Any:
    """The rows `idx` of an info value (`indices`, the same lanes as a backend array,
    for backend arrays)."""
    if isinstance(value, dict):
        return {k: _take(v, idx, indices, B) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value[idx]
    if B.is_array(value):
        return value[indices]
    return value


class MarketVecEnv(BatchVecEnv):
    """`Market-v1` as a native vector env: a NumPy `BatchMarketEnv` of `num_envs` lanes.

    What `sekai.make_vec("Market-v1", num_envs=...)` returns. Takes the arguments of
    `BatchMarketEnv` (`backend="numpy"`), plus `autoreset` and `max_episode_steps`.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        autoreset: bool = True,
        max_episode_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            BatchMarketEnv(num_envs, **kwargs),
            autoreset=autoreset,
            max_episode_steps=max_episode_steps,
        )


def make_market(**kwargs: Any) -> UnbatchedEnv:
    """`Market-v1` as a single env: a one-lane `BatchMarketEnv` (`sekai.make` uses it)."""
    if "num_envs" in kwargs:
        raise TypeError("make_market() builds one env; use sekai.make_vec() for several.")
    return UnbatchedEnv(BatchMarketEnv(1, **kwargs))


__all__ = ["BatchMarketEnv", "MarketVecEnv", "make_market", "cost_context", "EXECUTIONS"]
