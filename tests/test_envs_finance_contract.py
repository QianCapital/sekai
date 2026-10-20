"""The finance envs' shared contract: price sources, pandas dates, the end of an episode,
parameter checks and per-env reward state."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import (
    LimitOrderBookEnv,
    LimitOrderBookMarket,
    MarketEnv,
    MarketMakingEnv,
    OptimalExecutionEnv,
    PortfolioEnv,
    PortfolioVecEnv,
    TradingEnv,
    TradingVecEnv,
    simulate_gbm,
)
from sekai.finance import GBM, DifferentialSharpe, MarketData


def closes(n_steps: int = 300, n_assets: int = 3, seed: int = 0) -> np.ndarray:
    s0 = np.linspace(50.0, 150.0, n_assets)
    return simulate_gbm(np.random.default_rng(seed), n_steps - 1, s0=s0, mu=0.05, sigma=0.2)


def dated(n_steps: int = 300, n_assets: int = 3) -> MarketData:
    dates = np.busday_offset("2020-01-01", np.arange(n_steps), roll="forward")
    return MarketData(closes(n_steps, n_assets), timestamps=dates)


# ---------------------------------------------------------------------------
# Price sources: prices= (raw arrays), data= (MarketData), model= (simulation)
# ---------------------------------------------------------------------------


class TestSources:
    def test_data_is_the_keyword_for_market_data_in_every_env(self) -> None:
        data = dated()
        options = {"episode_length": 20, "lookback": 5, "sampler": 7}
        envs: list[Any] = [
            sekai.make("Trading-v0", data=data.select(["asset_0"]), **options),
            sekai.make("Portfolio-v0", data=data, **options),
            sekai.make("Market-v0", data=data, **options),
            TradingVecEnv(2, data=data.select(["asset_0"]), **options),
            PortfolioVecEnv(2, data=data, **options),
            sekai.make_vec("Portfolio-v0", 2, vectorization_mode="sync", data=data, **options),
        ]
        expected = data.timestamps[7 + 5]
        for env in envs:
            r = env.reset(seed=0)
            infos = r.infos if hasattr(r, "infos") else [r.info]
            assert all(info["timestamp"] == expected for info in infos), env

    def test_prices_and_data_give_the_same_episodes(self) -> None:
        prices = closes()
        options = {"episode_length": 20, "lookback": 5, "sampler": 3}
        a = PortfolioEnv(prices=prices, **options)
        b = PortfolioEnv(data=MarketData(prices), **options)
        a.reset(seed=0)
        b.reset(seed=0)
        for _ in range(20):
            action = np.array([1, 2, 0, 1], np.float32)
            assert a.step(action).reward == b.step(action).reward

    @pytest.mark.parametrize("make", [TradingEnv, PortfolioEnv])
    def test_market_data_as_prices_is_refused(self, make: Callable[..., Any]) -> None:
        data = dated(n_assets=1)
        with pytest.raises(TypeError, match="data="):
            make(prices=data, episode_length=20, lookback=5)
        with pytest.raises(TypeError, match="data="):
            make(data, episode_length=20, lookback=5)  # positionally it is prices= too
        env_id = "Trading-v0" if make is TradingEnv else "Portfolio-v0"
        with pytest.raises(TypeError, match="data="):
            sekai.make_vec(env_id, 2, prices=data, episode_length=20, lookback=5)

    @pytest.mark.parametrize("env_id", ["Trading-v0", "Portfolio-v0", "Market-v0"])
    def test_data_must_be_market_data(self, env_id: str) -> None:
        with pytest.raises(TypeError, match="data= takes a MarketData"):
            sekai.make(env_id, data=closes(n_assets=1)[:, None])

    @pytest.mark.parametrize("env_id", ["Trading-v0", "Portfolio-v0", "Market-v0"])
    def test_one_source_at_a_time(self, env_id: str) -> None:
        n = 1 if env_id == "Trading-v0" else 2
        data, model = dated(n_assets=n), GBM(n_assets=n)
        combos: list[dict[str, Any]] = [{"data": data, "model": model}]
        if env_id != "Market-v0":
            prices = data.prices[:, 0] if n == 1 else data.prices
            combos += [{"prices": prices, "data": data}, {"prices": prices, "model": model}]
        for sources in combos:
            with pytest.raises(ValueError, match="pass one source"):
                sekai.make(env_id, episode_length=20, lookback=5, **sources)

    def test_a_model_is_never_silently_ignored(self) -> None:
        with pytest.raises(ValueError, match="pass one source"):  # used to build a 2-asset env
            sekai.make("Portfolio-v0", prices=closes(n_assets=2), model=GBM(n_assets=3))
        with pytest.raises(ValueError, match="pass one source"):
            sekai.make_vec("Trading-v0", 2, prices=closes(n_assets=1)[:, 0], model=GBM())

    def test_n_assets_must_agree_with_the_history_or_model(self) -> None:
        assert PortfolioEnv().n_assets == 5
        assert PortfolioEnv(prices=closes(n_assets=2), n_assets=2).n_assets == 2
        assert PortfolioEnv(model=GBM(n_assets=4)).n_assets == 4
        with pytest.raises(ValueError, match="n_assets=3 but the model has 2"):
            PortfolioEnv(model=GBM(n_assets=2), n_assets=3)
        with pytest.raises(ValueError, match="n_assets=4 but the price history has 2"):
            PortfolioEnv(prices=closes(n_assets=2), n_assets=4)


# ---------------------------------------------------------------------------
# Pandas input keeps its dates
# ---------------------------------------------------------------------------


class TestPandasPrices:
    @pytest.fixture
    def frame(self) -> Any:
        pd = pytest.importorskip("pandas")
        index = pd.bdate_range("2014-01-01", periods=2520)  # ten years of business days
        return pd.DataFrame(closes(2520), index=index, columns=["AAA", "BBB", "CCC"])

    def test_dated_frame_behaves_like_market_data(self, frame: Any) -> None:
        options = {"episode_length": 63, "lookback": 20}
        env = PortfolioEnv(prices=frame, **options)
        reference = PortfolioEnv(data=MarketData.from_pandas(frame), **options)
        assert env.data is not None and env.data.assets == ("AAA", "BBB", "CCC")
        assert env.dt == reference.dt == pytest.approx(1 / 260.9, rel=1e-3)  # not 1/252
        r = env.reset(seed=0, options={"start": "2020-03-02"})
        assert r.info["timestamp"] == np.datetime64("2020-03-02")  # used to be 2019-10-01
        assert r.info["dt"] == env.dt
        np.testing.assert_array_equal(env.prices, frame.to_numpy())
        step = env.step(np.ones(4, np.float32))
        assert step.info["timestamp"] == np.datetime64("2020-03-03")
        rows = frame.index.get_loc("2020-03-02")
        returns = frame.iloc[rows + 1].to_numpy() / frame.iloc[rows].to_numpy() - 1
        np.testing.assert_allclose(step.info["asset_returns"], returns)
        replay = reference.reset(seed=0, options={"start": "2020-03-02"})
        assert replay.info["timestamp"] == r.info["timestamp"]

    def test_trading_takes_a_dated_series_or_column(self, frame: Any) -> None:
        for prices in (frame["BBB"], frame[["BBB"]]):
            env = TradingEnv(prices=prices, episode_length=63, lookback=20)
            r = env.reset(seed=0, options={"start": "2020-03-02"})
            assert r.info["timestamp"] == np.datetime64("2020-03-02")
            assert r.info["price"] == frame.loc["2020-03-02", "BBB"]
        with pytest.raises(ValueError, match="one asset"):
            TradingEnv(prices=frame, episode_length=63, lookback=20)

    def test_native_vector_env_keeps_the_dates(self, frame: Any) -> None:
        vec = sekai.make_vec("Portfolio-v0", 3, prices=frame, episode_length=63, lookback=20)
        r = vec.reset(seed=0, options={"start": "2020-03-02"})
        # Compared elementwise: before numpy 2.2, equal datetime64s of different units
        # (the index's ns, the literal's days) hash differently, so a set comparison fails.
        assert all(info["timestamp"] == np.datetime64("2020-03-02") for info in r.infos)
        assert vec.get_attr("dt")[0] == pytest.approx(1 / 260.9, rel=1e-3)

    def test_other_indexes_number_the_rows(self, frame: Any) -> None:
        plain = frame.reset_index(drop=True)
        env = PortfolioEnv(prices=plain, episode_length=63, lookback=20)
        assert env.dt == 1 / 252 and env.data is not None and not env.data.is_datetime
        assert env.reset(seed=0, options={"start": 100}).info["timestamp"] == 100
        one = TradingEnv(prices=plain[["AAA"]], episode_length=63, lookback=20)  # (T, 1)
        np.testing.assert_array_equal(one.prices, plain["AAA"].to_numpy())

    def test_timezone_aware_dates_are_refused(self, frame: Any) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            PortfolioEnv(prices=frame.tz_localize("UTC"))


# ---------------------------------------------------------------------------
# step() after the episode has ended
# ---------------------------------------------------------------------------

SHORT_EPISODES: dict[str, tuple[Callable[[], Any], Any]] = {
    "Trading-v0": (lambda: TradingEnv(episode_length=3), 2),
    "Portfolio-v0": (lambda: PortfolioEnv(n_assets=2, episode_length=3), np.ones(3, np.float32)),
    "Market-v0": (lambda: MarketEnv(episode_length=3), np.ones(11, np.float32)),
    "MarketMaking-v0": (lambda: MarketMakingEnv(n_steps=3), np.ones(2, np.float32)),
    "OptimalExecution-v0": (lambda: OptimalExecutionEnv(n_steps=3), np.zeros(1, np.float32)),
    "LimitOrderBook-v0": (lambda: LimitOrderBookEnv(session_length=3.0), np.array([1, 1, 1])),
}


class TestEndOfEpisode:
    @pytest.mark.parametrize("env_id", sorted(SHORT_EPISODES))
    def test_step_after_the_end_requires_a_reset(self, env_id: str) -> None:
        build, action = SHORT_EPISODES[env_id]
        env = build()
        with pytest.raises(sekai.ResetRequired, match="reset"):  # before the first reset
            env.step(action)
        for _ in range(2):  # a reset starts a new episode each time
            r = env.reset(seed=0)
            steps = 0
            while not r.done:
                r = env.step(action)
                steps += 1
            assert steps == 3
            for _ in range(2):  # no IndexError, no observation outside the space, no reward
                with pytest.raises(sekai.ResetRequired, match="episode is over"):
                    env.step(action)

    def test_early_termination_counts_too(self) -> None:
        env = OptimalExecutionEnv()
        env.reset(seed=0)
        assert env.step(np.ones(1, np.float32)).terminated  # sold everything at once
        with pytest.raises(sekai.ResetRequired, match="episode is over"):
            env.step(np.ones(1, np.float32))
        wiped = TradingEnv(positions=(0.0, 1.0), transaction_cost=1.0, episode_length=5)
        wiped.reset(seed=0)
        assert wiped.step(1).terminated  # the trade cost the whole account
        with pytest.raises(sekai.ResetRequired, match="episode is over"):
            wiped.step(1)

    def test_multi_agent_market(self) -> None:
        market = LimitOrderBookMarket(2, session_length=3.0)
        with pytest.raises(sekai.ResetRequired):
            market.step({})
        r = market.reset(seed=0)
        while not r.all_done:
            r = market.step({a: np.array([1, 1, 1]) for a in r.observations})
        assert market.agents == []
        with pytest.raises(sekai.ResetRequired, match="episode is over"):
            market.step({a: np.array([1, 1, 1]) for a in market.possible_agents})
        assert sorted(market.reset(seed=0).observations) == ["trader_0", "trader_1"]

    @pytest.mark.parametrize(
        "env_id", ["Trading-v0", "Portfolio-v0", "MarketMaking-v0", "OptimalExecution-v0"]
    )
    def test_native_vector_envs_reset_finished_sub_envs(self, env_id: str) -> None:
        horizon = {"episode_length": 3} if env_id in ("Trading-v0", "Portfolio-v0") else {}
        if env_id in ("MarketMaking-v0", "OptimalExecution-v0"):
            horizon = {"n_steps": 3}
        vec = sekai.make_vec(env_id, 2, **horizon)
        vec.reset(seed=0)
        dones = 0
        for _ in range(10):
            dones += int(vec.step(vec.batch_action_space.sample()).dones.sum())
        assert dones >= 6


# ---------------------------------------------------------------------------
# Parameters are checked at construction
# ---------------------------------------------------------------------------

BAD_PARAMETERS: list[tuple[Callable[..., Any], dict[str, Any], str]] = [
    (MarketMakingEnv, {"total_time": 0.0}, "total_time"),  # used to be silently inert
    (MarketMakingEnv, {"total_time": -1.0}, "total_time"),  # used to fail on the first step
    (MarketMakingEnv, {"total_time": float("nan")}, "total_time"),
    (MarketMakingEnv, {"sigma": -1.0}, "sigma"),
    (MarketMakingEnv, {"sigma": float("nan")}, "sigma"),
    (MarketMakingEnv, {"fill_intensity": 0.0}, "fill_intensity"),
    (MarketMakingEnv, {"fill_decay": -1.0}, "fill_decay"),
    (MarketMakingEnv, {"risk_aversion": -0.1}, "risk_aversion"),
    (MarketMakingEnv, {"max_offset": -1.0}, "max_offset"),
    (MarketMakingEnv, {"s0": float("inf")}, "s0"),
    (OptimalExecutionEnv, {"sigma": -1.0}, "sigma"),
    (OptimalExecutionEnv, {"initial_inventory": float("nan")}, "initial_inventory"),
    (OptimalExecutionEnv, {"s0": 0.0}, "s0"),
    (OptimalExecutionEnv, {"total_time": 0.0}, "total_time"),
    (OptimalExecutionEnv, {"permanent_impact": -1e-7}, "permanent_impact"),
    (OptimalExecutionEnv, {"temporary_impact": float("nan")}, "temporary_impact"),
    (OptimalExecutionEnv, {"fixed_cost": -0.01}, "fixed_cost"),
    (OptimalExecutionEnv, {"risk_aversion": -1e-6}, "risk_aversion"),
    (TradingEnv, {"positions": (0.0, float("nan"))}, "positions"),
    (TradingEnv, {"capital": 0.0}, "capital"),
    (TradingEnv, {"capital": -5.0}, "capital"),
    (TradingEnv, {"sigma": -0.2}, "sigma"),
    (TradingEnv, {"mu": float("nan")}, "mu"),
    (TradingEnv, {"s0": 0.0}, "s0"),
    (TradingEnv, {"dt": 0.0}, "dt"),
    (PortfolioEnv, {"capital": float("nan")}, "capital"),
    (PortfolioEnv, {"n_assets": 2, "sigma": [0.2, -0.1]}, "sigma"),
    (PortfolioEnv, {"n_assets": 2, "mu": [0.1, float("inf")]}, "mu"),
    (PortfolioEnv, {"n_assets": 3, "mu": [0.1, 0.2]}, "mu"),
    (PortfolioEnv, {"n_assets": 2, "corr": [[1.0, 2.0], [2.0, 1.0]]}, "corr"),
    (PortfolioEnv, {"dt": -1.0}, "dt"),
    (MarketEnv, {"capital": 0.0}, "capital"),
    (MarketEnv, {"max_leverage": float("nan")}, "max_leverage"),
    (MarketEnv, {"borrow_rate": -0.01}, "borrow_rate"),
    (MarketEnv, {"cash_rate": float("inf")}, "cash_rate"),
    (MarketEnv, {"margin_rate": float("nan")}, "margin_rate"),
    (MarketEnv, {"delisting_return": -2.0}, "delisting_return"),
    (MarketEnv, {"dt": 0.0}, "dt"),
    (LimitOrderBookEnv, {"session_length": float("nan")}, "session_length"),
    (LimitOrderBookEnv, {"session_length": 0.0}, "session_length"),
    (LimitOrderBookEnv, {"decision_interval": float("nan")}, "decision_interval"),
    (LimitOrderBookEnv, {"warmup": -1.0}, "warmup"),
    (LimitOrderBookEnv, {"inventory_penalty": float("nan")}, "inventory_penalty"),
    (LimitOrderBookMarket, {"decision_intervals": [1.0, float("nan")]}, "decision_intervals"),
    (LimitOrderBookMarket, {"decision_intervals": [1.0, 0.0]}, "decision_intervals"),
    (LimitOrderBookMarket, {"jitter": float("nan")}, "jitter"),
]


class TestParameterChecks:
    @pytest.mark.parametrize(
        "make, kwargs, name",
        BAD_PARAMETERS,
        ids=[f"{m.__name__}-{next(iter(k))}" for m, k, _ in BAD_PARAMETERS],
    )
    def test_out_of_range_parameters_raise(
        self, make: Callable[..., Any], kwargs: dict[str, Any], name: str
    ) -> None:
        with pytest.raises(ValueError, match=name):
            make(**kwargs)

    @pytest.mark.parametrize(
        "env_id, kwargs",
        [
            ("Trading-v0", {"capital": 0.0}),
            ("Portfolio-v0", {"sigma": -0.1}),
            ("MarketMaking-v0", {"total_time": 0.0}),
            ("OptimalExecution-v0", {"sigma": -1.0}),
        ],
    )
    def test_native_vector_envs_check_them_too(self, env_id: str, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            sekai.make_vec(env_id, 2, **kwargs)

    def test_boundary_values_are_valid(self) -> None:
        MarketMakingEnv(sigma=0.0, risk_aversion=0.0, max_offset=0.0).reset(seed=0)
        OptimalExecutionEnv(sigma=0.0, risk_aversion=0.0, fixed_cost=0.0, permanent_impact=0.0)
        TradingEnv(sigma=0.0, mu=-0.5, positions=(-2, 0, 2)).reset(seed=0)
        MarketEnv(delisting_return=-1.0, cash_rate=-0.01, borrow_rate=0.0).reset(seed=0)
        LimitOrderBookEnv(warmup=0.0, inventory_penalty=0.0).reset(seed=0)
        LimitOrderBookMarket(2, jitter=0.0, decision_intervals=[0.5, 2]).reset(seed=0)


# ---------------------------------------------------------------------------
# A reward set later is copied, as reward= is
# ---------------------------------------------------------------------------


class TestRewardCopies:
    @pytest.mark.parametrize("make", [TradingEnv, PortfolioEnv, MarketEnv])
    def test_setting_reward_fn_stores_a_copy(self, make: Callable[..., Any]) -> None:
        reward = DifferentialSharpe(0.1)
        a, b = make(), make()
        a.reward_fn = reward
        b.reward_fn = reward
        assert isinstance(a.reward_fn, DifferentialSharpe)
        assert a.reward_fn is not reward and a.reward_fn is not b.reward_fn
        assert make(reward=reward).reward_fn is not reward

    @pytest.mark.parametrize(
        "mode, vector_kwargs",
        [("sync", {}), ("native", {}), ("proc", {"num_workers": 1})],
    )
    def test_set_attr_gives_each_sub_env_its_own_state(
        self, mode: str, vector_kwargs: dict[str, Any]
    ) -> None:
        options = {"prices": closes(n_assets=1)[:, 0], "episode_length": 30, "lookback": 5}
        options["sampler"] = 7
        actions = np.random.default_rng(0).integers(0, 3, (20, 3))
        reference = sekai.make_vec(
            "Trading-v0", 3, vectorization_mode="sync", reward=DifferentialSharpe(0.1), **options
        )
        reference.reset(seed=0)
        expected = [reference.step(a).rewards for a in actions]
        vec = sekai.make_vec(
            "Trading-v0", 3, vectorization_mode=mode, vector_kwargs=vector_kwargs, **options
        )
        try:
            vec.set_attr("reward_fn", DifferentialSharpe(0.1))  # one object for every sub-env
            vec.reset(seed=0)
            got = [vec.step(a).rewards for a in actions]
        finally:
            vec.close()
        np.testing.assert_allclose(got, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Read-only state accessors (for renderers and loggers that can't see info)
# ---------------------------------------------------------------------------


class TestStateAccessors:
    @pytest.mark.parametrize("make", [TradingEnv, PortfolioEnv, MarketEnv])
    def test_accessors_need_a_reset(self, make: Callable[..., Any]) -> None:
        env = make()
        with pytest.raises(sekai.ResetRequired):
            env.timestamp
        if hasattr(make, "price_history"):
            with pytest.raises(sekai.ResetRequired):
                env.price_history

    @pytest.mark.parametrize(
        ("make", "data"),
        [
            (lambda d: TradingEnv(data=d, lookback=5, episode_length=20), dated(n_assets=1)),
            (lambda d: PortfolioEnv(data=d, lookback=5, episode_length=20), dated()),
            (lambda d: MarketEnv(data=d, lookback=5, episode_length=20), dated()),
            (lambda d: TradingEnv(lookback=5, episode_length=20), None),
            (lambda d: PortfolioEnv(lookback=5, episode_length=20), None),
            (lambda d: MarketEnv(lookback=5, episode_length=20), None),
        ],
    )
    def test_timestamp_is_the_info_timestamp(
        self, make: Callable[[Any], Any], data: MarketData | None
    ) -> None:
        env = make(data)
        result = env.reset(seed=0)
        for _ in range(5):
            assert env.timestamp == result.info["timestamp"]
            result = env.step(env.action_space.sample())
        assert env.timestamp == result.info["timestamp"]

    @pytest.mark.parametrize("make", [TradingEnv, PortfolioEnv])
    def test_price_history_ends_at_the_current_price(self, make: Callable[..., Any]) -> None:
        env = make(data=dated(n_assets=1) if make is TradingEnv else dated(), lookback=5)
        env.reset(seed=0)
        for t in range(4):
            history = env.price_history
            assert len(history) == 5 + t + 1
            current = env.price if make is TradingEnv else env._path[5 + t]
            np.testing.assert_array_equal(history[-1], current)
            history[-1] = -1.0  # a copy: the env is unaffected
            assert np.all(env.price_history[-1] != -1.0)
            env.step(env.action_space.sample())
        assert env.price_history.ndim == (1 if make is TradingEnv else 2)

    def test_latent_state_matches_privileged_info(self) -> None:
        env = MarketEnv(privileged_state=True, lookback=5, episode_length=10)
        result = env.reset(seed=0)
        for _ in range(4):
            state = env.latent_state
            assert state.keys() == result.info["state"].keys() and "regime" in state
            for key in state:
                np.testing.assert_array_equal(state[key], result.info["state"][key])
            result = env.step(env.action_space.sample())
        historical = MarketEnv(data=dated(), lookback=5)
        historical.reset(seed=0)
        assert historical.latent_state == {}

    def test_market_making_exposes_the_last_quotes_and_fills(self) -> None:
        env = MarketMakingEnv()
        env.reset(seed=0)
        assert np.isnan(env.bid) and np.isnan(env.ask)
        assert env.bid_filled is False and env.ask_filled is False
        for _ in range(20):
            info = env.step(env.action_space.sample()).info
            assert (env.bid, env.ask) == (info["bid"], info["ask"])
            assert (env.bid_filled, env.ask_filled) == (info["bid_filled"], info["ask_filled"])
        env.reset(seed=1)
        assert np.isnan(env.bid) and env.bid_filled is False
