"""The composable finance envs: Trading/Portfolio options, Market-v0, native vector envs, LOB."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import (
    LimitOrderBookEnv,
    LimitOrderBookMarket,
    MarketEnv,
    PortfolioEnv,
    PortfolioVecEnv,
    TradingEnv,
)
from sekai.finance import (
    GBM,
    DifferentialSharpe,
    FixedCost,
    Heston,
    ListingProcess,
    MarketData,
    MeanReverting,
    ProportionalCost,
    SequentialStart,
    SpreadCost,
    SquareRootImpact,
)
from sekai.finance.lob import BACKGROUND, Side, ZeroIntelligenceFlow
from sekai.finance.metrics import RecordPerformance

# Background flow so slow that the book is effectively static during a test.
STATIC_FLOW = ZeroIntelligenceFlow(limit_rate=1e-9, market_rate=1e-9, cancel_rate=0.0)


def history(n_steps: int = 400, n_assets: int = 3, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (n_steps, n_assets)), axis=0))
    return MarketData(
        prices,
        timestamps=np.datetime64("2020-01-01") + np.arange(n_steps),
        assets=[f"S{i}" for i in range(n_assets)],
        fields={
            "volume": np.full((n_steps, n_assets), 1e6),
            "spread": np.full((n_steps, n_assets), 4e-4),
        },
        exogenous={"rate": np.linspace(0.0, 0.05, n_steps)},
    )


# ---------------------------------------------------------------------------
# Trading-v0 / Portfolio-v0 options
# ---------------------------------------------------------------------------


class TestTradingAndPortfolioOptions:
    def test_market_data_start_and_timestamps(self) -> None:
        data = history()
        env = TradingEnv(data=data.select(["S0"]), episode_length=20, lookback=5)
        r = env.reset(seed=0, options={"start": "2020-03-01"})
        assert r.info["timestamp"] == np.datetime64("2020-03-01") and r.info["dt"] == env.dt
        assert env.price == pytest.approx(data.prices[data.index_of("2020-03-01"), 0])
        env.reset(options={"start": 10})
        assert env.price == pytest.approx(data.prices[10, 0])
        with pytest.raises(ValueError, match="valid range"):
            env.reset(options={"start": 2})  # needs `lookback` rows of history first

    def test_sequential_sampler_tiles_the_history(self) -> None:
        env = PortfolioEnv(data=history(), episode_length=50, lookback=10, sampler="sequential")
        firsts = [env.reset(seed=0).info["timestamp"] for _ in range(3)]
        assert firsts[1] - firsts[0] == np.timedelta64(50, "D") == firsts[2] - firsts[1]

    def test_cost_models_and_rewards(self) -> None:
        env = PortfolioEnv(
            data=history(), episode_length=20, lookback=5, transaction_cost=SpreadCost()
        )
        env.reset(seed=0)
        r = env.step(np.array([1, 0, 0, 0], np.float32))  # all into S0 from cash
        assert (
            r.info["cost"] == pytest.approx(0.5 * 4e-4)
            and r.info["equity"] == r.info["portfolio_value"]
        )
        sharpe = TradingEnv(reward=DifferentialSharpe(0.1))
        sharpe.reset(seed=0)
        rewards = [sharpe.step(2).reward for _ in range(30)]
        assert rewards[0] == 0.0 and any(r != 0.0 for r in rewards)

    def test_market_models(self) -> None:
        env = PortfolioEnv(model=Heston(n_assets=4), episode_length=30)
        assert env.n_assets == 4 and env.reset(seed=1).observation["returns"].shape == (30, 4)
        assert sekai.check_env(env, n_steps=60) == []
        trading = TradingEnv(model=MeanReverting(theta=20.0, mean=100.0, sigma=0.3))
        assert sekai.check_env(trading, n_steps=60) == []
        with pytest.raises(ValueError, match="one-asset"):
            TradingEnv(model=GBM(n_assets=2))

    @pytest.mark.parametrize("env_id", ["Trading-v0", "Portfolio-v0", "Market-v0"])
    @pytest.mark.parametrize("name", ["episode_length", "lookback"])
    def test_shape_parameters_are_read_only(self, env_id: str, name: str) -> None:
        env = sekai.make(env_id)
        value = getattr(env, name)
        with pytest.raises(AttributeError, match="fixed at construction"):
            setattr(env, name, value + 1)
        assert getattr(env, name) == value
        vec = sekai.make_vec(env_id, 2, vectorization_mode="sync")
        try:
            with pytest.raises(AttributeError, match="fixed at construction"):
                vec.set_attr(name, 5)
            assert vec.get_attr(name) == [value, value]
            vec.reset(seed=0)
            for _ in range(10):  # the arrays still match the parameters
                vec.step(vec.batch_action_space.sample())
        finally:
            vec.close()

    def test_validation(self) -> None:
        gappy = history().prices.copy()
        gappy[:10, 1] = np.nan
        with pytest.raises(ValueError, match="Market-v0"):
            PortfolioEnv(data=MarketData(gappy), episode_length=20, lookback=5)
        with pytest.raises(ValueError, match="select"):
            TradingEnv(data=history(), episode_length=20, lookback=5)
        with pytest.raises(ValueError, match="history"):
            TradingEnv().reset(options={"start": 3})
        listed = ListingProcess(GBM(n_assets=3), initial_listed=0.3, min_listed=1)
        with pytest.raises(ValueError, match="listed"):
            PortfolioEnv(model=listed).reset(seed=0)


# ---------------------------------------------------------------------------
# Market-v0
# ---------------------------------------------------------------------------


def toy_universe() -> MarketData:
    """3 assets: A always listed, B delists after row 30, C lists at row 20."""
    t = 60
    a = 100 * 1.001 ** np.arange(t)
    b = np.where(np.arange(t) <= 30, 50 * 1.002 ** np.arange(t), np.nan)
    c = np.where(np.arange(t) >= 20, 10 * 0.999 ** np.arange(t), np.nan)
    return MarketData(np.stack([a, b, c], axis=1), assets=["A", "B", "C"])


class TestMarketEnv:
    def test_universe_changes_are_reflected_in_mask_and_action_mask(self) -> None:
        env = MarketEnv(toy_universe(), lookback=5, episode_length=50)
        obs = env.reset(options={"start": 5}).observation
        assert obs["assets"]["mask"].tolist() == [1, 1, 0]
        assert env.action_mask().tolist() == [1, 1, 0, 1]  # + cash
        seen = []
        result: Any = None
        for _ in range(49):
            result = env.step(np.array([1, 1, 1, 0], np.float32))
            seen.append(result.observation["assets"]["mask"].tolist())
            assert env.observation_space.contains(result.observation)
        assert [1, 1, 1] in seen and seen[-1] == [1, 0, 1]
        assert result.info["n_assets"] == 2

    def test_delisting_moves_the_position_to_cash(self) -> None:
        env = MarketEnv(
            toy_universe(),
            lookback=5,
            episode_length=10,
            transaction_cost=0.0,
            delisting_return=-0.5,
        )
        env.reset(options={"start": 30})  # first decision at row 30: B's last listed row
        r = env.step(np.array([0, 1, 0, 0], np.float32))  # all in B as it delists
        assert r.info["equity"] == pytest.approx(0.5)
        assert r.observation["portfolio"][1] == 0 and r.observation["portfolio"][
            -1
        ] == pytest.approx(1.0)

    def test_untradable_assets_get_no_weight(self) -> None:
        env = MarketEnv(toy_universe(), lookback=5, episode_length=50)
        env.reset(options={"start": 5})
        w = env.target_weights(np.ones(4, np.float32))
        assert w[2] == 0.0 and w.sum() == pytest.approx(2 / 3)

    def test_financing_and_costs(self) -> None:
        data = history()
        env = MarketEnv(data, lookback=5, episode_length=10, cash_rate=0.05, transaction_cost=0.0)
        env.reset(seed=0)
        cash = np.zeros(4, np.float32)
        cash[-1] = 1
        r = env.step(cash)
        assert r.info["financing"] == pytest.approx(0.05 * env.dt)
        assert r.info["equity"] == pytest.approx(1 + 0.05 * env.dt)
        costly = MarketEnv(
            data, lookback=5, episode_length=10, transaction_cost=ProportionalCost(0.01)
        )
        costly.reset(seed=0)
        assert costly.step(np.array([1, 0, 0, 0], np.float32)).info["cost"] == pytest.approx(0.01)

    def test_account_observes_no_clock(self) -> None:
        # An episode is a window of an ongoing market: it truncates, and time is not observed.
        env = MarketEnv(history(), lookback=5, episode_length=10)
        assert env.observation_space["account"].shape == (2,)
        np.testing.assert_allclose(env.reset(seed=0).observation["account"], [0.0, 0.0])
        r = env.step(np.array([1, 0, 0, 0], np.float32))
        expected = [np.log(r.info["equity"]), r.info["gross_exposure"]]
        np.testing.assert_allclose(r.observation["account"], expected, rtol=1e-6)
        timed = sekai.TimeAwareObservation(env, max_episode_steps=10)
        assert "time" in timed.reset(seed=0).observation

    def test_long_short_leverage(self) -> None:
        env = MarketEnv(
            history(), lookback=5, episode_length=10, allow_short=True, max_leverage=2.0
        )
        env.reset(seed=0)
        r = env.step(np.array([1.0, -1.0, 1.0], np.float32))
        assert r.info["gross_exposure"] == pytest.approx(2.0, rel=0.05)
        assert env.action_space.shape == (3,)

    def test_features_exogenous_and_privileged_state(self) -> None:
        data = history().with_fields(score=np.tile(np.arange(400.0)[:, None], (1, 3)))
        env = MarketEnv(data, features=["score"], exogenous=["rate"], lookback=4, episode_length=10)
        obs = env.reset(options={"start": 100}).observation
        np.testing.assert_allclose(obs["assets"]["values"][0, :, 2], [97, 98, 99, 100])
        np.testing.assert_allclose(obs["market"][:, 0], data.exogenous["rate"][97:101], rtol=1e-6)
        sim = MarketEnv(exogenous=["regime"], privileged_state=True, episode_length=30)
        r = sim.reset(seed=0)
        assert r.observation["market"].shape == (20, 1) and r.info["state"]["regime"] in (0.0, 1.0)
        with pytest.raises(KeyError):
            MarketEnv(data, features=["nope"])
        with pytest.raises(ValueError, match="features"):
            MarketEnv(features=["score"])

    def test_slot_map_keeps_assets_in_their_slots(self) -> None:
        rng = np.random.default_rng(1)
        t, n = 200, 12
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (t, n)), axis=0))
        for j in range(n):  # staggered lifetimes: at most ~5 alive at once
            prices[: 15 * j, j] = np.nan
            prices[15 * j + 60 :, j] = np.nan
        env = MarketEnv(MarketData(prices), max_assets=6, lookback=5, episode_length=150)
        env.reset(options={"start": 5})
        history_of: dict[int, set[Any]] = {}
        for _ in range(150):
            r = env.step(env.action_space.sample(mask=env.action_mask()))
            for slot, asset in enumerate(env.slot_assets):
                if asset is not None:
                    history_of.setdefault(asset, set()).add(slot)
            if r.done:
                break
        # Every asset that lived during the episode (the last lists after it) kept one slot.
        assert len(history_of) == n - 1
        assert all(len(slots) == 1 for slots in history_of.values())

    def test_works_with_wrappers_and_vectors(self) -> None:
        env = sekai.NormalizeObservation(
            sekai.FrameStackObservation(MarketEnv(episode_length=20), 2), stats="ewm", halflife=50
        )
        r = env.reset(seed=0)
        assert r.observation["assets"]["values"].shape == (2, 10, 20, 2)
        perf = RecordPerformance(MarketEnv(episode_length=20))
        perf.reset(seed=0)
        while True:
            r = perf.step(perf.action_space.sample())
            if r.done:
                break
        assert r.info["performance"].n_periods == 20
        vec = sekai.make_vec("Market-v0", 2, vectorization_mode="proc", episode_length=15)
        try:
            assert vec.shared_memory
            assert vec.reset(seed=0).observations["assets"]["values"].shape == (2, 10, 20, 2)
        finally:
            vec.close()

    def test_more_slots_than_assets(self) -> None:
        env = MarketEnv(toy_universe(), max_assets=5, lookback=5, episode_length=10)
        obs = env.reset(options={"start": 5}).observation
        assert obs["assets"]["mask"].tolist() == [1, 1, 0, 0, 0]
        assert env.slot_assets == ("A", "B", "C", None, None)
        assert sekai.check_env(env, n_steps=30) == []

    def test_walk_forward_evaluation(self) -> None:
        data = history(600)
        folds = list(data.walk_forward(300, 100, embargo=5, history=10))
        for _train, test in folds:
            env = MarketEnv(test, lookback=10, episode_length=99, sampler=SequentialStart())
            first = env.reset(seed=0).info["timestamp"]
            assert first == test.timestamps[10]


# ---------------------------------------------------------------------------
# Native vector envs
# ---------------------------------------------------------------------------


class TestNativeFinance:
    @pytest.mark.parametrize(
        "env_id, kwargs",
        [
            ("Trading-v0", {"prices": history().prices[:, 0]}),
            ("Trading-v0", {"prices": history().prices[:, 0], "positions": (-2, -0.5, 0, 1.5)}),
            ("Portfolio-v0", {"data": history()}),
        ],
    )
    def test_matches_sync_on_fixed_history(self, env_id: str, kwargs: dict[str, Any]) -> None:
        common = {**kwargs, "episode_length": 30, "lookback": 8, "sampler": 50}
        native = sekai.make_vec(env_id, 3, **common)
        sync = sekai.make_vec(env_id, 3, vectorization_mode="sync", **common)
        native.reset(seed=0)
        sync.reset(seed=0)
        actions = native.batch_action_space
        actions.seed(0)
        for _ in range(70):  # crosses two autoresets
            a = actions.sample()
            x, y = native.step(a), sync.step(a)
            np.testing.assert_allclose(x.rewards, y.rewards, rtol=1e-12, atol=1e-15)
            assert (x.terminated == y.terminated).all() and (x.truncated == y.truncated).all()
            assert x.infos[0].keys() == y.infos[0].keys()
            assert x.infos[1]["timestamp"] == y.infos[1]["timestamp"]
            if env_id == "Trading-v0":  # drifted positions and the turnover back to the target
                np.testing.assert_allclose(x.observations, y.observations, rtol=1e-6)
                for i in range(3):
                    for key in ("position", "turnover", "cost", "equity"):
                        assert x.infos[i][key] == pytest.approx(
                            y.infos[i][key], rel=1e-12, abs=1e-15
                        )

    @pytest.mark.parametrize("env_id", ["Trading-v0", "Portfolio-v0"])
    def test_data_driven_costs_match_sync(self, env_id: str) -> None:
        data = history()
        prices = data.select(["S0"]) if env_id == "Trading-v0" else data
        costs = SpreadCost() + SquareRootImpact(0.5) + FixedCost(25.0)
        common = {
            "data": prices,
            "episode_length": 20,
            "lookback": 8,
            "sampler": 30,
            "transaction_cost": costs,
            "capital": 5e6,
        }
        native = sekai.make_vec(env_id, 2, **common)
        sync = sekai.make_vec(env_id, 2, vectorization_mode="sync", **common)
        native.reset(seed=0)
        sync.reset(seed=0)
        actions = native.batch_action_space
        actions.seed(1)
        paid = 0.0
        for _ in range(25):
            a = actions.sample()
            x, y = native.step(a), sync.step(a)
            for i in range(2):
                assert x.infos[i]["cost"] == pytest.approx(y.infos[i]["cost"], rel=1e-9, abs=1e-15)
                paid += y.infos[i]["cost"]
        assert paid > 0

    @pytest.mark.parametrize(
        "env_id", ["Trading-v0", "Portfolio-v0", "MarketMaking-v0", "OptimalExecution-v0"]
    )
    def test_native_is_the_default_and_well_formed(self, env_id: str) -> None:
        vec = sekai.make_vec(env_id, 8)
        assert type(vec).__name__.endswith("VecEnv") and not isinstance(vec, sekai.SyncVecEnv)
        r = vec.reset(seed=0)
        single = sekai.make(env_id)
        assert vec.observation_space == single.observation_space
        assert vec.batch_observation_space.contains(r.observations)
        s = vec.step(vec.batch_action_space.sample())
        assert set(s.infos[0]) == set(single.reset(seed=0).info) | set(
            single.step(single.action_space.sample()).info
        )

    def test_simulated_timestamps_match_the_single_env(self) -> None:
        vec = sekai.make_vec("Trading-v0", 2)
        vec.reset(seed=0)
        single = sekai.make("Trading-v0")
        single.reset(seed=0)
        a, b = vec.step(np.array([1, 1])).infos[0], single.step(1).info
        assert a["timestamp"] == b["timestamp"] == 1

    def test_market_making_baseline_vectorised(self) -> None:
        vec = sekai.make_vec("MarketMaking-v0", 64)
        vec.reset(seed=0)
        totals = np.zeros(64)
        for _ in range(200):
            totals += vec.step(vec.avellaneda_stoikov_action()).rewards
        naive = sekai.make_vec("MarketMaking-v0", 64)
        naive.reset(seed=0)
        naive_totals = np.zeros(64)
        for _ in range(200):
            naive_totals += naive.step(np.full((64, 2), 0.2, np.float32)).rewards
        assert totals.mean() > naive_totals.mean() + 10

    def test_execution_schedule_vectorised(self) -> None:
        vec = sekai.make_vec("OptimalExecution-v0", 32)
        vec.reset(seed=0)
        schedule = vec.almgren_chriss_actions()
        for k in range(len(schedule)):
            r = vec.step(np.tile(schedule[k], (32, 1)))
        assert r.terminated.all() and all(i["final_info"]["inventory"] == 0 for i in r.infos)

    @pytest.mark.parametrize("limit", [None, 12])
    def test_frozen_sub_envs_keep_their_state_like_sync(self, limit: int | None) -> None:
        # Levered trading on a volatile history: sub-envs are wiped out at different steps,
        # then stay frozen (autoreset off) while the others run on, and well past the end.
        prices = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0, 0.15, 300)))
        kwargs: dict[str, Any] = {
            "prices": prices,
            "episode_length": 30,
            "lookback": 5,
            "sampler": 7,
            "positions": (-3.0, 0.0, 3.0),
            "max_episode_steps": limit or -1,
            "vector_kwargs": {"autoreset": False},
        }
        native = sekai.make_vec("Trading-v0", 4, **kwargs)
        sync = sekai.make_vec("Trading-v0", 4, vectorization_mode="sync", **kwargs)
        native.reset(seed=0)
        sync.reset(seed=0)
        rng = np.random.default_rng(3)
        finished_at = np.zeros(4, int)
        for t in range(80):
            a = rng.integers(0, 3, 4)
            x, y = native.step(a), sync.step(a)
            np.testing.assert_allclose(x.observations, y.observations, rtol=1e-6)
            np.testing.assert_allclose(x.rewards, y.rewards, rtol=1e-12, atol=1e-15)
            assert x.terminated.tolist() == y.terminated.tolist(), t
            assert x.truncated.tolist() == y.truncated.tolist(), t
            finished_at[(finished_at == 0) & y.dones] = t + 1
        assert native.frozen.all() and (finished_at > 0).all()
        if limit is None:
            assert len(set(finished_at.tolist())) > 1  # lanes froze at different steps
        for name in ("equity", "position", "price", "elapsed_steps"):
            np.testing.assert_allclose(native.get_attr(name), sync.get_attr(name), rtol=1e-12)

    def test_shared_schedule_is_not_split_per_sub_env(self) -> None:
        vec = sekai.make_vec("OptimalExecution-v0", 25)  # n_steps == num_envs == 25
        vec.reset(seed=0)
        schedules = vec.call("almgren_chriss_actions")
        assert len(schedules) == 25 and all(np.shape(a) == (25, 1) for a in schedules)
        np.testing.assert_array_equal(schedules[3], vec.almgren_chriss_actions())

    def test_reward_state_is_reset_per_env(self) -> None:
        vec = PortfolioVecEnv(2, episode_length=3, reward=DifferentialSharpe(0.2))
        vec.reset(seed=0)
        for _ in range(3):
            r = vec.step(np.ones((2, 6), np.float32))
        assert r.dones.all()
        first = vec.step(np.ones((2, 6), np.float32))
        assert np.all(first.rewards == 0.0)  # fresh moving estimates after the autoreset


# ---------------------------------------------------------------------------
# Limit order books
# ---------------------------------------------------------------------------


class TestLimitOrderBook:
    def test_accounting_is_consistent(self) -> None:
        env = LimitOrderBookEnv(inventory_penalty=0.0, session_length=100)
        r = env.reset(seed=0)
        total = 0.0
        while not r.done:
            r = env.step(env.action_space.sample(mask=env.action_mask()))
            total += r.reward
        assert total == pytest.approx(r.info["wealth"])  # rewards sum to final wealth
        assert abs(r.info["inventory"]) <= env.core.max_inventory

    def test_unchanged_quote_keeps_its_queue_position(self) -> None:
        env = LimitOrderBookEnv(flow=STATIC_FLOW, warmup=0.0)
        env.reset(seed=3)
        env.step(np.array([2, 0, 1]))  # rest a bid one tick behind the best
        first_id = env.trader.bid_id
        assert first_id is not None
        env.step(np.array([2, 0, 1]))  # same level: the order (and its priority) is kept
        assert env.trader.bid_id == first_id
        env.step(np.array([3, 0, 1]))  # a different level: cancelled and re-entered
        assert env.trader.bid_id not in (None, first_id) and env.book.get(first_id) is None

    def test_inventory_limits_mask_actions(self) -> None:
        env = LimitOrderBookEnv(max_inventory=2)
        env.reset(seed=0)
        for _ in range(10):
            env.step(np.array([0, 0, 2]))  # keep buying at market
        assert env.trader.inventory == 2
        bid, _, market = env.action_mask()
        assert bid[1:].sum() == 0 and market.tolist() == [1, 1, 0]

    def test_trades_observation_is_a_padded_set(self) -> None:
        env = LimitOrderBookEnv(max_trades=4)
        r = env.reset(seed=0)
        for _ in range(5):
            r = env.step(np.array([1, 1, 1]))
        trades = r.observation["trades"]
        assert trades["values"].shape == (4, 3) and trades["mask"].sum() <= 4

    def test_multi_agent_event_driven_turns(self) -> None:
        market = LimitOrderBookMarket(2, decision_intervals=[1.0, 3.0], session_length=30)
        r = market.reset(seed=0)
        assert set(r.observations) == {"trader_0", "trader_1"}
        turns = {"trader_0": 1, "trader_1": 1}
        while True:
            actions = {
                a: market.action_space(a).sample(mask=market.action_mask(a)) for a in r.observations
            }
            r = market.step(actions)
            assert set(r.rewards) <= {"trader_0", "trader_1"}
            if r.all_done:
                break
            for agent in r.observations:
                turns[agent] += 1
        assert turns["trader_0"] == 30 and turns["trader_1"] == 10

    def test_multi_agent_rejects_out_of_turn_actions(self) -> None:
        market = LimitOrderBookMarket(2, decision_intervals=[1.0, 3.0])
        market.reset(seed=0)
        market.step({a: np.array([0, 0, 1]) for a in ("trader_0", "trader_1")})
        with pytest.raises(ValueError, match="acting"):
            market.step({"trader_1": np.array([0, 0, 1])})

    def test_traders_trade_with_each_other(self) -> None:
        market = LimitOrderBookMarket(2, flow=STATIC_FLOW, warmup=0.0, inventory_penalty=0.0)
        market.reset(seed=0)
        for order in market.book.orders_of(BACKGROUND):  # empty the ask side
            if order.side is Side.ASK:
                market.book.cancel(order.order_id)
        best_bid = market.book.best_bid()
        assert best_bid is not None
        market.step({"trader_0": np.array([0, 1, 1]), "trader_1": np.array([0, 0, 1])})
        maker, taker = market.traders["trader_0"], market.traders["trader_1"]
        assert maker.ask_id is not None
        ask = market.book.get(maker.ask_id)
        assert ask is not None and ask.price == best_bid + 1  # one tick inside the bid
        # Same action: the quote stays put (trader_0 acts first); trader_1 lifts it.
        market.step({"trader_0": np.array([0, 1, 1]), "trader_1": np.array([0, 0, 2])})
        assert (maker.inventory, taker.inventory) == (-1, 1)
        assert maker.cash == pytest.approx(ask.price) and taker.cash == pytest.approx(-ask.price)

    def test_quotes_do_not_walk_when_alone_at_the_best(self) -> None:
        env = LimitOrderBookEnv(flow=STATIC_FLOW, warmup=0.0)
        env.reset(seed=0)
        env.step(np.array([1, 0, 1]))  # join the best bid
        first = env.trader.bid_id
        assert first is not None
        own = env.book.get(first)
        assert own is not None
        for order in env.book.orders_of(BACKGROUND):  # others leave the best bid
            if order.side is Side.BID and order.price == own.price:
                env.book.cancel(order.order_id)
        prices = []
        for _ in range(3):
            env.step(np.array([2, 0, 1]))  # one tick behind the best other bid
            bid = env.book.get(env.trader.bid_id) if env.trader.bid_id is not None else None
            assert bid is not None
            prices.append(bid.price)
        assert prices[0] == prices[1] == prices[2]

    def test_quote_alone_at_the_best_keeps_its_place(self) -> None:
        env = LimitOrderBookEnv(flow=STATIC_FLOW, warmup=0.0)
        env.reset(seed=0)
        env.step(np.array([1, 0, 1]))  # join the best bid, behind the other orders there
        first = env.trader.bid_id
        assert first is not None
        own = env.book.get(first)
        assert own is not None and env.book.queue_ahead(first) > 0
        for order in env.book.orders_of(BACKGROUND):  # the orders ahead leave
            if order.side is Side.BID and order.price == own.price:
                env.book.cancel(order.order_id)
        assert env.book.queue_ahead(first) == 0 and env.book.best_bid() == own.price
        for _ in range(3):  # same action: alone at the best, the quote stays at the front
            env.step(np.array([1, 0, 1]))
            assert env.trader.bid_id == first and env.book.queue_ahead(first) == 0
        assert env.book.best_bid() == own.price
        env.book.limit(Side.BID, own.price + 1, 2)  # another trader improves the best bid
        env.step(np.array([1, 0, 1]))  # level 1 now joins that better price
        moved = env.book.get(env.trader.bid_id) if env.trader.bid_id is not None else None
        assert moved is not None and moved.price == own.price + 1 and env.book.get(first) is None

    def test_constant_action_never_backs_off_the_best(self, monkeypatch: Any) -> None:
        env = LimitOrderBookEnv(max_inventory=10**6)  # no inventory limit forces a cancel
        book = env.book  # one book for the env's lifetime
        cancel = book.cancel
        backoffs: list[int] = []

        def watched_cancel(order_id: int) -> bool:
            order = book.get(order_id)
            if order is not None and order.owner == env.AGENT:
                best = book.best_bid() if order.side is Side.BID else book.best_ask()
                if order.price == best:  # a level-1 quote pulled away from the best price
                    backoffs.append(order_id)
            return cancel(order_id)

        monkeypatch.setattr(book, "cancel", watched_cancel)
        moves = 0
        for seed in range(10):
            env.reset(seed=seed)
            for _ in range(300):
                ids = (env.trader.bid_id, env.trader.ask_id)
                env.step(np.array([1, 1, 1]))
                moves += sum(a != b for a, b in zip(ids, (env.trader.bid_id, env.trader.ask_id)))
        assert moves > 100 and backoffs == []  # quotes do follow the market, never off the best

    @pytest.mark.parametrize("session_length, n_steps, last", [(10.5, 11, 0.5), (10.0, 34, 0.1)])
    def test_dt_is_the_time_each_step_covers(
        self, session_length: float, n_steps: int, last: float
    ) -> None:
        interval = 1.0 if session_length == 10.5 else 0.3
        env = LimitOrderBookEnv(session_length=session_length, decision_interval=interval)
        r = env.reset(seed=0)
        dts = []
        while not r.done:
            r = env.step(np.array([1, 1, 1]))
            dts.append(r.info["dt"])
        assert env.n_steps == len(dts) == n_steps
        assert dts[:-1] == [interval] * (n_steps - 1)  # exactly, not up to rounding
        assert dts[-1] == pytest.approx(last) and sum(dts) == pytest.approx(session_length)

    def test_multi_agent_dt_is_exact(self) -> None:
        for intervals, expected in (([0.3], {0.3}), ([1.0, 3.0], {1.0}), ([0.3, 0.3], {0.3})):
            market = LimitOrderBookMarket(
                len(intervals), decision_intervals=intervals, session_length=9.0
            )
            r = market.reset(seed=0)
            dts = []
            while not r.all_done:
                r = market.step({a: np.array([1, 1, 1]) for a in r.observations})
                dts.append(r.info["trader_0"]["dt"])
            assert set(dts) == expected and sum(dts) == pytest.approx(9.0)

    def test_inventory_penalty_accrues_per_unit_time(self) -> None:
        totals = []
        for intervals in ([1.0, 1.0], [1.0, 0.1]):
            market = LimitOrderBookMarket(
                2, flow=STATIC_FLOW, warmup=0.0, decision_intervals=intervals, session_length=20
            )
            r = market.reset(seed=0)
            market.traders["trader_0"].inventory = 1  # hold one lot, never trade
            total = 0.0
            while not r.all_done:
                r = market.step({a: np.array([0, 0, 1]) for a in r.observations})
                total += r.rewards["trader_0"]
            totals.append(total)
        assert totals[0] == pytest.approx(totals[1]) == pytest.approx(-0.01 * 20, rel=0.02)

    def test_market_order_cancels_own_opposite_quote(self) -> None:
        env = LimitOrderBookEnv(flow=STATIC_FLOW, warmup=0.0, inventory_penalty=0.0)
        env.reset(seed=0)
        env.step(np.array([0, 1, 1]))  # rest an ask at the best ask
        assert env.trader.ask_id is not None
        own = env.book.get(env.trader.ask_id)
        assert own is not None
        for order in env.book.orders_of(BACKGROUND):  # leave it first in the queue
            if order.side is Side.ASK and order.price == own.price:
                env.book.cancel(order.order_id)
        assert env.book.best_ask() == own.price and env.book.queue_ahead(own.order_id) == 0
        r = env.step(np.array([0, 0, 2]))  # no ask, buy at market
        # Self-trade prevention: the own ask is cancelled first, so the buy lifts the next ask
        # (it used to fill against the agent's own ask: inventory 0, cash 0).
        assert r.info["inventory"] == 1 and r.info["fills"] == 1
        assert r.info["cash"] < -own.price and env.book.get(own.order_id) is None
        assert env.trader.ask_id is None
        env.step(np.array([0, 1, 1]))
        again = env.book.get(env.trader.ask_id) if env.trader.ask_id is not None else None
        assert again is not None
        r = env.step(np.array([0, 1, 2]))  # keep an ask and buy: the ask is placed anew
        assert env.trader.ask_id not in (None, again.order_id) and r.info["inventory"] == 2

    def test_no_self_trades_and_accounting_holds(self, monkeypatch: Any) -> None:
        env = LimitOrderBookEnv()
        book = env.book
        fills: list[Any] = []
        for name in ("market", "limit"):
            method = getattr(book, name)

            def logged(*args: Any, _method: Any = method, **kwargs: Any) -> Any:
                out = _method(*args, **kwargs)
                fills.extend(out[1])
                return out

            monkeypatch.setattr(book, name, logged)
        agent_fills = 0
        for seed in range(5):
            r = env.reset(seed=seed)
            env.action_space.seed(seed)
            fills.clear()
            total = penalties = 0.0
            while not r.done:
                r = env.step(env.action_space.sample(mask=env.action_mask()))
                total += r.reward
                penalties += env.core.inventory_penalty * r.info["inventory"] ** 2 * r.info["dt"]
            agent = [f for f in fills if env.AGENT in (f.maker_owner, f.taker_owner)]
            agent_fills += len(agent)
            assert not [f for f in fills if f.maker_owner == f.taker_owner == env.AGENT]
            assert total + penalties == pytest.approx(r.info["wealth"])  # rewards conserve wealth
        assert agent_fills > 100

    def test_multi_agent_traders_never_trade_with_themselves(self) -> None:
        market = LimitOrderBookMarket(2, flow=STATIC_FLOW, warmup=0.0, inventory_penalty=0.0)
        market.reset(seed=0)
        for order in market.book.orders_of(BACKGROUND):  # empty the ask side
            if order.side is Side.ASK:
                market.book.cancel(order.order_id)
        best_bid = market.book.best_bid()
        assert best_bid is not None
        # trader_0 rests the best ask (one tick inside the bid), trader_1 one tick behind it.
        market.step({"trader_0": np.array([0, 1, 1]), "trader_1": np.array([0, 2, 1])})
        t0, t1 = market.traders["trader_0"], market.traders["trader_1"]
        asks = [market.book.get(t.ask_id) for t in (t0, t1) if t.ask_id is not None]
        assert [a.price for a in asks if a is not None] == [best_bid + 1, best_bid + 2]
        market.step({"trader_0": np.array([0, 0, 2]), "trader_1": np.array([0, 0, 1])})
        assert (t0.inventory, t1.inventory) == (1, -1)  # bought trader_1's ask, not its own
        assert t0.cash == pytest.approx(-(best_bid + 2)) and t0.ask_id is None

    def test_multi_agent_invalid_action_changes_nothing(self) -> None:
        market = LimitOrderBookMarket(2)
        market.reset(seed=0)
        state = (market.book.depth(5), market.next_decision.copy(), market._clock, len(market.book))
        with pytest.raises(sekai.InvalidAction):
            market.step({"trader_0": np.array([0, 0, 2]), "trader_1": np.array([9, 9, 9])})
        # trader_0's market buy used to execute before trader_1's action was checked.
        assert market.traders["trader_0"].inventory == 0 and market.traders["trader_0"].cash == 0
        assert market._clock == state[2] and len(market.book) == state[3]
        np.testing.assert_array_equal(market.next_decision, state[1])
        for got, expected in zip(market.book.depth(5), state[0]):
            np.testing.assert_array_equal(got, expected)
        market.step({"trader_0": np.array([0, 0, 2]), "trader_1": np.array([0, 0, 1])})
        assert market.traders["trader_0"].inventory == 1  # the retry buys once, not twice
