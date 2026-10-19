from __future__ import annotations

from collections.abc import Iterator

import pytest

import sekai
from sekai.registry import EnvRegistry, EnvSpec, WrapperSpec
from tests.conftest import CountingEnv


@pytest.fixture
def registry() -> Iterator[EnvRegistry]:
    yield EnvRegistry()


class TestEnvSpec:
    @pytest.mark.parametrize(
        ("env_id", "namespace", "name", "version"),
        [
            ("CartPole-v1", None, "CartPole", 1),
            ("qc/MarketEnv-v0", "qc", "MarketEnv", 0),
            ("my_ns/CustomEnv", "my_ns", "CustomEnv", None),
            ("Foo-Bar-v12", None, "Foo-Bar", 12),
            ("Plain", None, "Plain", None),
        ],
    )
    def test_id_parsing(
        self, env_id: str, namespace: str | None, name: str, version: int | None
    ) -> None:
        spec = EnvSpec(env_id)
        assert (spec.namespace, spec.name, spec.version) == (namespace, name, version)

    def test_invalid_id(self) -> None:
        with pytest.raises(ValueError):
            EnvSpec("bad id!")

    def test_json_round_trip(self) -> None:
        spec = EnvSpec(
            "ns/Thing-v2",
            entry_point="tests.conftest:CountingEnv",
            max_episode_steps=7,
            kwargs={"horizon": 3},
            additional_wrappers=(WrapperSpec("O", "sekai.wrappers:OrderEnforcing"),),
        )
        rebuilt = EnvSpec.from_json(spec.to_json())
        assert rebuilt.id == spec.id and rebuilt.kwargs == {"horizon": 3}
        assert rebuilt.additional_wrappers == spec.additional_wrappers
        assert "max_episode_steps=7" in spec.pprint()


class TestRegistry:
    def test_register_make(self, registry: EnvRegistry) -> None:
        registry.register(
            "Count-v0", "tests.conftest:CountingEnv", max_episode_steps=2, kwargs={"horizon": 9}
        )
        env = registry.make("Count-v0")
        assert isinstance(env, sekai.TimeLimit) and env.max_episode_steps == 2
        assert env.unwrapped.horizon == 9
        assert env.spec.id == "Count-v0"

    def test_make_overrides(self, registry: EnvRegistry) -> None:
        registry.register("Count-v0", CountingEnv, max_episode_steps=2)
        env = registry.make("Count-v0", horizon=4, max_episode_steps=10)
        assert env.max_episode_steps == 10 and env.unwrapped.horizon == 4
        assert env.spec.kwargs == {"horizon": 4} and env.spec.max_episode_steps == 10
        rebuilt = env.spec.make()
        assert rebuilt.max_episode_steps == 10 and rebuilt.unwrapped.horizon == 4

    def test_make_from_spec(self, registry: EnvRegistry) -> None:
        spec = registry.register("Count-v0", CountingEnv)
        assert isinstance(registry.make(spec), CountingEnv)

    def test_additional_wrappers(self, registry: EnvRegistry) -> None:
        registry.register(
            "Count-v0",
            CountingEnv,
            additional_wrappers=(
                WrapperSpec("Order", "sekai.wrappers.order_enforcing:OrderEnforcing"),
            ),
        )
        env = registry.make("Count-v0")
        with pytest.raises(sekai.ResetRequired):
            env.step(0)

    def test_duplicate_and_missing(self, registry: EnvRegistry) -> None:
        registry.register("Count-v0", CountingEnv)
        with pytest.raises(sekai.EnvAlreadyRegistered):
            registry.register("Count-v0", CountingEnv)
        with pytest.raises(sekai.EnvNotFound) as exc:
            registry.spec("Cuont-v0")
        assert "Count-v0" in exc.value.suggestions

    def test_deregister(self, registry: EnvRegistry) -> None:
        registry.register("Count-v0", CountingEnv)
        registry.deregister("Count-v0")
        assert "Count-v0" not in registry and len(registry) == 0

    def test_namespace_loader_called_once(self, registry: EnvRegistry) -> None:
        calls = []

        def loader() -> None:
            calls.append(1)
            registry.register("plug/Count-v0", CountingEnv)

        registry.register_namespace("plug", loader)
        registry.make("plug/Count-v0")
        registry.make("plug/Count-v0")
        assert calls == [1]

    def test_entry_point_validation(self, registry: EnvRegistry) -> None:
        registry.register("Bad-v0", "no_colon_here")
        with pytest.raises(ValueError):
            registry.make("Bad-v0")

    def test_factory_returning_wrapped_env(self, registry: EnvRegistry) -> None:
        registry.register("Wrapped-v0", lambda: sekai.OrderEnforcing(CountingEnv()))
        env = registry.make("Wrapped-v0")
        assert env.spec.id == "Wrapped-v0" and env.unwrapped.spec.id == "Wrapped-v0"

    def test_pprint(self, registry: EnvRegistry) -> None:
        registry.register("a/X-v0", CountingEnv)
        registry.register("Y-v0", CountingEnv)
        text = registry.pprint()
        assert "[a]" in text and "Y-v0" in text


def test_builtins_registered() -> None:
    ids = {s.id for s in sekai.all_specs()}
    assert {
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "Acrobot-v1",
        "Trading-v0",
        "Portfolio-v0",
        "MarketMaking-v0",
        "OptimalExecution-v0",
    } <= ids


def test_global_register_deregister() -> None:
    sekai.register("tests/Count-v0", CountingEnv)
    try:
        assert isinstance(sekai.make("tests/Count-v0"), CountingEnv)
    finally:
        sekai.deregister("tests/Count-v0")
    with pytest.raises(sekai.EnvNotFound):
        sekai.spec("tests/Count-v0")


def test_namespace_loader_can_be_retried(registry: EnvRegistry) -> None:
    attempts = []

    def flaky_loader() -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise ImportError("transient")
        registry.register("plug/Count-v0", CountingEnv)

    registry.register_namespace("plug", flaky_loader)
    with pytest.raises(ImportError):
        registry.make("plug/Count-v0")
    assert isinstance(registry.make("plug/Count-v0"), CountingEnv)


def test_make_can_remove_time_limit(registry: EnvRegistry) -> None:
    registry.register("Count-v0", CountingEnv, max_episode_steps=2)
    env = registry.make("Count-v0", max_episode_steps=-1)
    assert isinstance(env, CountingEnv) and env.spec.max_episode_steps is None
