from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
import pytest

import sekai

torch = pytest.importorskip("torch")
from sekai.nn import MaskedSetEncoder, masked_max, masked_mean  # noqa: E402

ENCODERS = [
    pytest.param({"element_encoder": "mlp"}, id="mlp"),
    pytest.param({"element_encoder": "mlp", "attention_layers": 2}, id="mlp-attention"),
    pytest.param({"element_encoder": "conv", "hidden_sizes": (8, 8)}, id="conv"),
    pytest.param({"element_encoder": "transformer", "attention_layers": 1}, id="transformer"),
]
SHAPE = (6, 3)  # an (L, F) window, as in Market-v0


def make(**kwargs: Any) -> MaskedSetEncoder:
    torch.manual_seed(0)
    return MaskedSetEncoder(SHAPE, 16, pooling=("mean", "max", "sum"), **kwargs).double()


def batch(size: int = 5, seed: int = 0) -> tuple[Any, Any]:
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(4, size, *SHAPE, generator=g, dtype=torch.float64)
    mask = torch.rand(4, size, generator=g) < 0.6
    mask[0] = True  # a full set
    mask[1] = False  # an empty one
    mask[2, :2] = torch.tensor([True, False])
    return values, mask


def test_import_sekai_does_not_import_torch() -> None:
    code = "import sys, sekai; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_dependencies_raise_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.delitem(sys.modules, "sekai.nn.sb3", raising=False)
    for name in [
        "stable_baselines3",
        *[m for m in sys.modules if m.startswith("stable_baselines3.")],
    ]:
        monkeypatch.setitem(sys.modules, name, None)  # as if it were not installed
    with pytest.raises(ImportError, match="pip install stable-baselines3"):
        importlib.import_module("sekai.nn.sb3")
    monkeypatch.delitem(sys.modules, "sekai.nn")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"sekai\[torch\]"):
        importlib.import_module("sekai.nn")


@pytest.mark.parametrize("kwargs", ENCODERS)
class TestMaskedSetEncoder:
    def test_shapes(self, kwargs: dict[str, Any]) -> None:
        encoder = make(**kwargs)
        out = encoder(*batch())
        assert out.slots.shape == (4, 5, 16) and out.pooled.shape == (4, encoder.pooled_dim)
        assert encoder.pooled_dim == 48

    def test_permutation_equivariant_and_invariant(self, kwargs: dict[str, Any]) -> None:
        encoder = make(**kwargs)
        values, mask = batch()
        out = encoder(values, mask)
        perm = torch.randperm(5, generator=torch.Generator().manual_seed(1))
        moved = encoder(values[:, perm], mask[:, perm])
        torch.testing.assert_close(moved.slots, out.slots[:, perm])
        torch.testing.assert_close(moved.pooled, out.pooled)

    def test_padding_has_no_effect(self, kwargs: dict[str, Any]) -> None:
        encoder = make(**kwargs)
        values, mask = batch()
        out = encoder(values, mask)
        loss = out.pooled.square().sum() + out.slots.sum()
        grads = torch.autograd.grad(loss, list(encoder.parameters()))
        junk = values.clone()
        junk[~mask] = torch.tensor([np.nan, np.inf, -1e30], dtype=torch.float64)
        other = encoder(junk, mask)
        assert torch.equal(other.slots, out.slots) and torch.equal(other.pooled, out.pooled)
        other_grads = torch.autograd.grad(
            other.pooled.square().sum() + other.slots.sum(), list(encoder.parameters())
        )
        assert all(torch.equal(a, b) for a, b in zip(grads, other_grads))
        assert torch.equal(out.slots[~mask], torch.zeros_like(out.slots[~mask]))

    def test_extra_empty_slots_change_nothing(self, kwargs: dict[str, Any]) -> None:
        encoder = make(**kwargs)
        values, mask = batch()
        wider = encoder(
            torch.cat([values, torch.randn(4, 3, *SHAPE, dtype=torch.float64)], dim=1),
            torch.cat([mask, torch.zeros(4, 3, dtype=torch.bool)], dim=1),
        )
        out = encoder(values, mask)
        torch.testing.assert_close(wider.pooled, out.pooled)
        torch.testing.assert_close(wider.slots[:, :5], out.slots)

    def test_empty_sets(self, kwargs: dict[str, Any]) -> None:
        encoder = make(**kwargs)
        values = torch.randn(2, 5, *SHAPE, dtype=torch.float64, requires_grad=True)
        out = encoder(values, torch.zeros(2, 5, dtype=torch.int8))
        assert torch.equal(out.pooled, torch.zeros_like(out.pooled))
        assert torch.equal(out.slots, torch.zeros_like(out.slots))
        (out.pooled.sum() + out.slots.sum()).backward()
        assert values.grad is not None and torch.equal(values.grad, torch.zeros_like(values))
        assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)

    def test_gradients_reach_every_parameter(self, kwargs: dict[str, Any]) -> None:
        encoder = make(**kwargs)
        values, mask = batch()
        values.requires_grad_(True)
        out = encoder(values, mask)
        (out.pooled.square().sum() + out.slots.square().sum()).backward()
        for name, p in encoder.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
        assert values.grad is not None
        assert torch.equal(values.grad[~mask], torch.zeros_like(values.grad[~mask]))

    def test_mask_dtypes(self, kwargs: dict[str, Any]) -> None:
        encoder = make(**kwargs)
        values, mask = batch()
        expected = encoder(values, mask).pooled
        for as_type in (torch.int8, torch.float32, torch.int64):
            torch.testing.assert_close(encoder(values, mask.to(as_type)).pooled, expected)

    def test_inference_mode_matches(self, kwargs: dict[str, Any]) -> None:
        # PyTorch's transformer fast path runs in eval mode without gradients (SB3's predict).
        encoder = make(**kwargs)
        values, mask = batch()
        expected = encoder(values, mask)
        encoder.eval()
        with torch.no_grad():
            out = encoder(values, mask)
            junk = encoder(torch.where(mask[..., None, None], values, 1e6), mask)
        torch.testing.assert_close(out.pooled, expected.pooled)
        assert torch.equal(junk.pooled, out.pooled) and torch.isfinite(out.pooled).all()


def test_masked_pooling_helpers() -> None:
    x = torch.tensor([[[1.0, -5.0], [3.0, 7.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    torch.testing.assert_close(masked_mean(x, mask), torch.tensor([[2.0, 1.0]]))
    torch.testing.assert_close(masked_max(x, mask), torch.tensor([[3.0, 7.0]]))
    empty = torch.zeros_like(mask)
    assert torch.equal(masked_mean(x, empty), torch.zeros(1, 2))
    assert torch.equal(masked_max(x, empty), torch.zeros(1, 2))


def test_custom_element_encoder_and_single_pooling() -> None:
    encoder = MaskedSetEncoder(3, 8, element_encoder=torch.nn.Linear(3, 8), pooling="max")
    out = encoder(torch.randn(2, 4, 3), torch.ones(2, 4))
    assert out.pooled.shape == (2, 8) and encoder.pooled_dim == 8


def test_integer_values() -> None:
    encoder = MaskedSetEncoder((2,), 4)
    out = encoder(torch.ones(1, 3, 2, dtype=torch.int64), torch.ones(1, 3))
    assert out.pooled.dtype == torch.float32


def test_validation() -> None:
    with pytest.raises(ValueError, match=r"\(L, F\)"):
        MaskedSetEncoder((4,), element_encoder="conv")
    with pytest.raises(ValueError, match="pooling"):
        MaskedSetEncoder((4,), pooling=("mean", "median"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_heads"):
        MaskedSetEncoder((4,), 10, attention_layers=1, num_heads=4)
    with pytest.raises(ValueError, match="element_encoder"):
        MaskedSetEncoder((4,), element_encoder="gru")  # type: ignore[arg-type]
    encoder = MaskedSetEncoder((2, 3))
    with pytest.raises(ValueError, match="values must have shape"):
        encoder(torch.zeros(1, 4, 3, 2), torch.ones(1, 4))
    with pytest.raises(ValueError, match="mask must have shape"):
        encoder(torch.zeros(1, 4, 2, 3), torch.ones(1, 5))


# ---------------------------------------------------------------------------
# Stable-Baselines3
# ---------------------------------------------------------------------------


@pytest.fixture
def sb3() -> Any:
    pytest.importorskip("gymnasium")
    return pytest.importorskip("stable_baselines3")


def market(env_id: str = "Market-v0", **kwargs: Any) -> Any:
    from sekai.compat import to_gymnasium

    return to_gymnasium(sekai.UnnestObservation(sekai.make(env_id, **kwargs)))


def observations(env: Any, n: int = 3, *, preprocess: bool = True) -> dict[str, Any]:
    """A batch of `n` observations as tensors, preprocessed as SB3 hands them to extractors."""
    from stable_baselines3.common.preprocessing import preprocess_obs
    from stable_baselines3.common.utils import obs_as_tensor

    env.action_space.seed(0)
    obs = [env.reset(seed=0)[0]]
    obs += [env.step(env.action_space.sample())[0] for _ in range(n - 1)]
    stacked = {key: np.stack([o[key] for o in obs]) for key in obs[0]}
    tensors: dict[str, Any] = obs_as_tensor(stacked, torch.device("cpu"))
    if not preprocess:
        return tensors
    out: dict[str, Any] = preprocess_obs(tensors, env.observation_space)
    return out


def permute_slots(obs: dict[str, Any], prefix: str, perm: Any) -> dict[str, Any]:
    return {
        key: value[:, perm] if key.startswith(prefix + "/") else value for key, value in obs.items()
    }


class TestMarketFeaturesExtractor:
    def test_groups_sets_and_plain_entries(self, sb3: Any) -> None:
        from sekai.nn.sb3 import MarketFeaturesExtractor

        lob = MarketFeaturesExtractor(market("LimitOrderBook-v0").observation_space, 32)
        assert lob.set_names == ("trades",)
        assert lob.plain_keys == ("book", "agent")
        assert lob.features_dim == lob.context_dim == 2 * 32 + 64
        env = market()
        extractor = MarketFeaturesExtractor(env.observation_space, slot_set="assets")
        assert extractor.set_keys == (("assets/values", "assets/mask"),)
        assert extractor.features_dim == extractor.context_dim + 10 * 64

    def test_features_ignore_slot_order(self, sb3: Any) -> None:
        from sekai.nn.sb3 import MarketFeaturesExtractor

        env = market()
        obs = observations(env)
        torch.manual_seed(0)
        extractor = MarketFeaturesExtractor(
            env.observation_space,
            16,
            slot_set="assets",
            set_kwargs={"element_encoder": "conv", "attention_layers": 1},
        )
        enc = extractor.encode(obs)
        perm = torch.randperm(10, generator=torch.Generator().manual_seed(0))
        moved = extractor.encode(permute_slots(obs, "assets", perm))
        torch.testing.assert_close(moved.context, enc.context)
        torch.testing.assert_close(moved.sets["assets"].slots, enc.sets["assets"].slots[:, perm])
        assert torch.equal(extractor(obs), enc.features)
        assert enc.features.shape == (3, extractor.features_dim)

    def test_validation(self, sb3: Any) -> None:
        from gymnasium import spaces

        from sekai.nn.sb3 import MarketFeaturesExtractor

        with pytest.raises(ValueError, match="slot_set 'stocks'"):
            MarketFeaturesExtractor(market().observation_space, slot_set="stocks")
        with pytest.raises(TypeError, match="Dict"):
            MarketFeaturesExtractor(spaces.Box(0, 1, (3,)))  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPPO:
    def test_learns_with_the_extractor(self, sb3: Any) -> None:
        from stable_baselines3 import PPO

        from sekai.nn.sb3 import MarketFeaturesExtractor

        policy_kwargs = {
            "features_extractor_class": MarketFeaturesExtractor,
            "features_extractor_kwargs": {"set_kwargs": {"element_encoder": "conv"}},
        }
        model = PPO(
            "MultiInputPolicy", market(), n_steps=128, batch_size=64, policy_kwargs=policy_kwargs
        )
        before = [p.detach().clone() for p in model.policy.parameters()]
        model.learn(total_timesteps=256)
        after = list(model.policy.parameters())
        assert all(torch.isfinite(p).all() for p in after)
        assert any(not torch.equal(a, b) for a, b in zip(before, after))

    def test_learns_on_the_order_book(self, sb3: Any) -> None:
        from stable_baselines3 import PPO

        from sekai.nn.sb3 import MarketFeaturesExtractor

        policy_kwargs = {"features_extractor_class": MarketFeaturesExtractor}
        model = PPO(
            "MultiInputPolicy",
            market("LimitOrderBook-v0"),
            n_steps=128,
            batch_size=64,
            policy_kwargs=policy_kwargs,
            seed=0,
        )
        model.learn(total_timesteps=256)

    def test_slot_policy(self, sb3: Any, tmp_path: Any) -> None:
        from stable_baselines3 import PPO

        from sekai.nn.sb3 import SlotActorCriticPolicy

        env = market()
        model = PPO(SlotActorCriticPolicy, env, n_steps=128, batch_size=64, seed=0)
        model.learn(total_timesteps=256)
        policy = model.policy
        obs = observations(env, preprocess=False)
        perm = torch.randperm(10, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            mean = policy.get_distribution(obs).distribution.mean
            moved = policy.get_distribution(permute_slots(obs, "assets", perm)).distribution.mean
            torch.testing.assert_close(moved[:, :10], mean[:, perm])  # each asset's own score
            torch.testing.assert_close(moved[:, 10], mean[:, 10])  # cash
            torch.testing.assert_close(
                policy.predict_values(permute_slots(obs, "assets", perm)),
                policy.predict_values(obs),
            )
        model.save(tmp_path / "slot.zip")
        loaded = PPO.load(tmp_path / "slot.zip")
        first, _ = env.reset(seed=1)
        np.testing.assert_allclose(
            loaded.predict(first, deterministic=True)[0],
            model.predict(first, deterministic=True)[0],
        )

    def test_slot_policy_without_extra_actions(self, sb3: Any) -> None:
        from stable_baselines3 import PPO

        from sekai.nn.sb3 import SlotActorCriticPolicy

        env = market(allow_short=True)  # Box(-1, 1, (10,)): one weight per slot, no cash
        model = PPO(SlotActorCriticPolicy, env, n_steps=64, batch_size=32, seed=0)
        model.learn(total_timesteps=64)
        assert model.predict(env.reset(seed=0)[0])[0].shape == (10,)

    def test_slot_policy_validation(self, sb3: Any) -> None:
        from gymnasium import spaces
        from stable_baselines3 import PPO

        from sekai.nn.sb3 import SlotActorCriticPolicy

        lob = market("LimitOrderBook-v0")
        with pytest.raises(TypeError, match="1-D Box"):
            PPO(SlotActorCriticPolicy, lob, policy_kwargs={"slot_set": "trades"})
        with pytest.raises(ValueError, match="pi"):
            PPO(SlotActorCriticPolicy, market(), policy_kwargs={"net_arch": {"pi": [64]}})
        with pytest.raises(ValueError, match="fewer entries"):
            SlotActorCriticPolicy(
                market().observation_space, spaces.Box(0, 1, (3,)), lambda _: 1e-3
            )
