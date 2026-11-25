"""Draw every finance environment as a dashboard and save each episode as a GIF.

Each finance environment has a renderer in `sekai.envs.finance` (TradingRenderer,
PortfolioRenderer, ...). Here each one runs with a simple baseline policy, recorded
with RecordFrames. The renderers draw their header in the Qian Capital house style;
`branding=False` draws the plain header of sekai 0.2 instead, and each episode is saved
both ways, in the light theme (the default) and the dark one (`finance_trading.gif`,
`finance_trading_dark.gif`, `finance_trading_plain.gif`, `finance_trading_plain_dark.gif`,
...), followed by stills of the last Trading-v0 frame in each style and theme. Needs
Pillow: pip install "sekai[render]".

Pass a directory to keep the images (this is how the documentation's animations are
made):

    python examples/render_finance.py docs/assets
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import sekai
from sekai.envs.finance import (
    LimitOrderBookMarket,
    LimitOrderBookMarketRenderer,
    LimitOrderBookRenderer,
    MarketMakingRenderer,
    MarketRenderer,
    OptimalExecutionRenderer,
    PortfolioRenderer,
    TradingRenderer,
)
from sekai.rendering import RecordFrames, save_gif

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
out.mkdir(parents=True, exist_ok=True)
size = {"width": 600, "height": 375}


def run(make_env, renderers, policy, seed):
    """Run one episode with `policy(base_env)`, each renderer recording a frame per step.

    Returns one RecordFrames per renderer, in the same order."""
    env = make_env()
    recordings = []
    for renderer in renderers:
        env = RecordFrames(env, renderer)
        recordings.append(env)
    result = env.reset(seed=seed)
    while not result.done:
        result = env.step(policy(env.unwrapped))
    return recordings


def record(name, make_env, renderer_cls, policy, seed):
    """Save one episode drawn branded (the default) and plain (`branding=False`), each in
    the light theme (the default) and the dark one."""
    for branding, style in ((True, ""), (False, "_plain")):
        renderers = [renderer_cls(theme=t, branding=branding, **size) for t in ("light", "dark")]
        recordings = run(make_env, renderers, policy, seed)
        for recording, suffix in zip(recordings, (style, f"{style}_dark")):
            path = recording.save(out / f"finance_{name}{suffix}.gif")
            print(f"{name}{suffix}: {len(recording.frames)} frames -> {path}")
        recordings[-1].close()


def momentum(env):
    """Long after a 20-day rise, short after a fall (Trading-v0 positions are -1, 0, 1)."""
    prices = env.price_history
    return 2 if prices[-1] > prices[-21] else 0


def momentum_tilt(env):
    """Weights tilted towards the recent winners, 10% in cash."""
    prices = env.price_history
    scores = np.exp(10 * np.log(prices[-1] / prices[-21]))
    return np.append(0.9 * scores / scores.sum(), 0.1).astype(np.float32)


def equal_weight(env):
    """Equal weights across whatever is tradable now, a little cash."""
    scores = env.action_mask().astype(np.float32)
    scores[-1] = 0.1
    return scores


def skewed_quotes(env):
    """Quote at the best prices, stepping back on the side that would add to inventory."""
    inventory = env.trader.inventory
    return np.array([1 + (inventory > 2), 1 + (inventory < -2), 1])


def make_trading():
    return sekai.make("Trading-v0", episode_length=120)


record("trading", make_trading, TradingRenderer, momentum, seed=3)
record(
    "portfolio",
    lambda: sekai.make("Portfolio-v0", episode_length=120),
    PortfolioRenderer,
    momentum_tilt,
    seed=2,
)
record(
    "market",
    lambda: sekai.make("Market-v0", episode_length=160),
    MarketRenderer,
    equal_weight,
    seed=11,
)
record(
    "market_making",
    lambda: sekai.make("MarketMaking-v0", n_steps=120),
    MarketMakingRenderer,
    lambda env: env.avellaneda_stoikov_action(),
    seed=0,
)
record(
    "optimal_execution",
    lambda: sekai.make("OptimalExecution-v0", risk_aversion=2e-6),
    OptimalExecutionRenderer,
    lambda env: env.almgren_chriss_actions()[env.elapsed_steps],
    seed=0,
)
record(
    "limit_order_book",
    lambda: sekai.make("LimitOrderBook-v0", session_length=120),
    LimitOrderBookRenderer,
    skewed_quotes,
    seed=1,
)

# RecordFrames wraps single-agent environments, so a multi-agent one is drawn directly.
market = LimitOrderBookMarket(n_agents=3, decision_intervals=[1.0, 1.5, 2.5], session_length=90)
renderers = {
    "": LimitOrderBookMarketRenderer(**size),
    "_dark": LimitOrderBookMarketRenderer(theme="dark", **size),
}
rng = np.random.default_rng(0)
market.reset(seed=0)
frames = {suffix: [renderer.render(market)] for suffix, renderer in renderers.items()}
while market.agents:
    actions = {
        a: np.array([1 + rng.integers(2), 1 + rng.integers(2), rng.choice([0, 1, 1, 1, 2])])
        for a in market.acting
    }
    market.step(actions)
    for suffix, renderer in renderers.items():
        frames[suffix].append(renderer.render(market))
for suffix, theme_frames in frames.items():
    path = save_gif(theme_frames, out / f"finance_order_book_market{suffix}.gif", fps=10)
    print(f"order_book_market{suffix}: {len(theme_frames)} frames -> {path}")

# The last frame of the Trading-v0 episode in each style and theme, as stills.
for theme in ("light", "dark"):
    for branding, style in ((True, "branded"), (False, "plain")):
        (env,) = run(
            make_trading, [TradingRenderer(theme=theme, branding=branding, **size)], momentum, 3
        )
        path = out / f"branding_{style}_{theme}.png"
        Image.fromarray(env.frames[-1]).save(path, optimize=True)
        env.close()
        print(f"still: {path}")
