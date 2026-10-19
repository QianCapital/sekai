"""Command-line interface: `sekai list | info | check | bench`, or `python -m sekai ...`."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from collections.abc import Sequence
from typing import Any

import sekai


def _import_modules(modules: Sequence[str]) -> None:
    """Import plugin modules so their environments register before we look them up.

    The current directory is searched too, as with `python -m`, so `sekai -m my_envs list`
    works from a project checkout.
    """
    if modules and os.getcwd() not in sys.path and "" not in sys.path:
        sys.path.insert(0, os.getcwd())
    for module in modules:
        importlib.import_module(module)


def _cmd_list(args: argparse.Namespace) -> int:
    specs = sorted(sekai.all_specs(), key=lambda s: s.id)
    if args.namespace is not None:
        specs = [s for s in specs if (s.namespace or "") == args.namespace]
    if not specs:
        if args.namespace is not None:
            print(f"No environments in namespace {args.namespace!r}.")
        else:
            print("No environments registered.")
        return 0
    width = max(len(s.id) for s in specs)
    print(f"{'ID':<{width}}  {'MAX STEPS':>9}  ENTRY POINT")
    for s in specs:
        steps = "-" if s.max_episode_steps is None else str(s.max_episode_steps)
        entry = s.entry_point if isinstance(s.entry_point, str) else repr(s.entry_point)
        native = "  [native vec]" if s.vector_entry_point is not None else ""
        print(f"{s.id:<{width}}  {steps:>9}  {entry}{native}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    spec = sekai.spec(args.env_id)
    print(spec.pprint())
    env = sekai.make(args.env_id)
    try:
        print(f"observation_space={env.observation_space!r}")
        print(f"action_space={env.action_space!r}")
        if env.metadata:
            print(f"metadata={env.metadata}")
        layers = [type(layer).__name__ for layer in env] if isinstance(env, sekai.Wrapper) else []
        if layers:
            print("wrappers=" + " -> ".join(layers))
    finally:
        env.close()
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    env = sekai.make(args.env_id)
    try:
        issues = sekai.check_env(env, n_steps=args.steps, check_determinism=args.determinism)
    finally:
        env.close()
    for issue in issues:
        print(issue)
    errors = [i for i in issues if i.startswith("ERROR")]
    if not issues:
        print(f"{args.env_id}: all checks passed.")
    return 1 if errors or (args.strict and issues) else 0


def _cmd_bench(args: argparse.Namespace) -> int:
    mode = None if args.mode == "auto" else args.mode
    if args.workers is not None and mode != "proc":
        print("error: --workers applies to --mode proc only", file=sys.stderr)
        return 2
    if args.num_envs > 1 or mode not in (None, "sync"):
        vector_kwargs = {"num_workers": args.workers} if args.workers else None
        try:
            venv = sekai.make_vec(
                args.env_id,
                num_envs=args.num_envs,
                vectorization_mode=mode,
                vector_kwargs=vector_kwargs,
            )
        except ValueError as e:  # e.g. --mode native for an env without a native implementation
            print(f"error: {e}", file=sys.stderr)
            return 2
        target: Any = venv
        label = f"{type(venv).__name__} x{args.num_envs}"
        venv.reset(seed=args.seed)
        actions = venv.batch_action_space
        actions.seed(args.seed)

        def step_once() -> None:
            venv.step(actions.sample())

        steps_per_call = venv.num_envs
    else:
        env = sekai.make(args.env_id)
        target = env
        label = "single env"
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)

        def step_once() -> None:
            if env.step(env.action_space.sample()).done:
                env.reset()

        steps_per_call = 1

    try:
        for _ in range(min(args.steps // 10, 100)):  # warm-up
            step_once()
        start = time.perf_counter()
        for _ in range(args.steps):
            step_once()
        elapsed = time.perf_counter() - start
    finally:
        target.close()
    total = args.steps * steps_per_call
    print(
        f"{args.env_id} [{label}]: {total:,} env steps "
        f"in {elapsed:.2f}s = {total / elapsed:,.0f} steps/s"
    )
    return 0


def _int_arg(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None


def _positive_int(text: str) -> int:
    value = _int_arg(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = _int_arg(text)
    if value < 0:  # numpy seeds must be >= 0
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sekai", description="sekai — RL environment framework command-line tools."
    )
    parser.add_argument("--version", action="version", version=f"sekai {sekai.__version__}")
    parser.add_argument(
        "-m",
        "--import",
        dest="modules",
        action="append",
        default=[],
        metavar="MODULE",
        help="import MODULE first so its environments are registered (repeatable)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list registered environments")
    p.add_argument("--namespace", help="only show this namespace ('' for none)")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("info", help="show an environment's spec and spaces")
    p.add_argument("env_id")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("check", help="run sekai.check_env on an environment")
    p.add_argument("env_id")
    p.add_argument(
        "--steps", type=_positive_int, default=500, help="random steps to take (default 500)"
    )
    p.add_argument("--determinism", action="store_true", help="also check seeded determinism")
    p.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("bench", help="measure random-action steps per second")
    p.add_argument("env_id")
    p.add_argument(
        "--steps",
        type=_positive_int,
        default=10_000,
        help="step() calls to time (default 10000); vectorised runs step every sub-env per call",
    )
    p.add_argument("--num-envs", type=_positive_int, default=1, help="sub-envs for vectorised runs")
    p.add_argument(
        "--mode",
        choices=["auto", "native", "sync", "async", "proc"],
        default="auto",
        help="vectorisation backend (default: native when the env has one, else sync)",
    )
    p.add_argument(
        "--workers", type=_positive_int, default=None, help="worker processes for --mode proc"
    )
    p.add_argument("--seed", type=_non_negative_int, default=0)
    p.set_defaults(func=_cmd_bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _import_modules(args.modules)
        result: int = args.func(args)
        return result
    except sekai.SekaiError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"error: could not import {e.name or 'module'}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
