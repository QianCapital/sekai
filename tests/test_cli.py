from __future__ import annotations

import subprocess
import sys

import pytest

from sekai import __version__
from sekai.cli import main


def test_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "CartPole-v1" in out and "Portfolio-v0" in out


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "CartPole-v1"]) == 0
    out = capsys.readouterr().out
    assert "observation_space=Box" in out and "TimeLimit" in out


def test_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "Trading-v0", "--steps", "50", "--determinism"]) == 0
    assert "all checks passed" in capsys.readouterr().out


def test_bench(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "CartPole-v1", "--steps", "200"]) == 0
    assert main(["bench", "CartPole-v1", "--steps", "20", "--num-envs", "2"]) == 0
    assert "steps/s" in capsys.readouterr().out


def test_bench_user_errors_exit_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    # make_vec's ValueError is reported as an error message, not a traceback
    assert main(["bench", "CartPole-v1", "--steps", "10", "--mode", "native"]) == 2
    assert "error: CartPole-v1 has no native vector implementation" in capsys.readouterr().err
    # --workers is checked before choosing between the single-env and vectorised paths
    assert main(["bench", "CartPole-v1", "--steps", "10", "--workers", "2"]) == 2
    assert "--workers applies to --mode proc only" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["bench", "CartPole-v1", "--steps", "0"], "must be a positive integer"),
        (["bench", "CartPole-v1", "--num-envs", "0"], "must be a positive integer"),
        (["bench", "CartPole-v1", "--mode", "proc", "--workers", "-1"], "must be a positive"),
        (["check", "CartPole-v1", "--steps", "-5"], "must be a positive integer"),
        (["check", "CartPole-v1", "--steps", "ten"], "invalid int value"),
        (["bench", "CartPole-v1", "--steps", "10", "--seed", "-1"], "must be a non-negative"),
        (["bench", "Trading-v0", "--num-envs", "2", "--seed", "-3"], "must be a non-negative"),
    ],
)
def test_invalid_numbers_are_usage_errors(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_unknown_env_suggests(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "CartPol-v1"]) == 2
    assert "Did you mean" in capsys.readouterr().err


def test_import_plugin_module(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-m", "json", "list", "--namespace", "nope"]) == 0
    assert "No environments in namespace 'nope'" in capsys.readouterr().out
    assert main(["-m", "no_such_module_xyz", "list"]) == 2
    assert "could not import" in capsys.readouterr().err


def test_python_dash_m() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "sekai", "--version"], capture_output=True, text=True, check=True
    )
    assert __version__ in out.stdout
