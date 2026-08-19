"""Regression test for the wandb step-logging fix (plai-group/plaicraft-debug#69).

Without an explicit step= kwarg, wandb's internal _step counter free-runs
from 0 on every process start. After a resume, new log calls silently
overwrite the run's earliest history rows instead of appending new ones, so
training progress past the first resume never becomes visible on the
dashboard even though checkpoints and the live summary are unaffected.
"""
from unittest.mock import patch

from improved_diffusion.logger import Logger


def test_dumpkvs_passes_the_real_training_step_to_wandb():
    log = Logger()
    log.logkv("step", 536590)
    log.logkv("mse", 0.01)

    with patch("improved_diffusion.logger.wandb.log") as mock_log:
        log.dumpkvs()

    args, kwargs = mock_log.call_args
    assert kwargs.get("step") == 536590
    assert args[0]["step"] == 536590


def test_dumpkvs_falls_back_to_none_step_when_step_was_never_logged():
    log = Logger()
    log.logkv("mse", 0.01)

    with patch("improved_diffusion.logger.wandb.log") as mock_log:
        log.dumpkvs()

    _, kwargs = mock_log.call_args
    assert kwargs.get("step") is None


def test_dumpkvs_still_clears_state_after_logging():
    log = Logger()
    log.logkv("step", 10)
    log.logkv("mse", 0.5)

    with patch("improved_diffusion.logger.wandb.log"):
        log.dumpkvs()

    assert log.name2val == {}
    assert log.name2cnt == {}
