"""Action metrics report what an all-zeros predictor would have scored.

Keys in this corpus are pressed 2-33% of the time, so doing nothing already
scores ~0.93 key_acc and the trivial mouse_mse is ~3.4. Without the *_trivial
series alongside them, a dead action head and a learning one are
indistinguishable on the dashboard.
"""
import re
from pathlib import Path

import torch

from improved_diffusion.debug_validation import _action_metrics

T, n_obs = 6, 3
NEXT, ROLL = slice(n_obs, n_obs + 1), slice(n_obs, None)

# Deliberately unequal press rates: a bug that averaged over frames before
# keys, or keys before frames, would still pass if every rate were the same.
RATES = [0.0, 1 / 3, 1 / 3, 2 / 3, 1.0, 0.0, 1 / 3, 2 / 3]
MOUSE = torch.tensor([[1.0, -3.0]] * T)


def _gt(press_rates=RATES, mouse=MOUSE):
    """(T, 10) GT where key i is held for round(rate*T) leading frames."""
    a = torch.zeros(T, 10)
    for i, r in enumerate(press_rates):
        a[: round(r * T), i] = 1.0
    a[:, 8:10] = mouse
    return a


def test_key_acc_trivial_is_the_all_zero_hit_rate():
    g = _gt()
    # The predictor is irrelevant to the trivial series by construction.
    m = _action_metrics(torch.rand(T, 10), g, ROLL)
    held = [max(0, round(r * T) - n_obs) for r in RATES]
    expected = sum(1 - h / (T - n_obs) for h in held) / len(RATES)
    assert abs(m["key_acc_trivial"] - expected) < 1e-7


def test_mouse_trivial_is_mean_abs_and_mean_square_of_gt():
    g = _gt()
    m = _action_metrics(torch.rand(T, 10), g, ROLL)
    gm = g[ROLL, 8:10]
    assert abs(m["mouse_l1_trivial"] - gm.abs().mean().item()) < 1e-7
    assert abs(m["mouse_mse_trivial"] - (gm ** 2).mean().item()) < 1e-7


def test_all_zero_predictor_scores_exactly_the_trivial_series():
    """The property the baseline exists for: it is what a dead head achieves."""
    g = _gt()
    for sl in (NEXT, ROLL):
        m = _action_metrics(torch.zeros(T, 10), g, sl)
        assert abs(m["key_acc"] - m["key_acc_trivial"]) < 1e-7
        assert abs(m["mouse_l1"] - m["mouse_l1_trivial"]) < 1e-7
        assert abs(m["mouse_mse"] - m["mouse_mse_trivial"]) < 1e-7


def test_perfect_predictor_leaves_the_trivial_series_alone():
    """The two series are independent -- the baseline tracks GT, not the head."""
    g = _gt()
    m = _action_metrics(g.clone(), g, ROLL)
    assert m["key_acc"] == 1.0
    assert m["mouse_mse"] == 0.0
    assert m["key_acc_trivial"] < 1.0
    assert m["mouse_mse_trivial"] > 0.0


def test_next_and_roll_windows_differ():
    # A bug that ignored `sl` would make the two scopes identical. Key 4 is
    # held on every frame, keys 1/2/6 only on the leading third.
    g = _gt()
    z = torch.zeros(T, 10)
    assert (_action_metrics(z, g, NEXT)["key_acc_trivial"]
            != _action_metrics(z, g, ROLL)["key_acc_trivial"])


def test_trivial_keys_are_aggregated():
    # The metrics are useless if run_debug_validation drops them before wandb.
    src = Path("improved_diffusion/debug_validation.py").read_text()
    keys = re.search(r"ACT_METRIC_KEYS = \((.*?)\)", src, re.S).group(1)
    for k in ("key_acc_trivial", "mouse_l1_trivial", "mouse_mse_trivial"):
        assert k in keys
