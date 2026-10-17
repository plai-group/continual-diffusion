"""CE must reach wandb, and must be scored against the real multi-hot.

Two failure modes this pins down, both silent:

  * A metric _action_metrics computes but ACT_METRIC_KEYS does not list never
    reaches wandb. Aggregation iterates the allowlist, not per_row, so the key
    is simply absent -- no error, no warning, just a panel that never appears.
  * Under km_fsq the existing g_key is a thresholded decoder round-trip, not
    truth. Scoring CE against it would measure the tokenizer's reconstruction,
    not the model's. CE takes keypress_raw_chunk instead.
"""
import inspect

import torch

from improved_diffusion.debug_validation import _action_metrics, run_debug_validation

T, n_obs = 20, 10
SL = slice(n_obs + 1, None)


def _multi_hot(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(T, 8, generator=g) > 0.7).float()


def test_action_metrics_emits_ce_when_probabilities_are_given():
    y = _multi_hot()
    out = _action_metrics(None, None, None, None, SL, p_key_prob=y.clone(), g_key_true=y)
    assert "key_cross_entropy" in out and "key_ce_baserate" in out
    assert out["key_cross_entropy"] < 1e-3  # a perfect probability


def test_action_metrics_omits_ce_without_a_true_multi_hot():
    y = _multi_hot()
    out = _action_metrics(None, None, None, None, SL, p_key_prob=y.clone(), g_key_true=None)
    assert "key_cross_entropy" not in out and "key_ce_baserate" not in out


def test_ce_beats_its_baserate_for_a_good_head_and_loses_for_a_dead_one():
    y = _multi_hot(1)
    good = _action_metrics(None, None, None, None, SL, p_key_prob=y.clone(), g_key_true=y)
    dead = _action_metrics(None, None, None, None, SL, p_key_prob=1 - y, g_key_true=y)
    assert good["key_cross_entropy"] < good["key_ce_baserate"]
    assert dead["key_cross_entropy"] > dead["key_ce_baserate"]


def test_ce_keys_are_in_the_wandb_allowlist():
    # ACT_METRIC_KEYS is function-local, so read it out of the source rather than import it.
    src = inspect.getsource(run_debug_validation)
    allowlist = src.split("ACT_METRIC_KEYS = (")[1].split(")")[0]
    assert "key_cross_entropy" in allowlist
    assert "key_ce_baserate" in allowlist
    assert "key_jaccard_distance" in allowlist
