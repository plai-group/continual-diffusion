"""Classifier-free guidance on the player label (issue #85).

_CFGWrapper already existed for action guidance, but that arm is inert under
generate_actions (the action is a sequence token, so action_embedder is None and
force_action_drop has nothing to drop). The label arm is the one that works in every mode,
so it gets its own scale rather than sharing cfg_scale.
"""
import torch as th
import torch.nn as nn

from improved_diffusion.debug_validation import _CFGWrapper


class _Recorder(nn.Module):
    """Returns a constant per (force_action_drop, force_label_drop) combination and records
    the kwargs it was called with, so guidance arithmetic is checkable exactly."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, timesteps=None, **kwargs):
        self.calls.append((kwargs.get("force_action_drop", "absent"),
                           kwargs.get("force_label_drop", "absent")))
        base = {(False, False): 1.0, (True, False): 2.0,
                (False, True): 4.0, (True, True): 8.0}
        key = (bool(kwargs.get("force_action_drop", False)),
               bool(kwargs.get("force_label_drop", False)))
        return th.full_like(x, base[key]), None


def _call(w, label_w):
    m = _Recorder()
    out, second = _CFGWrapper(m, w, label_w)(th.zeros(1, 2), timesteps=th.zeros(1))
    return m, out, second


def test_both_scales_one_means_a_single_untouched_pass():
    m, out, second = _call(1.0, 1.0)
    assert len(m.calls) == 1
    assert m.calls[0] == ("absent", "absent"), "no drop flag should be injected when off"
    assert out.unique().tolist() == [1.0]
    assert second is None


def test_label_guidance_runs_two_passes_and_extrapolates():
    m, out, _ = _call(1.0, 3.0)
    assert len(m.calls) == 2
    # cond = 1.0, null-label = 4.0  ->  4.0 + 3.0 * (1.0 - 4.0) = -5.0
    assert out.unique().tolist() == [-5.0]


def test_label_guidance_does_not_touch_the_action_flag():
    m, _, _ = _call(1.0, 3.0)
    assert all(c[0] == "absent" for c in m.calls), (
        "cfg_scale is off, so force_action_drop must not be injected -- under "
        "generate_actions it is meaningless, and injecting it changes nothing but noise"
    )


def test_action_guidance_alone_is_unchanged():
    m, out, _ = _call(2.0, 1.0)
    assert len(m.calls) == 2
    assert all(c[1] == "absent" for c in m.calls)
    # cond = 1.0, null-action = 2.0  ->  2.0 + 2.0 * (1.0 - 2.0) = 0.0
    assert out.unique().tolist() == [0.0]


def test_both_scales_compose_over_three_passes():
    m, out, _ = _call(2.0, 3.0)
    assert len(m.calls) == 3, "cond, null-action, null-label -- each arm from its own null"
    # cond=1, null-action=2 -> 2 + 2*(1-2) = 0; null-label=4 -> 4 + 3*(0-4) = -8
    assert out.unique().tolist() == [-8.0]


def test_wrapper_returns_the_two_tuple_the_sampler_expects():
    """heun_sample unpacks (eps, extras); a bare tensor would fail deep inside sampling."""
    _, out, second = _call(1.0, 2.0)
    assert isinstance(out, th.Tensor) and second is None
