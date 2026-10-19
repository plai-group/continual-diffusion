"""Label-swap delta -- the live inertness gate (issue #85, third pass): would have caught the
label-inertness failure at 20k instead of 410k."""
import pytest
import torch as th

from improved_diffusion.debug_validation import _label_swap_probe, run_label_swap_probe
from improved_diffusion.vdt import VDT


def _model(num_classes=0, class_dropout_prob=0.1, **kw):
    """Mirrors tests/test_player_label_conditioning.py's _model: randomise the DiT-style
    zero-init so the model is a real function of its conditioning."""
    th.manual_seed(1234)
    m = VDT(
        input_size=(24, 40), patch_size=4, in_channels=3, num_frames=4, learn_sigma=False,
        depth=1, hidden_size=64, num_heads=4, num_classes=num_classes,
        class_dropout_prob=class_dropout_prob, **kw,
    ).eval()
    with th.no_grad():
        for param in m.parameters():
            if param.count_nonzero() == 0:
                param.normal_(0.0, 0.05)
    return m


def _x(b=1, t=4):
    return th.randn(b, t, 3, 24, 40), th.zeros(b, dtype=th.long)


def test_a_label_that_changes_the_output_reports_a_non_trivial_delta():
    th.manual_seed(0)
    m = _model(num_classes=2)
    x, t = _x()
    delta, _, _ = _label_swap_probe(m, x, t, {})
    assert delta > 1e-3, "well above the ~1e-4 inert reading the gate is watching for"


def test_an_inert_model_reports_zero_delta():
    """num_classes=0 -> the table has one row (the null class); y1 clamps to y0 instead of
    an out-of-range lookup, so both forward passes are literally the same call."""
    th.manual_seed(0)
    m = _model(num_classes=0)
    x, t = _x()
    delta, cos, dist = _label_swap_probe(m, x, t, {})
    assert delta == 0.0
    assert cos == pytest.approx(1.0)
    assert dist == pytest.approx(0.0)


def test_label_table_metrics_match_a_hand_computed_cosine_and_distance():
    m = _model(num_classes=2)
    x, t = _x()
    _, cos, dist = _label_swap_probe(m, x, t, {})
    W = m.y_embedder.embedding_table.weight
    assert cos == pytest.approx(th.nn.functional.cosine_similarity(W[0:1], W[1:2]).item())
    assert dist == pytest.approx(th.linalg.norm(W[0] - W[1]).item())


def _diffusion():
    from improved_diffusion.script_util import create_gaussian_diffusion
    return create_gaussian_diffusion(steps=1000)


def _probe_inputs(b=1, t=4):
    obs_mask = th.zeros(b, t, 1, 1, 1)
    obs_mask[:, : t // 2] = 1.0
    return th.randn(b, t, 3, 24, 40), th.zeros(b, t, 8), th.zeros(b, t, 2), obs_mask


def test_the_probe_reads_the_noisy_end_of_the_schedule():
    """Regression: edm_sigmas rises with t, so probing t=0 measures a near-identity denoiser
    and reports a label-using model as inert."""
    from improved_diffusion.debug_validation import LABEL_SWAP_PROBE_FRACS
    assert min(LABEL_SWAP_PROBE_FRACS) >= 0.5


def test_a_label_using_model_clears_the_gate_at_the_probed_timesteps():
    th.manual_seed(0)
    m, d = _model(num_classes=2), _diffusion()
    x0, act, mouse, obs_mask = _probe_inputs()
    out = run_label_swap_probe(m, d, x0, act, mouse, obs_mask)
    assert out["val/player/swap_delta"] > 1e-3
    assert set(out) == {"val/player/swap_delta", "val/label/cos_y0_y1", "val/label/dist_y0_y1"}


def test_an_inert_model_clears_nothing():
    th.manual_seed(0)
    m, d = _model(num_classes=0), _diffusion()
    x0, act, mouse, obs_mask = _probe_inputs()
    assert run_label_swap_probe(m, d, x0, act, mouse, obs_mask)["val/player/swap_delta"] == 0.0


def test_the_probe_is_deterministic_across_calls():
    """The seed is fixed so the scalar is a trend line, not noise."""
    th.manual_seed(0)
    m, d = _model(num_classes=2), _diffusion()
    args = _probe_inputs()
    first = run_label_swap_probe(m, d, *args)
    assert run_label_swap_probe(m, d, *args) == first
