"""plaicraft-debug#98: per-loss gradient norms on the shared VDT blocks."""
import math
from types import SimpleNamespace

import torch

from improved_diffusion import train_util
from improved_diffusion.train_util import TrainLoop
from test_action_loss_weighting import _build, _losses


def _probe(w, monkeypatch):
    logged = {}
    monkeypatch.setattr(train_util.logger, "logkv_mean", lambda k, v: logged.__setitem__(k, v))
    torch.manual_seed(0)
    model, diffusion = _build(action_dim=10, keypress_loss_weight=w)
    with torch.no_grad():  # DiT zero-inits the output layers, so every block grad is 0 at init
        for p in model.parameters():
            p.normal_(0, 0.02)
    torch.manual_seed(1)
    terms = _losses(model, diffusion, action_dim=10, keypress_loss_weight=w)
    TrainLoop._log_grad_probe(SimpleNamespace(model=model), terms["loss_video"].mean(), terms["loss_action_weighted"].mean())
    return terms, logged, model


def test_weighted_action_term_is_what_total_loss_adds(monkeypatch):
    terms, _, _ = _probe(2.0, monkeypatch)
    assert torch.allclose(terms["loss_action_weighted"], 2.0 * terms["action_dim_ratio"] * terms["loss_action"])
    assert torch.allclose(terms["loss"], terms["loss_video"] + terms["loss_action_weighted"], atol=1e-6)


def test_probe_logs_finite_norms_and_scales_with_weight(monkeypatch):
    _, one, _ = _probe(1.0, monkeypatch)
    _, two, _ = _probe(2.0, monkeypatch)
    assert all(math.isfinite(v) for v in one.values())
    assert one["grad_norm/video_blocks"] > 0 and one["grad_norm/action_blocks"] > 0
    assert math.isclose(two["grad_norm/action_blocks"], 2 * one["grad_norm/action_blocks"], rel_tol=1e-4)
    assert math.isclose(two["grad_norm/video_blocks"], one["grad_norm/video_blocks"], rel_tol=1e-5)


def test_zero_action_weight_gives_zero_action_norm(monkeypatch):
    _, logged, _ = _probe(0.0, monkeypatch)
    assert logged["grad_norm/action_blocks"] == 0.0


def test_probe_leaves_param_grads_untouched(monkeypatch):
    _, _, model = _probe(1.0, monkeypatch)
    assert all(p.grad is None for p in model.parameters())
