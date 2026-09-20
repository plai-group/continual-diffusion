"""VDT's class label embedder carrying a player id (issue #85).

The embedder has been present but inert since the DiT port: num_classes=0 makes it a single
learned constant and forward() hardcoded y = zeros. These tests pin both the new behaviour
and -- importantly -- that the default path is byte-identical to what every existing run
script trains today.
"""
import pytest
import torch as th

from improved_diffusion import train_util
from improved_diffusion.vdt import VDT


def _model(num_classes=0, class_dropout_prob=0.1, **kw):
    """VDT directly rather than VDT_SM_2: the factories pin depth/hidden_size, and a 1-layer
    model is all these assertions need.

    The zero-init is undone deliberately. DiT-style init zeroes final_layer and every adaLN
    modulation, so a freshly built model returns exactly zeros for ANY input -- which would
    make every assertion below pass without testing a thing. Randomising those parameters
    turns the model back into a real function of its conditioning."""
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


def _x(b=2, t=4):
    return th.randn(b, t, 3, 24, 40), th.zeros(b, dtype=th.long)


# ── the default path must not move ─────────────────────────────────────────────────────────

def test_num_classes_zero_keeps_the_single_constant_embedding():
    m = _model(num_classes=0)
    assert m.y_embedder.embedding_table.weight.shape[0] == 1, (
        "at num_classes=0 the table is one learned constant; a second row would mean the "
        "null-class path silently switched on"
    )


def test_y_none_matches_explicit_zeros_at_num_classes_zero():
    """Every existing run script passes no y at all. That path has to stay exactly what it
    was before this change."""
    th.manual_seed(0)
    m = _model(num_classes=0)
    x, t = _x()
    with th.no_grad():
        a, _ = m(x, timesteps=t)
        b, _ = m(x, timesteps=t, y=th.zeros(2, dtype=th.long))
    assert a.count_nonzero() > 0, "degenerate model -- the equality below would be vacuous"
    assert th.equal(a, b)


# ── the player path ────────────────────────────────────────────────────────────────────────

def test_two_classes_add_a_null_row_for_cfg():
    m = _model(num_classes=2)
    assert m.y_embedder.embedding_table.weight.shape[0] == 3, "2 players + the null class"


def test_distinct_players_give_distinct_output():
    th.manual_seed(0)
    m = _model(num_classes=2)
    x, t = _x()
    with th.no_grad():
        p0, _ = m(x, timesteps=t, y=th.zeros(2, dtype=th.long))
        p1, _ = m(x, timesteps=t, y=th.ones(2, dtype=th.long))
    assert not th.allclose(p0, p1), "the player label must reach the output"


def test_force_label_drop_true_equals_passing_the_null_index():
    th.manual_seed(0)
    m = _model(num_classes=2)
    x, t = _x()
    with th.no_grad():
        dropped, _ = m(x, timesteps=t, y=th.ones(2, dtype=th.long), force_label_drop=True)
        null, _ = m(x, timesteps=t, y=th.full((2,), 2, dtype=th.long))
    assert th.equal(dropped, null), "the CFG unconditional pass must hit the null row"


def test_force_label_drop_false_leaves_the_label_alone():
    th.manual_seed(0)
    m = _model(num_classes=2)
    x, t = _x()
    with th.no_grad():
        forced, _ = m(x, timesteps=t, y=th.ones(2, dtype=th.long), force_label_drop=False)
        plain, _ = m(x, timesteps=t, y=th.ones(2, dtype=th.long))
    assert th.equal(forced, plain)


@pytest.mark.parametrize("shape", [(2,), (2, 1)])
def test_y_accepts_the_shapes_a_dataloader_produces(shape):
    """A collated per-sequence scalar arrives as (B,) or (B, 1) depending on how the dataset
    returns it; both have to mean the same thing."""
    th.manual_seed(0)
    m = _model(num_classes=2)
    x, t = _x()
    with th.no_grad():
        out, _ = m(x, timesteps=t, y=th.ones(*shape, dtype=th.long))
    assert out.shape == (2, 4, 3, 24, 40)


def test_label_dropout_is_off_when_prob_is_zero():
    th.manual_seed(0)
    m = _model(num_classes=2, class_dropout_prob=0.0)
    assert m.y_embedder.embedding_table.weight.shape[0] == 2, "no dropout means no null row"


# ── the factory default ────────────────────────────────────────────────────────────────────

def test_factory_default_is_still_zero_classes():
    from improved_diffusion.script_util import create_vdt_model
    m = create_vdt_model("VDT-SM", input_size=(24, 40), patch_size=4, in_channels=3,
                         num_frames=4, learn_sigma=False)
    assert m.y_embedder.num_classes == 0


# ── cond_combine: how t and y become c (issue #85 follow-up) ───────────────────────────────

def test_add_is_the_default_and_adds_no_parameters():
    """Every checkpoint trained so far has no cond_proj in its state_dict; the default path
    must keep loading them."""
    m = _model(num_classes=2)
    assert m.cond_combine == "add"
    assert m.cond_proj is None
    assert not [k for k in m.state_dict() if "cond_proj" in k or "cond_norm" in k]


@pytest.mark.parametrize("mode", ["concat", "concat_ln"])
def test_concat_projects_back_to_hidden_size(mode):
    """One projection, not 14 widened adaLN layers: everything downstream of c keeps its shape."""
    m = _model(num_classes=2, cond_combine=mode)
    assert [l.weight.shape for l in m.cond_proj if isinstance(l, th.nn.Linear)] == [(128, 128), (64, 128)]
    assert any(isinstance(l, th.nn.SiLU) for l in m.cond_proj), (
        "without the nonlinearity this is only a reparametrisation of c = t + y"
    )
    for block in m.blocks:
        assert block.adaLN_modulation[-1].weight.shape == (6 * 64, 64)
    assert m.final_layer.adaLN_modulation[-1].weight.shape == (2 * 64, 64)


@pytest.mark.parametrize("mode", ["concat", "concat_ln"])
def test_distinct_players_give_distinct_output_under_concat(mode):
    th.manual_seed(0)
    m = _model(num_classes=2, cond_combine=mode)
    x, t = _x()
    with th.no_grad():
        p0, _ = m(x, timesteps=t, y=th.zeros(2, dtype=th.long))
        p1, _ = m(x, timesteps=t, y=th.ones(2, dtype=th.long))
    assert p0.shape == (2, 4, 3, 24, 40)
    assert p0.count_nonzero() > 0, "degenerate model -- the inequality below would be vacuous"
    assert not th.allclose(p0, p1)


def test_concat_ln_normalises_both_halves_before_projecting():
    """The point of concat_ln: y arrives at cond_proj at the same scale as t, instead of the
    80x gap measured on run hu9hvxqv (|t| 117 vs |y| 1.2)."""
    th.manual_seed(0)
    m = _model(num_classes=2, cond_combine="concat_ln")
    seen = []
    m.cond_proj.register_forward_pre_hook(lambda _mod, args: seen.append(args[0]))
    x, t = _x()
    with th.no_grad():
        m(x, timesteps=t, y=th.ones(2, dtype=th.long))
    t_part, y_part = seen[0][:, :64], seen[0][:, 64:]
    for part in (t_part, y_part):
        assert abs(part.pow(2).mean().sqrt().item() - 1.0) < 0.15


@pytest.mark.parametrize("mode", ["concat", "concat_ln"])
def test_concat_refuses_the_per_frame_action_embedder(mode):
    """The action_embedder branch builds a (B, T, D) c by addition; concatenating there is a
    separate design question and no current config reaches it."""
    th.manual_seed(0)
    m = _model(num_classes=2, cond_combine=mode, action_dim=8)
    assert m.action_embedder is not None, "wrong branch -- the assert below would never fire"
    x, t = _x()
    with pytest.raises(AssertionError, match="cond_combine"):
        m(x, timesteps=t, y=th.ones(2, dtype=th.long), actions=th.randn(2, 4, 8))


# ── loading checkpoints trained before the flag existed ────────────────────────────────────

def test_backfill_defaults_missing_cond_combine_to_add():
    """args_to_dict getattrs every default key, so a pre-flag checkpoint's config would raise
    AttributeError before the model is ever built."""
    import argparse

    from improved_diffusion.script_util import backfill_cond_combine

    ns = argparse.Namespace(num_classes=2)
    backfill_cond_combine(ns)
    assert ns.cond_combine == "add"


def test_backfill_leaves_an_explicit_cond_combine_alone():
    import argparse

    from improved_diffusion.script_util import backfill_cond_combine

    ns = argparse.Namespace(cond_combine="concat_ln")
    backfill_cond_combine(ns)
    assert ns.cond_combine == "concat_ln"


# ── frozen orthogonal label embedding (issue #85, third pass) ─────────────────────────────

def test_frozen_rows_are_orthogonal_with_norm_1_5():
    m = _model(num_classes=2, label_embedding_frozen=True)
    W = m.y_embedder.embedding_table.weight
    assert W.shape[0] == 3, "2 players + the null class"
    assert th.allclose(W.norm(dim=-1), th.full((3,), 1.5), atol=1e-4)
    gram = W @ W.T
    off_diag = gram[~th.eye(3, dtype=th.bool)]
    assert th.allclose(off_diag, th.zeros_like(off_diag), atol=1e-4)


def test_frozen_at_zero_dropout_has_exactly_num_classes_rows():
    m = _model(num_classes=2, class_dropout_prob=0.0, label_embedding_frozen=True)
    assert m.y_embedder.embedding_table.weight.shape[0] == 2


def test_frozen_table_requires_no_grad():
    m = _model(num_classes=2, label_embedding_frozen=True)
    assert not m.y_embedder.embedding_table.weight.requires_grad


def test_frozen_table_survives_a_training_step():
    th.manual_seed(0)
    m = _model(num_classes=2, label_embedding_frozen=True)
    before = m.y_embedder.embedding_table.weight.clone()
    other_before = next(p for p in m.blocks[0].parameters() if p.requires_grad).clone()
    x, t = _x()
    out, _ = m(x, timesteps=t, y=th.ones(2, dtype=th.long))
    out.sum().backward()
    th.optim.SGD(m.parameters(), lr=0.1).step()
    assert th.equal(m.y_embedder.embedding_table.weight, before)
    other_after = next(p for p in m.blocks[0].parameters() if p.requires_grad)
    assert not th.equal(other_after, other_before), "degenerate step -- the equality above would be vacuous"


def test_label_embedding_frozen_false_is_byte_identical_to_today():
    th.manual_seed(0)
    m_off = _model(num_classes=2)
    th.manual_seed(0)
    m_explicit = _model(num_classes=2, label_embedding_frozen=False)
    assert th.equal(m_off.y_embedder.embedding_table.weight, m_explicit.y_embedder.embedding_table.weight)


def test_backfill_defaults_missing_label_embedding_frozen_to_false():
    import argparse

    from improved_diffusion.script_util import backfill_label_embedding_frozen

    ns = argparse.Namespace(num_classes=2)
    backfill_label_embedding_frozen(ns)
    assert ns.label_embedding_frozen is False


def test_backfill_leaves_an_explicit_label_embedding_frozen_alone():
    import argparse

    from improved_diffusion.script_util import backfill_label_embedding_frozen

    ns = argparse.Namespace(label_embedding_frozen=True)
    backfill_label_embedding_frozen(ns)
    assert ns.label_embedding_frozen is True


def test_label_init_std_defaults_to_dits_small_value():
    m = _model(num_classes=2)
    assert m.label_init_std == 0.02
    assert m.y_embedder.embedding_table.weight.std().item() < 0.1


def test_label_init_std_one_restores_nn_embeddings_own_default():
    """std=1.0 is what nn.Embedding initialises to on its own; at hidden_size=640 that starts
    ||y|| near 25 instead of 0.51, so the label is not negligible beside ||t|| from step 0."""
    th.manual_seed(0)
    m = VDT(
        input_size=(24, 40), patch_size=4, in_channels=3, num_frames=4, learn_sigma=False,
        depth=1, hidden_size=640, num_heads=4, num_classes=2, class_dropout_prob=0.0,
        label_init_std=1.0,
    )
    rows = m.y_embedder.embedding_table.weight
    assert 0.8 < rows.std().item() < 1.2
    assert 20.0 < rows.norm(dim=-1).mean().item() < 30.0


def test_freezing_wins_over_label_init_std():
    """Both set is a config trap: the frozen rows must survive, not be re-drawn at std=1.0."""
    from improved_diffusion.vdt import FROZEN_LABEL_SCALE
    m = VDT(
        input_size=(24, 40), patch_size=4, in_channels=3, num_frames=4, learn_sigma=False,
        depth=1, hidden_size=64, num_heads=4, num_classes=2, class_dropout_prob=0.0,
        label_embedding_frozen=True, label_init_std=1.0,
    )
    rows = m.y_embedder.embedding_table.weight
    assert rows.norm(dim=-1).allclose(th.full((2,), FROZEN_LABEL_SCALE), atol=1e-4)
    assert rows.requires_grad is False


def test_backfill_label_init_std():
    import argparse

    from improved_diffusion.script_util import backfill_label_init_std
    ns = argparse.Namespace()
    backfill_label_init_std(ns)
    assert ns.label_init_std == 0.02
    ns2 = argparse.Namespace(label_init_std=1.0)
    backfill_label_init_std(ns2)
    assert ns2.label_init_std == 1.0


# ── what the label is actually being taught (issue #85, fourth pass) ────────────────────────

def _logged(monkeypatch):
    """Capture logkv_mean, the only place the label's gradient becomes visible."""
    seen = {}
    monkeypatch.setattr(train_util.logger, "logkv_mean", lambda k, v: seen.__setitem__(k, v))
    return seen


def _with_grad(m, g0, g1):
    w = m.y_embedder.embedding_table.weight
    w.grad = th.stack([g0, g1])
    return w


def test_label_grad_keys_populate_after_a_real_backward(monkeypatch):
    seen = _logged(monkeypatch)
    m = _model(num_classes=2, class_dropout_prob=0.0)
    x, _ = _x()
    m(x, th.zeros(2), y=th.tensor([0, 1]))[0].sum().backward()
    assert train_util.log_label_grad(m, 1.0) is not None
    for key in ("grad/label/norm", "grad/label/frac", "grad/label/row0", "grad/label/row1",
                "grad/label/cos_rows", "param/label/norm", "param/label/cos_rows"):
        assert key in seen, key
    assert seen["grad/label/norm"] > 0


def test_cos_rows_reports_whether_the_rows_are_driven_together(monkeypatch):
    """The whole point of the metric: co-driven rows translate instead of separating."""
    m = _model(num_classes=2, class_dropout_prob=0.0)
    shared = th.randn(m.y_embedder.embedding_table.weight.shape[1])
    for g1, expected in ((shared, 1.0), (-shared, -1.0)):
        seen = _logged(monkeypatch)
        _with_grad(m, shared, g1)
        train_util.log_label_grad(m, 1.0)
        assert seen["grad/label/cos_rows"] == pytest.approx(expected, abs=1e-5)


def test_frac_is_the_label_share_of_the_total_gradient(monkeypatch):
    seen = _logged(monkeypatch)
    m = _model(num_classes=2, class_dropout_prob=0.0)
    w = _with_grad(m, th.ones(m.y_embedder.embedding_table.weight.shape[1]),
                   th.zeros(m.y_embedder.embedding_table.weight.shape[1]))
    train_util.log_label_grad(m, 4.0)
    assert seen["grad/label/frac"] == pytest.approx(w.grad.norm().item() / 4.0)
    assert seen["grad/label/row1"] == 0.0


def test_delta_needs_a_previous_table_and_then_tracks_the_realised_update(monkeypatch):
    m = _model(num_classes=2, class_dropout_prob=0.0)
    seen = _logged(monkeypatch)
    prev = train_util.log_label_grad(m, 1.0)
    assert "param/label/delta" not in seen, "nothing to difference against on the first step"
    with th.no_grad():
        m.y_embedder.embedding_table.weight.add_(0.1)
    seen = _logged(monkeypatch)
    train_util.log_label_grad(m, 1.0, prev)
    assert seen["param/label/delta"] > 0
    assert seen["param/label/rel_delta"] == pytest.approx(
        seen["param/label/delta"] / seen["param/label/norm"])


def test_a_single_row_table_logs_nothing(monkeypatch):
    """num_classes=0 keeps one inert constant; there is no pair to compare and no run to break."""
    seen = _logged(monkeypatch)
    assert train_util.log_label_grad(_model(num_classes=0), 1.0) is None
    assert seen == {}


# ── a frozen table must still report its gradient (issue #85, fifth pass) ───────────────────

def test_a_frozen_label_still_logs_its_gradient(monkeypatch):
    """The frozen arms are the ones whose rows we most want to compare, so grad/label/* cannot
    go blank just because requires_grad is False."""
    seen = _logged(monkeypatch)
    m = _model(num_classes=2, class_dropout_prob=0.0, label_embedding_frozen=True)
    x, _ = _x()
    m(x, th.zeros(2), y=th.tensor([0, 1]))[0].sum().backward()
    train_util.log_label_grad(m, 1.0)
    for key in ("grad/label/norm", "grad/label/frac", "grad/label/row0", "grad/label/row1"):
        assert key in seen, key
    assert seen["grad/label/norm"] > 0


def test_the_probe_does_not_unfreeze_the_table():
    m = _model(num_classes=2, class_dropout_prob=0.0, label_embedding_frozen=True)
    w = m.y_embedder.embedding_table.weight
    before = w.detach().clone()
    x, _ = _x()
    m(x, th.zeros(2), y=th.tensor([0, 1]))[0].sum().backward()
    assert w.grad is None and not w.requires_grad
    assert th.equal(w.detach(), before)


def test_frozen_row_gradients_are_row_sparse(monkeypatch):
    """Both batch items carry y=0, so row 1 must receive exactly nothing."""
    seen = _logged(monkeypatch)
    m = _model(num_classes=2, class_dropout_prob=0.0, label_embedding_frozen=True)
    x, _ = _x()
    m(x, th.zeros(2), y=th.tensor([0, 0]))[0].sum().backward()
    train_util.log_label_grad(m, 1.0)
    assert seen["grad/label/row0"] > 0
    assert seen["grad/label/row1"] == 0.0


def test_label_frozen_scale_sets_the_row_norm():
    """The magnitude sweep rides on this: under cond_combine=add the row norm IS |y|/|t|."""
    for scale in (1.5, 25.0, 100.0):
        w = _model(num_classes=2, class_dropout_prob=0.0,
                   label_embedding_frozen=True, label_frozen_scale=scale
                   ).y_embedder.embedding_table.weight
        assert th.allclose(w[0].norm(), th.tensor(scale), rtol=1e-4)
        assert th.allclose(w[1].norm(), th.tensor(scale), rtol=1e-4)
        assert abs(th.cosine_similarity(w[0], w[1], dim=0).item()) < 1e-5


def test_label_frozen_scale_is_inert_when_not_frozen():
    a = _model(num_classes=2, class_dropout_prob=0.0, label_init_std=1.0, label_frozen_scale=1.5)
    b = _model(num_classes=2, class_dropout_prob=0.0, label_init_std=1.0, label_frozen_scale=99.0)
    assert th.equal(a.y_embedder.embedding_table.weight, b.y_embedder.embedding_table.weight)
