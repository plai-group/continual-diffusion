"""VDT's class label embedder carrying a player id (issue #85).

The embedder has been present but inert since the DiT port: num_classes=0 makes it a single
learned constant and forward() hardcoded y = zeros. These tests pin both the new behaviour
and -- importantly -- that the default path is byte-identical to what every existing run
script trains today.
"""
import pytest
import torch as th

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
