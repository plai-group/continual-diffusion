"""--keypress_loss_weight must win over the km_fsq/km_ctx14 default under any argv spelling
(bare-token substring checks on sys.argv miss --flag=value; plaicraft-debug#80, #96)."""
from scripts.video_train_vdt import (
    create_argparser, resolve_keypress_loss_weight, resolve_km_tokenizer_checkpoint,
    _KM_FSQ_KEYPRESS_LOSS_WEIGHT, _KM_CTX14_KEYPRESS_LOSS_WEIGHT,
)


def _resolved(argv):
    args = create_argparser().parse_args(argv)
    resolve_keypress_loss_weight(args)
    return args.keypress_loss_weight


def test_omitted_km_fsq_derives_default():
    assert _resolved(["--action_encoding", "km_fsq"]) == _KM_FSQ_KEYPRESS_LOSS_WEIGHT


def test_omitted_km_ctx14_derives_default():
    assert _resolved(["--action_encoding", "km_ctx14"]) == _KM_CTX14_KEYPRESS_LOSS_WEIGHT


def test_omitted_raw_stays_one():
    assert _resolved(["--action_encoding", "raw"]) == 1.0


def test_space_form_wins_over_km_fsq_default():
    assert _resolved(["--action_encoding", "km_fsq", "--keypress_loss_weight", "0.5"]) == 0.5


def test_equals_form_wins_over_km_fsq_default():
    assert _resolved(["--action_encoding", "km_fsq", "--keypress_loss_weight=0.5"]) == 0.5


def test_empty_km_tokenizer_checkpoint_resolves_per_encoding():
    args_fsq = create_argparser().parse_args(["--action_encoding", "km_fsq"])
    args_ctx14 = create_argparser().parse_args(["--action_encoding", "km_ctx14"])
    assert resolve_km_tokenizer_checkpoint(args_fsq).endswith("pytorch_model.bin")
    assert resolve_km_tokenizer_checkpoint(args_ctx14).endswith("km_ctx14_10hz")


def test_explicit_km_tokenizer_checkpoint_wins():
    args = create_argparser().parse_args(["--action_encoding", "km_ctx14", "--km_tokenizer_checkpoint", "/tmp/custom"])
    assert resolve_km_tokenizer_checkpoint(args) == "/tmp/custom"
