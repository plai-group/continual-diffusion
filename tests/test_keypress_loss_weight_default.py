"""--keypress_loss_weight must win over the km_fsq default under any argv spelling
(bare-token substring checks on sys.argv miss --flag=value; plaicraft-debug#80)."""
from scripts.video_train_vdt import create_argparser, resolve_keypress_loss_weight, _KM_FSQ_KEYPRESS_LOSS_WEIGHT


def _resolved(argv):
    args = create_argparser().parse_args(argv)
    resolve_keypress_loss_weight(args)
    return args.keypress_loss_weight


def test_omitted_km_fsq_derives_default():
    assert _resolved(["--action_encoding", "km_fsq"]) == _KM_FSQ_KEYPRESS_LOSS_WEIGHT


def test_omitted_raw_stays_one():
    assert _resolved(["--action_encoding", "raw"]) == 1.0


def test_space_form_wins_over_km_fsq_default():
    assert _resolved(["--action_encoding", "km_fsq", "--keypress_loss_weight", "0.5"]) == 0.5


def test_equals_form_wins_over_km_fsq_default():
    assert _resolved(["--action_encoding", "km_fsq", "--keypress_loss_weight=0.5"]) == 0.5
