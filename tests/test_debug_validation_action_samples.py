"""plaicraft-debug#106: action_samples=N scores key CE on the mean decoded key probability over N samples; N=1 is the single-sample path."""
import torch

import improved_diffusion.debug_validation as dv

T, N_OBS, N = 20, 10, 3


class _Valset:
    T, n_observed = T, N_OBS
    rows = [{"num": i, "prompt": "p", "test_type": "t"} for i in range(N)]

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.keys79 = (torch.rand(N, T, 79, generator=g) < 0.1).float()
        self.codes = torch.randn(N, T, 32, generator=g)

    def load_all(self):
        return torch.zeros(N, T, 3, 8, 8)

    def load_all_actions(self):
        return self.codes, None

    load_all_actions_raw = load_all_actions

    def load_all_keys79(self):
        return self.keys79

    def slug(self, row):
        return str(row["num"])


class _Diffusion:
    num_timesteps, action_quantization, action_encoding = 100, "none", "km_ctx14"

    def timestep2sigma(self, t):
        return 1000.0

    def heun_sample(self, model, shape, **kw):
        return (torch.zeros(*shape).double(), (torch.randn(shape[0], shape[1], 32).double(), None)), None


class _Model(torch.nn.Module):
    action_dim, mouse_dim, generate_actions, generate_mouse = 32, 0, True, False


def _run(monkeypatch, tmp_path, n):
    monkeypatch.setattr(dv, "_METRICS", lambda p, g: {"psnr": 1.0})
    monkeypatch.setattr(dv, "_get_km_tokenizer", lambda *a, **k: None)
    probs = []

    def decode(tok, codes, all_keys=False):
        p = torch.sigmoid(codes[..., :1].float()).expand(*codes.shape[:2], 79)
        probs.append(p)
        return (p > 0.5).float(), torch.zeros(*codes.shape[:2], 2), p

    seen = []
    real = dv._action_metrics
    monkeypatch.setattr(dv, "_decode_km_actions", decode)
    monkeypatch.setattr(dv, "_action_metrics", lambda *a, **k: (seen.append(k["pred_key_prob"]), real(*a, **k))[1])
    dv.run_debug_validation(_Model(), _Diffusion(), _Valset(), "cpu", tmp_path, chunk_size=N, log_videos=False,
                            swap_test=False, fvd_repeats=0, action_samples=n)
    return probs, seen


def test_n1_scores_the_single_sample(monkeypatch, tmp_path):
    probs, seen = _run(monkeypatch, tmp_path, 1)
    assert torch.equal(seen[0], probs[0][0])


def test_n_averages_key_probabilities_over_samples(monkeypatch, tmp_path):
    probs, seen = _run(monkeypatch, tmp_path, 4)
    pred = [probs[0]] + probs[2:]  # probs[1] is the GT decode
    assert len(pred) == 4 and not torch.equal(pred[0], pred[1])
    assert torch.allclose(seen[0], torch.stack(pred).mean(0)[0], atol=1e-6)


def test_gt_decode_floor_scores_the_gt_codes_against_true_keys(monkeypatch, tmp_path):
    vs = _Valset()
    monkeypatch.setattr(dv, "_METRICS", lambda p, g: {"psnr": 1.0})
    monkeypatch.setattr(dv, "_get_km_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(dv, "_decode_km_actions", lambda tok, codes, all_keys=False: (
        (p := vs.keys79.clamp(1e-3, 1 - 1e-3)).round(), torch.zeros(N, T, 2), p) if torch.equal(codes, vs.codes) else
        (torch.zeros(N, T, 79), torch.zeros(N, T, 2), torch.full((N, T, 79), 0.5)))
    agg = dv.run_debug_validation(_Model(), _Diffusion(), vs, "cpu", tmp_path, chunk_size=N, log_videos=False,
                                  swap_test=False, fvd_repeats=0)["aggregate"]
    assert agg["val/action_roll/floor_key_jaccard_distance"] == 0.0
    assert agg["val/action_roll/floor_key_cross_entropy"] < 0.1 < agg["val/action_roll/key_cross_entropy"]


def test_per_tick_metrics_and_persistence_baseline(monkeypatch, tmp_path):
    vs = _Valset()
    monkeypatch.setattr(dv, "_METRICS", lambda p, g: {"psnr": 1.0})
    monkeypatch.setattr(dv, "_get_km_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(dv, "_decode_km_actions", lambda tok, codes, all_keys=False: (
        (p := vs.keys79.clamp(1e-3, 1 - 1e-3)).round(), torch.zeros(N, T, 2), p) if torch.equal(codes, vs.codes) else
        (torch.zeros(N, T, 79), torch.zeros(N, T, 2), torch.full((N, T, 79), 0.5)))
    agg = dv.run_debug_validation(_Model(), _Diffusion(), vs, "cpu", tmp_path, chunk_size=N, log_videos=False,
                                  swap_test=False, fvd_repeats=0)["aggregate"]
    first = N_OBS + 1
    assert all(f"val/action_tick/{i}/key_cross_entropy" in agg for i in range(T - first)) and f"val/action_tick/{T - first}/key_cross_entropy" not in agg
    q = vs.keys79.reshape(-1, 79).mean(0)
    p, last = 0.9 * vs.keys79[:, first - 1] + 0.1 * q, vs.keys79[:, first - 1]
    for scope, ticks in (("val/action", [first]), ("val/action_tick/2", [first + 2])):
        y = vs.keys79[:, ticks]
        ce = -(y * p[:, None].log() + (1 - y) * (1 - p[:, None]).log()).sum(-1).mean()
        assert abs(agg[f"{scope}/key_ce_persist"] - ce.item()) < 1e-4
        tp, fp, fn = [((y[:, 0] == a) & (last == b)).sum(-1).float() for a, b in ((1, 1), (0, 1), (1, 0))]
        jac = torch.where(tp + fp + fn > 0, 1 - tp / (tp + fp + fn).clamp(min=1), torch.zeros(N)).mean()
        assert abs(agg[f"{scope}/key_jaccard_persist"] - jac.item()) < 1e-6
