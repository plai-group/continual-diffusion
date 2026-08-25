"""The forward filter must reproduce the policy it claims to describe (plaicraft-debug#70).

If p_policy is wrong, the KL measures the filter's error and not the model's. So the filter is
checked against an independent sampler that reads the same JSON and implements the archetypes
directly -- never against PolicyChain's own transition matrix, which would be circular.
"""
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from improved_diffusion.debug_policy_kl import (
    NO_KEY, KEY_UNCONSTRAINED, SIG_ZERO, SIG_DX, SIG_DY_NEG, SIG_DY_POS, SIG_BOTH,
    N_KEY_DIMS, PolicyChain, bernoulli_kl, evaluate, load_spec, observations_from_actions,
    observed_signature,
)

SPEC_PATH = Path(__file__).parent / "fixtures" / "policy_chain_spec.json"
_SIG_CHOICES = {"none": [SIG_ZERO], "dx": [SIG_DX], "dy_neg": [SIG_DY_NEG],
                "dy_pos": [SIG_DY_POS], "dy_any": [SIG_DY_NEG, SIG_DY_POS], "both": [SIG_BOTH]}


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC_PATH)


@pytest.fixture(scope="module")
def chain(spec):
    return PolicyChain(spec)


def _sample(spec, steps, rng):
    """Independent reference sampler: emits (key index, mouse signature) per frame."""
    dims = spec["action_dims"]
    names = list(spec["menu"])
    weights = [spec["menu"][n] for n in names]
    out, force_level = [], False
    while len(out) < steps:
        if force_level:
            name, force_level = spec["relevel_phase"], False
        else:
            name = rng.choices(names, weights=weights, k=1)[0]
            force_level = spec["phases"][name]["relevel"]
        p = spec["phases"][name]
        # Drawn once per phase, as the policy does: level's sign is the sign of the pitch it is
        # correcting and does not flip mid-servo.
        chosen = rng.choice(_SIG_CHOICES[p["mouse"]])
        sig = lambda: chosen
        arch = p["arch"]
        if arch == "none":
            out += [(NO_KEY, sig()) for _ in range(rng.randint(*p["dur"]))]
        elif arch == "hold":
            k = dims.index(p["key"])
            out += [(k, sig()) for _ in range(rng.randint(*p["dur"]))]
        elif arch == "lead_hold":
            key = dims.index(rng.choice(p["keys"]))
            lead = rng.randint(*p["lead"]) if rng.random() < p["lead_prob"] else 0
            out += [(dims.index(p["lead_key"]), sig()) for _ in range(lead)]
            out += [(key, sig()) for _ in range(rng.randint(*p["dur"]))]
        elif arch == "pulse":
            k, on = dims.index(p["key"]), True
            for _ in range(rng.randint(*p["dur"])):
                out.append((k if on else NO_KEY, sig()))
                if rng.random() < p["flip_prob"]:
                    on = not on
        elif arch == "plan":
            k = dims.index(p["key"])
            for _ in range(rng.randint(*p["bursts"])):
                out += [(k, sig()) for _ in range(rng.randint(*p["press"]))]
                out += [(NO_KEY, sig()) for _ in range(rng.randint(*p["gap"]))]
    return out[:steps]


def test_transition_matrix_is_stochastic_and_stationary_is_a_fixed_point(chain):
    assert np.allclose(chain.T.sum(axis=1), 1.0)
    assert np.allclose(chain.stationary @ chain.T, chain.stationary, atol=1e-10)
    assert chain.stationary.min() >= 0.0
    assert len(chain.states) == len(set(chain.states))


def test_unconditional_press_probabilities_match_the_sampler(chain, spec):
    seq = _sample(spec, 400000, random.Random(0))
    empirical = np.zeros(N_KEY_DIMS)
    for k, _ in seq:
        if k >= 0:
            empirical[k] += 1
    empirical /= len(seq)
    predicted = chain.emission_probs(chain.stationary)
    # Frames within a phase are perfectly correlated, so the effective sample size is the number
    # of phases; the longest declared phase is the conservative autocorrelation scale.
    ess = len(seq) / 20.0
    for d in range(N_KEY_DIMS):
        tol = 5.0 * math.sqrt(max(predicted[d], 1e-6) / ess) + 1e-4
        assert abs(empirical[d] - predicted[d]) < tol, (
            f"dim {chain.dims[d]}: sampler {empirical[d]:.5f} vs filter {predicted[d]:.5f}")


def test_filter_matches_empirical_conditionals_after_a_prefix(chain, spec):
    # The real claim: after seeing a prefix, the filter's next-frame press probabilities equal
    # the frequencies the policy actually produces given that same prefix.
    prefix_len, horizon = 6, 3
    seq = _sample(spec, 600000, random.Random(1))
    buckets = defaultdict(list)
    for i in range(len(seq) - prefix_len - horizon):
        key = tuple(seq[i:i + prefix_len])
        buckets[key].append([seq[i + prefix_len + h][0] for h in range(horizon)])

    common = sorted(buckets, key=lambda k: -len(buckets[k]))[:8]
    assert len(buckets[common[0]]) >= 500, "not enough samples in the commonest prefix"
    for pref in common:
        futures = buckets[pref]
        n = len(futures)
        if n < 500:
            continue
        predicted, collapsed = chain.predict([k for k, _ in pref], [s for _, s in pref], horizon)
        assert collapsed == 0, "a prefix the sampler produced was outside the filter's support"
        counts = np.zeros((horizon, N_KEY_DIMS))
        for fut in futures:
            for h, k in enumerate(fut):
                if k >= 0:
                    counts[h, k] += 1
        empirical = counts / n
        for h in range(horizon):
            for d in range(N_KEY_DIMS):
                p, q = predicted[h, d], empirical[h, d]
                tol = 5.0 * math.sqrt(max(p, 1e-6) * (1 - max(p, 1e-6)) / n) + 0.01
                assert abs(p - q) < tol, (
                    f"prefix {pref} h={h} dim {chain.dims[d]}: "
                    f"filter {p:.4f} vs empirical {q:.4f} (n={n}, tol={tol:.4f})")


def test_observations_outside_support_do_not_crash_the_filter(chain):
    # Human-recorded sessions hold two keys at once, which the policy can never do. The filter
    # must fall back rather than divide by zero.
    keys = np.array([KEY_UNCONSTRAINED, 0, NO_KEY])
    sigs = np.array([SIG_ZERO, SIG_BOTH, SIG_ZERO])   # 'w' with a free_look signature is impossible
    probs, collapses = chain.predict(keys, sigs, 4)
    assert collapses >= 1
    assert np.all(np.isfinite(probs)) and probs.min() >= 0.0


def test_bernoulli_kl_matches_the_closed_form_and_is_finite_at_saturation():
    p = np.array([0.0, 0.25, 0.5, 1.0])
    q = np.array([0.1, 0.25, 0.9, 0.5])
    want = np.array([
        math.log(1 / 0.9),
        0.0,
        0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1),
        math.log(1 / 0.5),
    ])
    assert np.allclose(bernoulli_kl(p, q), want)
    assert np.isfinite(bernoulli_kl(np.array([1.0]), np.array([0.0]))).all()
    assert bernoulli_kl(np.array([0.3]), np.array([0.3]))[0] == pytest.approx(0.0)


def test_observations_from_actions_reads_keys_and_mouse_signs():
    a = np.zeros((4, 10))
    a[0, 0] = 1.0                      # w
    a[1, 4] = 1.0; a[1, 8] = -2.0      # space with a leftward flick
    a[2, 9] = 3.0                      # no key, look down
    a[3, 0] = 1.0; a[3, 1] = 1.0       # two keys at once: the policy cannot do this
    keys, sigs = observations_from_actions(a)
    assert list(keys) == [0, 4, NO_KEY, KEY_UNCONSTRAINED]
    assert list(sigs) == [SIG_ZERO, SIG_DX, SIG_DY_POS, SIG_ZERO]
    assert observed_signature(-1.0, 2.0) == SIG_BOTH


def test_evaluate_reports_kl_diagnostics_and_bottoms_out_on_a_perfect_model(chain, spec):
    rng = random.Random(3)
    B, T, n_obs = 6, 20, 10
    gt = np.zeros((B, T, 10))
    for b in range(B):
        for t, (k, sig) in enumerate(_sample(spec, T, rng)):
            if k >= 0:
                gt[b, t, k] = 1.0
            gt[b, t, 8] = {SIG_DX: 1.0, SIG_BOTH: 1.0}.get(sig, 0.0)
            gt[b, t, 9] = {SIG_DY_NEG: -1.0, SIG_DY_POS: 1.0, SIG_BOTH: 1.0}.get(sig, 0.0)

    first_gen = n_obs + 1
    perfect = np.zeros((B, T, N_KEY_DIMS))
    for b in range(B):
        keys, sigs = observations_from_actions(gt[b])
        p, _ = chain.predict(keys[:first_gen], sigs[:first_gen], T - first_gen)
        perfect[b, first_gen:] = p

    good = evaluate(chain, gt, perfect, n_obs)
    bad = evaluate(chain, gt, np.full((B, T, N_KEY_DIMS), 0.5), n_obs)
    # Not exactly zero: clamping q to Q_CLAMP costs ~1e-6 per dim on the rows where p is 0.
    assert good["kl_total"] == pytest.approx(0.0, abs=1e-4)
    assert bad["kl_total"] > 1.0
    assert set(good["kl_per_dim"]) == set(chain.dims[:N_KEY_DIMS])
    assert len(good["kl_per_frame"]) == T - first_gen
    assert 0.0 <= good["multikey_rate"] <= 1.0
    assert bad["multikey_rate"] == 0.0        # a flat 0.5 head never crosses the threshold
    assert good["filter_collapse_rate"] == 0.0
    assert len(good["reliability"]["count"]) == 10


# ── model side: the readout must work against a real VDT before a GPU job depends on it ────────

class _FakeDataset:
    """Held-out corpus windows, without needing a corpus."""

    def __init__(self, spec, n=6, T=20, C=3, H=32, W=32, seed=7):
        self.T, self.n = T, n
        rng = random.Random(seed)
        self.frames = [np.random.RandomState(seed + i).randn(T, C, H, W).astype("float32")
                       for i in range(n)]
        self.actions = []
        for _ in range(n):
            a = np.zeros((T, 10), dtype="float32")
            for t, (k, sig) in enumerate(_sample(spec, T, rng)):
                if k >= 0:
                    a[t, k] = 1.0
                a[t, 8] = 1.0 if sig in (SIG_DX, SIG_BOTH) else 0.0
                a[t, 9] = {SIG_DY_NEG: -1.0, SIG_DY_POS: 1.0, SIG_BOTH: 1.0}.get(sig, 0.0)
            self.actions.append(a)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import torch
        return (torch.from_numpy(self.frames[i]), torch.arange(self.T),
                torch.from_numpy(self.actions[i]))


def _build_vdt(T=20):
    from improved_diffusion.script_util import create_vdt_model_and_diffusion
    return create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(32, 32), in_channels=3,
        num_frames=T, learn_sigma=False, sigma_small=False, diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False,
        predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False, action_dim=10,
        action_dropout_prob=0.0, generate_actions=True, action_loss_weight=1.0,
    )


def test_model_press_probabilities_are_in_range_and_vary_with_context(spec):
    import torch
    from improved_diffusion.debug_policy_kl import model_press_probabilities

    model, diffusion = _build_vdt()
    with torch.no_grad():
        for p in model.action_head.linear.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ds = _FakeDataset(spec, n=4)
    x0 = torch.stack([ds[i][0] for i in range(4)])
    actions = torch.stack([ds[i][2] for i in range(4)])
    obs = torch.zeros(4, ds.T, 1, 1, 1)
    obs[:, :10] = 1.0

    q = model_press_probabilities(model, diffusion, x0, actions, obs, 1.0 - obs)
    assert q.shape == (4, ds.T, N_KEY_DIMS)
    assert q.min() >= 0.0 and q.max() <= 1.0
    assert q.std() > 0.0, "head output does not depend on the context at all"


def test_run_policy_kl_end_to_end(chain, spec):
    from improved_diffusion.debug_policy_kl import run_policy_kl

    model, diffusion = _build_vdt()
    metrics = run_policy_kl(model, diffusion, _FakeDataset(spec), chain,
                            device="cpu", n_windows=4, batch_size=2)
    assert metrics["n_windows"] == 4
    assert np.isfinite(metrics["kl_total"]) and metrics["kl_total"] >= 0.0
    assert set(metrics["kl_per_dim"]) == set(chain.dims[:N_KEY_DIMS])
    # The head is zero-initialised, so every probability is 0.5 and the KL is the distance from
    # the policy's marginals to a coin flip -- large, finite, and identical across dims that the
    # policy never presses.
    assert metrics["kl_total"] > 1.0
    assert metrics["filter_collapse_rate"] == 0.0


def test_log_policy_kl_emits_every_series(chain, spec):
    from improved_diffusion.debug_policy_kl import log_policy_kl, run_policy_kl

    class _Logger:
        def __init__(self): self.kv = {}
        def logkv(self, k, v, distributed=True): self.kv[k] = v

    model, diffusion = _build_vdt()
    metrics = run_policy_kl(model, diffusion, _FakeDataset(spec), chain,
                            device="cpu", n_windows=2, batch_size=2)
    log = _Logger()
    log_policy_kl(metrics, log)
    assert "val/kl/total" in log.kv
    assert "val/kl/multikey_rate" in log.kv
    assert "val/kl/calibration_error" in log.kv
    assert any(k.startswith("val/kl/dim/") for k in log.kv)
    assert any(k.startswith("val/kl/frame/") for k in log.kv)
    assert any(k.startswith("val/kl/reliability/") for k in log.kv)
    assert all(np.isfinite(v) for v in log.kv.values())
