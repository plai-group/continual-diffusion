"""KL between the training policy and the model's action head (plaicraft-debug#70).

The training policy is a fully specified Markov chain with deterministic emissions, exported
next to the corpus as policy_chain_spec.json. So its per-frame press probabilities, conditioned
on the action prefix the model was teacher-forced with, are an exact forward-filtering result
rather than a Monte Carlo estimate or a learned guess. The model side is one forward pass at
the schedule's maximum sigma, where the noised action carries no information and the Bernoulli
head's optimum collapses to p(press | context).

We report the SUM OF MARGINAL Bernoulli KLs, not a joint KL over the 2^8 action space. A
factorized head cannot represent mutual exclusivity -- 70/30 uncertainty between w-only and
space-only necessarily puts 0.21 on {w, space} -- so a joint KL has an irreducible floor that
shrinks only by driving marginals to 0/1. It would reward overconfidence and punish
calibration. The two differ by the policy's total correlation, which is constant across
checkpoints, so trends are unaffected.

Direction is load-bearing: KL(policy || model) is finite because a sigmoid never outputs
exactly 0. KL(model || policy) is infinite, since the model puts mass on multi-key states the
policy can never produce.

Known residual floor: the level phase's mouse sign is entered 50/50, but it is really the sign
of the pitch the phase is correcting, so look_up is always followed by a downward servo and
look_down by an upward one. A model that learns that correlation is slightly sharper than this
filter. Closing it means declaring the induced sign per relevel phase in the spec. The floor is
small -- level emits no keys, so it only blurs when the next key-emitting phase starts -- and
it is constant across checkpoints.
"""
import json
from pathlib import Path

import numpy as np

SPEC_FILENAME = "policy_chain_spec.json"
N_KEY_DIMS = 8
Q_CLAMP = 1e-6

NO_KEY = -1
KEY_UNCONSTRAINED = -2
SIG_UNCONSTRAINED = -1

# Observed mouse signatures. Dims 8-9 are symlog sums, so zero-ness and sign survive intact.
SIG_ZERO, SIG_DX, SIG_DY_NEG, SIG_DY_POS, SIG_BOTH = range(5)
N_SIGS = 5

_SIG_ALLOWED = {
    "none": (SIG_ZERO,),
    "dx": (SIG_DX,),
    "dy_neg": (SIG_DY_NEG,),
    "dy_pos": (SIG_DY_POS,),
    "dy_any": (SIG_DY_NEG, SIG_DY_POS),
    "both": (SIG_BOTH,),
}


def find_spec(*roots):
    """First policy_chain_spec.json found at any of the given corpus roots, else None."""
    for root in roots:
        if root is None:
            continue
        path = Path(root)
        path = path if path.suffix == ".json" else path / SPEC_FILENAME
        if path.exists():
            return path
    return None


def load_spec(path):
    return json.loads(Path(path).read_text())


def observed_signature(dx, dy):
    if dx == 0.0:
        return SIG_ZERO if dy == 0.0 else (SIG_DY_NEG if dy < 0.0 else SIG_DY_POS)
    return SIG_DX if dy == 0.0 else SIG_BOTH


def _uniform(lo, hi):
    return [(k, 1.0 / (hi - lo + 1)) for k in range(lo, hi + 1)]


class PolicyChain:
    """Enumerated (phase, frames-remaining, phase-local latent) chain with exact filtering."""

    def __init__(self, spec):
        self.spec = spec
        self.dims = list(spec["action_dims"])
        self._key_index = {k: i for i, k in enumerate(self.dims)}
        self._enumerate()
        self._build_transitions()
        self._build_stationary()

    # -- state space ------------------------------------------------------------------------

    def _enumerate(self):
        states = []
        for name, p in self.spec["phases"].items():
            arch = p["arch"]
            if arch == "none":
                # A phase's mouse sign is drawn once and held for the whole phase (level servos
                # monotonically home), so it belongs in the state. Without it the filter cannot
                # rule out level after seeing six consecutive up-flicks.
                states += [(name, sig, k) for sig in _SIG_ALLOWED[p["mouse"]]
                           for k in range(1, p["dur"][1] + 1)]
            elif arch == "hold":
                states += [(name, k) for k in range(1, p["dur"][1] + 1)]
            elif arch == "lead_hold":
                states += [(name, key, lead, k)
                           for key in p["keys"]
                           for lead in range(0, p["lead"][1] + 1)
                           for k in range(1, p["dur"][1] + 1)]
            elif arch == "pulse":
                states += [(name, on, k) for on in (True, False)
                           for k in range(1, p["dur"][1] + 1)]
            elif arch == "plan":
                for nb in range(p["bursts"][0], p["bursts"][1] + 1):
                    for seg in range(nb):
                        states += [(name, nb, seg, "press", k)
                                   for k in range(1, p["press"][1] + 1)]
                        states += [(name, nb, seg, "gap", k)
                                   for k in range(1, p["gap"][1] + 1)]
            else:
                raise ValueError(f"unknown archetype {arch!r} for phase {name!r}")
        self.states = states
        self.index = {s: i for i, s in enumerate(states)}

        n = len(states)
        self.emit = np.full(n, NO_KEY, dtype=np.int64)
        self.sig_ok = np.zeros((n, N_SIGS), dtype=bool)
        for i, s in enumerate(states):
            self.emit[i] = self._emission(s)
            phase = self.spec["phases"][s[0]]
            codes = (s[1],) if phase["arch"] == "none" else _SIG_ALLOWED[phase["mouse"]]
            for code in codes:
                self.sig_ok[i, code] = True

    def _emission(self, s):
        name = s[0]
        p = self.spec["phases"][name]
        arch = p["arch"]
        if arch == "none":
            return NO_KEY
        if arch == "hold":
            return self._key_index[p["key"]]
        if arch == "lead_hold":
            _, key, lead, _ = s
            return self._key_index[p["lead_key"] if lead > 0 else key]
        if arch == "pulse":
            return self._key_index[p["key"]] if s[1] else NO_KEY
        return self._key_index[p["key"]] if s[3] == "press" else NO_KEY

    # -- transitions ------------------------------------------------------------------------

    def _entry(self, name):
        """Distribution over a phase's first-frame states, as {state: prob}."""
        p = self.spec["phases"][name]
        arch = p["arch"]
        out = {}
        if arch == "none":
            sigs = _SIG_ALLOWED[p["mouse"]]
            for sig in sigs:
                for k, w in _uniform(*p["dur"]):
                    out[(name, sig, k)] = w / len(sigs)
        elif arch == "hold":
            for k, w in _uniform(*p["dur"]):
                out[(name, k)] = w
        elif arch == "lead_hold":
            leads = [(0, 1.0 - p["lead_prob"])]
            leads += [(L, p["lead_prob"] * w) for L, w in _uniform(*p["lead"])]
            for key in p["keys"]:
                for lead, wl in leads:
                    for k, wk in _uniform(*p["dur"]):
                        st = (name, key, lead, k)
                        out[st] = out.get(st, 0.0) + wl * wk / len(p["keys"])
        elif arch == "pulse":
            for k, w in _uniform(*p["dur"]):
                out[(name, True, k)] = w
        elif arch == "plan":
            nb_lo, nb_hi = p["bursts"]
            wb = 1.0 / (nb_hi - nb_lo + 1)
            for nb in range(nb_lo, nb_hi + 1):
                for k, w in _uniform(*p["press"]):
                    out[(name, nb, 0, "press", k)] = wb * w
        return out

    def _phase_end(self, name):
        """Where the chain goes when a phase runs out: level if it moved the pitch, else the menu."""
        if self.spec["phases"][name]["relevel"]:
            return self._entry(self.spec["relevel_phase"])
        menu = self.spec["menu"]
        total = float(sum(menu.values()))
        out = {}
        for nxt, weight in menu.items():
            for st, w in self._entry(nxt).items():
                out[st] = out.get(st, 0.0) + w * weight / total
        return out

    def _successors(self, s):
        name = s[0]
        p = self.spec["phases"][name]
        arch = p["arch"]
        if arch == "none":
            return {(name, s[1], s[2] - 1): 1.0} if s[2] > 1 else self._phase_end(name)
        if arch == "hold":
            return {(name, s[1] - 1): 1.0} if s[1] > 1 else self._phase_end(name)
        if arch == "lead_hold":
            _, key, lead, k = s
            if lead > 0:
                return {(name, key, lead - 1, k): 1.0}
            return {(name, key, 0, k - 1): 1.0} if k > 1 else self._phase_end(name)
        if arch == "pulse":
            _, on, k = s
            if k <= 1:
                return self._phase_end(name)
            f = p["flip_prob"]
            return {(name, on, k - 1): 1.0 - f, (name, not on, k - 1): f}
        _, nb, seg, kind, k = s
        if k > 1:
            return {(name, nb, seg, kind, k - 1): 1.0}
        if kind == "press":
            return {(name, nb, seg, "gap", kk): w for kk, w in _uniform(*p["gap"])}
        if seg + 1 < nb:
            return {(name, nb, seg + 1, "press", kk): w for kk, w in _uniform(*p["press"])}
        return self._phase_end(name)

    def _build_transitions(self):
        n = len(self.states)
        T = np.zeros((n, n), dtype=np.float64)
        for i, s in enumerate(self.states):
            for nxt, w in self._successors(s).items():
                T[i, self.index[nxt]] += w
        rows = T.sum(axis=1)
        assert np.allclose(rows, 1.0), f"transition rows do not normalise: {rows.min()}..{rows.max()}"
        self.T = T

    def _build_stationary(self, iters=2000, tol=1e-14):
        b = np.full(len(self.states), 1.0 / len(self.states))
        for _ in range(iters):
            nb = b @ self.T
            if np.abs(nb - b).max() < tol:
                b = nb
                break
            b = nb
        self.stationary = b / b.sum()

    # -- filtering --------------------------------------------------------------------------

    def emission_probs(self, belief):
        """Per-dim press probability under a belief over states."""
        out = np.zeros(N_KEY_DIMS)
        for d in range(N_KEY_DIMS):
            out[d] = belief[self.emit == d].sum()
        return out

    def condition(self, belief, key, sig):
        """Posterior after observing one frame. Returns (belief, collapsed)."""
        mask = np.ones(len(self.states), dtype=bool)
        if key != KEY_UNCONSTRAINED:
            mask &= (self.emit == key)
        if sig != SIG_UNCONSTRAINED:
            mask &= self.sig_ok[:, sig]
        nb = belief * mask
        total = nb.sum()
        if total <= 0.0:
            return belief, True      # observation outside the policy's support; keep the prior
        return nb / total, False

    def predict(self, obs_keys, obs_sigs, n_future):
        """Filter the observed prefix, then roll forward. Returns (n_future, 8) press probs."""
        belief = self.stationary
        collapses = 0
        for key, sig in zip(obs_keys, obs_sigs):
            belief, bad = self.condition(belief, int(key), int(sig))
            collapses += int(bad)
            belief = belief @ self.T
        out = np.zeros((n_future, N_KEY_DIMS))
        for i in range(n_future):
            out[i] = self.emission_probs(belief)
            belief = belief @ self.T
        return out, collapses


# -- metrics ---------------------------------------------------------------------------------

def bernoulli_kl(p, q, eps=Q_CLAMP):
    """Elementwise KL(Bernoulli(p) || Bernoulli(q)). q must be clamped or a saturated head is inf."""
    q = np.clip(q, eps, 1.0 - eps)
    p = np.clip(p, 0.0, 1.0)
    a = np.where(p > 0.0, p * np.log(np.where(p > 0.0, p, 1.0) / q), 0.0)
    b = np.where(p < 1.0, (1.0 - p) * np.log(np.where(p < 1.0, 1.0 - p, 1.0) / (1.0 - q)), 0.0)
    return a + b


def observations_from_actions(actions):
    """(T, >=10) ground-truth action rows -> (key index per row, mouse signature per row)."""
    keys = np.asarray(actions)[:, :N_KEY_DIMS] > 0.5
    counts = keys.sum(axis=1)
    idx = np.where(counts == 1, keys.argmax(axis=1), np.where(counts == 0, NO_KEY, KEY_UNCONSTRAINED))
    mouse = np.asarray(actions)[:, N_KEY_DIMS:N_KEY_DIMS + 2]
    sigs = np.array([observed_signature(float(dx), float(dy)) for dx, dy in mouse])
    return idx.astype(np.int64), sigs


def evaluate(chain, gt_actions, q_model, n_obs, n_bins=10):
    """Sum-of-marginals KL plus the two diagnostics that say whether to trust it.

    gt_actions: (B, T, >=10) ground truth. q_model: (B, T, 8) head probabilities at max sigma.
    Rows 0..n_obs are the teacher-forced prefix (the action mask lags the frame mask by one);
    rows n_obs+1.. are what the model generated.
    """
    gt = np.asarray(gt_actions, dtype=np.float64)
    q = np.asarray(q_model, dtype=np.float64)
    B, T = gt.shape[0], gt.shape[1]
    first_gen = n_obs + 1
    n_future = T - first_gen
    if n_future <= 0:
        raise ValueError(f"no generated action rows for T={T}, n_obs={n_obs}")

    kl_per_dim = np.zeros(N_KEY_DIMS)
    kl_per_frame = np.zeros(n_future)
    collapses = 0
    bin_hits = np.zeros(n_bins)
    bin_q = np.zeros(n_bins)
    bin_p = np.zeros(n_bins)
    multikey = 0
    for b in range(B):
        keys, sigs = observations_from_actions(gt[b])
        p, bad = chain.predict(keys[:first_gen], sigs[:first_gen], n_future)
        collapses += bad
        kl = bernoulli_kl(p, q[b, first_gen:])
        kl_per_dim += kl.sum(axis=0)
        kl_per_frame += kl.sum(axis=1)
        multikey += int(((q[b, first_gen:] > 0.5).sum(axis=1) >= 2).sum())
        # Reliability: does a predicted 0.3 press 30% of the time? Otherwise the KL is measuring
        # calibration error, not policy divergence.
        flat_q = q[b, first_gen:].ravel()
        flat_p = p.ravel()
        idx = np.clip((flat_q * n_bins).astype(int), 0, n_bins - 1)
        np.add.at(bin_hits, idx, 1.0)
        np.add.at(bin_q, idx, flat_q)
        np.add.at(bin_p, idx, flat_p)

    n_rows = B * n_future
    seen = bin_hits > 0
    return {
        "kl_total": float(kl_per_dim.sum() / n_rows),
        "kl_per_dim": {chain.dims[d]: float(kl_per_dim[d] / n_rows) for d in range(N_KEY_DIMS)},
        "kl_per_frame": (kl_per_frame / B).tolist(),
        "multikey_rate": multikey / n_rows,
        "filter_collapse_rate": collapses / (B * first_gen),
        "reliability": {
            "count": bin_hits.tolist(),
            "mean_q": np.where(seen, bin_q / np.maximum(bin_hits, 1), np.nan).tolist(),
            "mean_p": np.where(seen, bin_p / np.maximum(bin_hits, 1), np.nan).tolist(),
        },
        "calibration_error": float(
            np.abs((bin_q - bin_p)[seen]).sum() / max(bin_hits.sum(), 1.0)),
    }


# -- model side --------------------------------------------------------------------------------

def model_press_probabilities(model, diffusion, x0, actions, obs_mask, latent_mask):
    """p(press | context) from one forward pass at the schedule's maximum sigma.

    At sigma_max the noised action carries no information about the true one, so the Bernoulli
    head's optimum collapses from E[a0 | a_t, context] to E[a0 | context] -- which for a binary
    a0 is exactly the press probability. No Monte Carlo, no sampling loop.
    """
    import torch as th
    from .action_masks import frame_mask_to_action_mask

    t = diffusion.num_timesteps - 1
    sigma_max = float(diffusion.timestep2sigma(t))
    scale = float(diffusion.sqrt_alphas_cumprod[t])
    device = x0.device
    with th.no_grad():
        noise_x = th.randn_like(x0) * sigma_max * scale
        noise_a = th.randn_like(actions) * sigma_max * scale
        out = diffusion.p_mean_variance(
            model, noise_x.to(th.float32),
            th.full((x0.shape[0],), t, device=device, dtype=th.int32),
            clip_denoised=True,
            model_kwargs={
                "frame_indices": None, "x0": x0,
                "obs_mask": obs_mask, "latent_mask": latent_mask,
                "actions": noise_a.to(th.float32),
                "actions0": actions,
                "obs_action_mask": frame_mask_to_action_mask(obs_mask),
            },
        )
    act = out.get("pred_actstart")
    if act is None:
        raise RuntimeError("model produced no pred_actstart; is generate_actions set?")
    return act[..., :N_KEY_DIMS].float().cpu().numpy()


def run_policy_kl(model, diffusion, dataset, chain, device, n_windows=64,
                  batch_size=8, n_obs=None, seed=20250824):
    """Filter the policy over held-out corpus windows and score the head against it.

    Held-out corpus windows, not the curated validation tasks: those are human-recorded and hold
    two keys at once, which the policy can never do, so the filter's likelihood there is zero.
    """
    import numpy as _np
    import torch as th

    T = dataset.T
    n_obs = T // 2 if n_obs is None else n_obs
    idx = _np.random.RandomState(seed).choice(
        len(dataset), size=min(n_windows, len(dataset)), replace=False)

    gt_all, q_all = [], []
    for lo in range(0, len(idx), batch_size):
        rows = [dataset[int(i)] for i in idx[lo:lo + batch_size]]
        x0 = th.stack([r[0] for r in rows]).to(device)
        actions = th.stack([r[2] for r in rows]).to(device, dtype=th.float32)
        obs_mask = th.zeros(x0.shape[0], T, 1, 1, 1, device=device)
        obs_mask[:, :n_obs] = 1.0
        q_all.append(model_press_probabilities(
            model, diffusion, x0, actions, obs_mask, 1.0 - obs_mask))
        gt_all.append(actions.cpu().numpy())

    metrics = evaluate(chain, _np.concatenate(gt_all), _np.concatenate(q_all), n_obs)
    metrics["n_windows"] = int(len(idx))
    return metrics


def log_policy_kl(metrics, logger, prefix="val/kl"):
    """Flatten the metrics dict onto the logger; the reliability curve goes out per bin."""
    logger.logkv(f"{prefix}/total", metrics["kl_total"], distributed=False)
    for dim, value in metrics["kl_per_dim"].items():
        logger.logkv(f"{prefix}/dim/{dim}", value, distributed=False)
    for i, value in enumerate(metrics["kl_per_frame"]):
        logger.logkv(f"{prefix}/frame/{i}", value, distributed=False)
    logger.logkv(f"{prefix}/multikey_rate", metrics["multikey_rate"], distributed=False)
    logger.logkv(f"{prefix}/filter_collapse_rate", metrics["filter_collapse_rate"],
                 distributed=False)
    logger.logkv(f"{prefix}/calibration_error", metrics["calibration_error"], distributed=False)
    rel = metrics["reliability"]
    for i, (mq, mp) in enumerate(zip(rel["mean_q"], rel["mean_p"])):
        if rel["count"][i] > 0:
            logger.logkv(f"{prefix}/reliability/bin{i}_predicted", mq, distributed=False)
            logger.logkv(f"{prefix}/reliability/bin{i}_actual", mp, distributed=False)
