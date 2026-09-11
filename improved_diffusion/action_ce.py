"""Bernoulli cross-entropy over the 8 compact keypress dims, encoding-agnostic.

Callers hand this module a per-key probability and the true multi-hot. Nothing here
imports the tokenizer, branches on action_encoding, or knows what produced p.

Departs from issue #76's keypress_cross_entropy in three ways: scores p as a probability,
not a logit (raw/raw_fused heads are MSE-trained toward 0/1 and already live in [0,1] --
running that through logsigmoid would score a perfect 1.0 as 0.31 nats instead of 0); sums
the full Bernoulli (both the pressed and unpressed term), not #76's positive-dims-only form,
so false presses are penalised; and means over ALL frames, not just pressed ones -- #76's
pressed-only mean existed only because a positive-only CE scores a keyless frame at exactly
0, a degeneracy that full Bernoulli no longer has.

We sum over the action space but average over frames
"""
import torch
import torch.nn.functional as F

EPS = 1e-6


def keypress_cross_entropy(p, y):
    """Bernoulli CE of probabilities p against multi-hot y, summed over keys, meaned over frames."""
    # clamp to EPS so that if model predicts 0 or 1, we don't get log(0) errors.
    p = p.clamp(EPS, 1 - EPS)
    return F.binary_cross_entropy(p, y, reduction="none").sum(dim=-1).mean()


def keypress_ce_baserate(y, q=None):
    """Same CE, scored against the per-key base rate q -- what a predictor that ignores
    the frames entirely would have paid on y. Pass q measured over the whole validation
    set; deriving it from y alone is degenerate on a short window (a single frame gives
    q == y, hence exactly 0).

    The validation pool over-states the rare keys relative to the training policy -- the
    exercises are curated, and five keys are held through exactly one exercise each -- so
    read this as an anchor to beat, not as the policy's press rate."""
    if q is None:
        q = y.reshape(-1, y.shape[-1]).mean(dim=0)
    q = q.clamp(0.0, 1.0)
    # xlogy is used to guard against log(0) errors.
    per_frame = -(torch.special.xlogy(y, q) + torch.special.xlogy(1 - y, 1 - q)).sum(dim=-1)
    return per_frame.mean()
