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


def keypress_ce_baserate(y):
    """Same CE, scored against y's own per-key base rate, the true p across all actions/frames."""
    q = y.reshape(-1, y.shape[-1]).mean(dim=0)
    q = q.clamp(0.0, 1.0)
    # xlogy is used to guard against log(0) errors. 
    per_frame = -(torch.special.xlogy(y, q) + torch.special.xlogy(1 - y, 1 - q)).sum(dim=-1)
    return per_frame.mean()
