"""Alignment between the frame mask and the action mask.

The action cache is CAUSAL: row i holds the action from window [i-1, i), i.e.
the one that produced frame i. The environment is a_t -> s_{t+1}, so the action
*taken at* frame i lives at cache row i+1.

A state is (frame, action). Observing the first n frames therefore means
observing n frames AND the n actions taken during them -- cache rows 1..n --
so the action mask is the frame mask shifted one step later. Row 0 is the
action that produced the first frame; it precedes the window, so it inherits
frame 0's status rather than being derived from a predecessor.

Using the frame mask directly leaves cache[n] latent, which makes the model
generate the action at the boundary frame -- the very action that is supposed
to explain the first generated frame.
"""


def frame_mask_to_action_mask(mask):
    """(B, T, ...) frame mask -> (B, T, 1) action mask, shifted one step later."""
    m = mask
    if m.ndim > 3:
        m = m.view(m.shape[0], m.shape[1], 1)
    elif m.ndim == 2:
        m = m.unsqueeze(-1)
    out = m.clone()
    out[:, 1:] = m[:, :-1]
    return out
