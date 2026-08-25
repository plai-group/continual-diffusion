"""Shift a frame mask to match the causal cache debug_actions.build_action_array
builds: cache row i = the action that caused frame i (recorded during frame
i-1's window).

Observing frames 0..n-1 implies row n -- the action that caused frame n,
already decided during frame n-1 -- is known too, one row past the frame
mask's own cutoff. E.g. n=10: frames 0-9 observed, but cache row 10 is also
known. Left unshifted, row n stays latent and the model has to generate the
action that already explains the first frame it's asked to generate.
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
