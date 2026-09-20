"""Player-homogeneous batching (issue #85).

The default sampler mixes players inside one gradient step, so the MSE the optimizer sees is an
average over two conditional distributions and both embedding rows move on one shared error.
PlayerBlockSampler makes every step single-player. What must hold is exactly that: no batch may
ever straddle two players, however the remainders fall.
"""
import numpy as np
import pytest
import torch

from improved_diffusion.data_sampler import DistributedOfflineSampler, PlayerBlockSampler
from improved_diffusion.debug_dataset import ContinuousDebugDataset
# pytest prepends tests/ to sys.path (no __init__.py), so the fixture builder imports bare.
from test_player_dataset import _corpus


class _Fake(torch.utils.data.Dataset):
    def __init__(self, players):
        self.players = np.asarray(players)

    def window_players(self):
        return self.players

    def __len__(self):
        return len(self.players)

    def __getitem__(self, i):
        return i


def _batches(sampler, batch_size):
    idx = list(sampler)
    assert len(idx) % batch_size == 0
    return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]


# ── the invariant ──────────────────────────────────────────────────────────────────────────

def test_every_batch_is_drawn_from_a_single_player():
    ds = _Fake([0] * 40 + [1] * 40)
    for batch in _batches(PlayerBlockSampler(ds, batch_size=4, num_replicas=1, rank=0), 4):
        assert len({ds.players[i] for i in batch}) == 1


def test_the_default_sampler_does_mix_players():
    """The control: without the toggle, batches straddle players -- the behaviour being ablated."""
    ds = _Fake([0] * 40 + [1] * 40)
    batches = _batches(DistributedOfflineSampler(ds, batch_size=4, num_replicas=1, rank=0), 4)
    assert any(len({ds.players[i] for i in b}) > 1 for b in batches)


def test_both_players_are_still_seen():
    ds = _Fake([0] * 40 + [1] * 40)
    seen = {ds.players[b[0]] for b in _batches(PlayerBlockSampler(ds, 4, num_replicas=1, rank=0), 4)}
    assert seen == {0, 1}


def test_batch_order_is_shuffled_not_player_blocked():
    """Sorting the index by player would also impose a curriculum; only composition should change."""
    ds = _Fake([0] * 200 + [1] * 200)
    first = [ds.players[b[0]] for b in _batches(PlayerBlockSampler(ds, 4, num_replicas=1, rank=0), 4)]
    assert 0 in first[:20] and 1 in first[:20]


def test_an_unequal_split_keeps_every_batch_pure():
    ds = _Fake([0] * 37 + [1] * 11)  # neither count divides the batch size
    for batch in _batches(PlayerBlockSampler(ds, batch_size=5, num_replicas=1, rank=0), 5):
        assert len({ds.players[i] for i in batch}) == 1


def test_a_single_player_corpus_is_allowed():
    ds = _Fake([0] * 40)
    assert len(list(PlayerBlockSampler(ds, 4, num_replicas=1, rank=0))) > 0


# ── refusals ───────────────────────────────────────────────────────────────────────────────

def test_a_dataset_without_window_players_is_refused():
    class Bare(torch.utils.data.Dataset):
        def __len__(self): return 8
        def __getitem__(self, i): return i

    with pytest.raises(TypeError, match="window_players"):
        list(PlayerBlockSampler(Bare(), 4, num_replicas=1, rank=0))


def test_a_player_too_small_to_fill_one_batch_is_refused():
    with pytest.raises(ValueError, match="player-homogeneous batch"):
        list(PlayerBlockSampler(_Fake([0, 0, 1]), batch_size=8, num_replicas=1, rank=0))


# ── window_players() agrees with __getitem__ ───────────────────────────────────────────────

def test_window_players_matches_the_label_getitem_returns(tmp_path):
    """The sampler trusts this mapping; if it drifted, 'homogeneous' batches would be mixed."""
    with pytest.warns(RuntimeWarning):
        ds = ContinuousDebugDataset(_corpus(tmp_path, [0, 1, 1, 0]), window_length=20, num_classes=2)
    labels = ds.window_players()
    assert len(labels) == len(ds)
    for i in range(0, len(ds), 7):
        assert int(ds[i][-1]) == int(labels[i])
