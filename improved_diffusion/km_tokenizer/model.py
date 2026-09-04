"""Vendored from PLAIOmni's keyboard_mouse_tokenizer.py (plaicraft-debug#80).

Upstream commit: bdc567e8b53ee96857b77126cf71d035b297ddbd. Two changes from
upstream beyond the trim below: `codec.base.BaseStreamEncoder` is inlined (it
was thin -- device placement + a rate check) and `core.grid.conform_to_ticks`
is dropped entirely along with the `encode_window` method that used it -- VDT
calls `_encode_prequantized` / `_quantize` / `decode_codes` directly. The
`vector_quantize_pytorch` try/except around FSQ is also removed: only the
vendored fallback remains, so the container's package set can never silently
choose which FSQ quantizes the corpus (it does not ship the package anyway).
Checkpoint loading no longer touches huggingface_hub -- see `load_tokenizer`.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rate of the raw key/mouse frames this tokenizer consumes.
CONTROL_FPS = 100.0
# The 12.5 Hz action grid every tokenizer output lands on (core.grid.TICK_HZ upstream).
TICK_HZ = 12.5

DEFAULT_CHECKPOINT = Path(
    "/ubc/cs/research/plai-scratch/ctardy/models/km_fsq_12x240/pytorch_model.bin"
)
CHECKPOINT_SHA256 = "8984f0409862c2579b9636a566b771d678a1fb166c8eca22f8612d18d1e11cdf"


class BaseStreamEncoder:
    """Vendored from codec.base.BaseStreamEncoder: device placement + a native-rate check."""

    native_rate: float = 0.0

    def ensure_loaded(self, device: Optional[torch.device] = None) -> "BaseStreamEncoder":
        return self if device is None else self.to(device)

    def check_rate(self, observed: Optional[float] = None) -> None:
        rate = self.native_rate if observed is None else observed
        if abs(float(rate) - TICK_HZ) > 1e-6:
            raise RuntimeError(
                f"{type(self).__name__} runs at {rate} Hz but the action grid is "
                f"{TICK_HZ} Hz -- one encoder frame per action tick is required."
            )


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    rounded = x.round()
    return x + (rounded - x).detach()


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al., ICLR 2024). Vendored fallback only --
    see the module docstring for why the upstream vector_quantize_pytorch import is gone."""

    def __init__(
        self,
        levels: list[int],
        dim: int,
        force_quantization_f32: bool = True,
    ) -> None:
        super().__init__()
        if len(levels) != dim:
            raise ValueError(f"Expected {dim} FSQ levels, got {len(levels)}.")

        levels_tensor = torch.tensor(levels, dtype=torch.int32)
        basis = torch.cumprod(
            torch.tensor([1, *levels[:-1]], dtype=torch.int32),
            dim=0,
        )
        self.dim = int(dim)
        self.force_quantization_f32 = bool(force_quantization_f32)
        self.register_buffer("_levels", levels_tensor, persistent=False)
        self.register_buffer("_basis", basis, persistent=False)

    def _bound(self, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        levels = self._levels.to(device=x.device, dtype=x.dtype)
        half_l = (levels - 1.0) * (1.0 + eps) / 2.0
        offset = torch.where(
            levels.remainder(2) == 0,
            torch.full_like(levels, 0.5),
            torch.zeros_like(levels),
        )
        shift = torch.atanh(offset / half_l)
        return torch.tanh(x + shift) * half_l - offset

    def _scale_and_shift(self, codes: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels // 2).to(device=codes.device, dtype=codes.dtype)
        return codes * half_width + half_width

    def _scale_and_shift_inverse(self, indices: torch.Tensor) -> torch.Tensor:
        half_width = (
            self._levels // 2
        ).to(device=indices.device, dtype=indices.dtype)
        return (indices - half_width) / half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        level_indices = self._scale_and_shift(codes).round().long()
        levels = self._levels.to(device=codes.device).long()
        level_indices = torch.minimum(level_indices.clamp_min(0), levels - 1)
        basis = self._basis.to(device=codes.device).long()
        return (level_indices * basis).sum(dim=-1).long()

    def indices_to_level_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.long()
        basis = self._basis.to(device=indices.device).long()
        levels = self._levels.to(device=indices.device).long()
        return torch.div(indices[..., None], basis, rounding_mode="floor").remainder(levels)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        level_indices = self.indices_to_level_indices(indices).float()
        return self._scale_and_shift_inverse(level_indices)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        if self.force_quantization_f32:
            x = x.float()

        bounded = self._bound(x)
        quantized = _round_ste(bounded)
        half_width = (self._levels // 2).to(device=x.device, dtype=x.dtype)
        codes = quantized / half_width
        indices = self.codes_to_indices(codes)
        return codes.to(dtype=orig_dtype), indices


@dataclass
class KeyboardMouseTokenizerConfig:
    num_keys: int = 79
    block_size: int = 8

    key_feature_dim: int = 237
    mouse_feature_dim: int = 7

    model_dim: int = 256
    ffn_dim: int = 1024
    num_heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout: float = 0.0

    # Queries always self-attend -- see CrossAttentionBlock. `frame_encoder_layers`
    # is the depth of the self-attention stack that lets a block's own frames
    # contextualize each other before the registers pool them.
    frame_encoder_layers: int = 2

    # Grouped FSQ (Mentzer et al., ICLR 2024) in NanoCodec's arrangement
    # (arXiv 2508.05835): `num_tokens` disjoint groups, each independently
    # quantized to `vocab_size` codes and packed by mixed-radix indexing.
    num_tokens: int = 12
    fsq_levels: tuple[int, ...] = (8, 6, 5)
    fsq_dim: int = 3
    vocab_size: int = 240

    # Semantic factorization of the code groups: keyboard and mouse each route
    # to their own groups, so the encoder has to put key information in the
    # key groups and mouse information in the mouse groups.
    use_semantic_split: bool = True
    key_group_indices: tuple[int, ...] = (0, 1)
    mouse_group_indices: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    # Per-axis bounds, matching the 720p capture.
    mouse_max_abs_dx: int = 1280
    mouse_max_abs_dy: int = 720

    # Non-uniform bins: unit-width integer bins for |delta| <= mouse_linear_bins,
    # then a geometric tail out to the axis bound.
    mouse_linear_bins: int = 96
    mouse_tail_bins: int = 24

    # "expectation" decodes the distribution's mean (soft-argmax), the L1-optimal
    # point estimate; "argmax" decodes the mode.
    mouse_decode: str = "expectation"

    mouse_norm_scale: float = 50.0


@dataclass
class KeyboardMouseTokenizerOutput:
    token_ids: torch.Tensor
    key_logits: torch.Tensor
    mouse_logits: torch.Tensor
    mouse_pred: torch.Tensor
    frame_mask: torch.Tensor
    block_mask: torch.Tensor
    prequantized: torch.Tensor


class CrossAttentionBlock(nn.Module):
    """Cross-attend the queries into a fixed memory, let the queries self-attend, then FFN."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.memory_norm = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_norm = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            query=self.query_norm(query),
            key=self.memory_norm(memory),
            value=self.memory_norm(memory),
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout(attn_out)

        normed = self.self_norm(query)
        self_out, _ = self.self_attn(
            query=normed,
            key=normed,
            value=normed,
            need_weights=False,
        )
        query = query + self.dropout(self_out)

        query = query + self.ffn(query)
        return query


class KeyboardMouseTokenizer(BaseStreamEncoder, nn.Module):
    """
    Grouped-FSQ keyboard+mouse tokenizer over fixed-size control blocks.

    Input:
        key_press:      [B, T, 79]
        mouse_movement: [B, T, 2]

    Output:
        token_ids:    [B, ceil(T / block_size), num_tokens]
        key_logits:   [B, N, block_size, 79]
        mouse_logits: [B, N, block_size, 2, mouse_num_classes]
        mouse_pred:   [B, N, block_size, 2]

    Every block is encoded and decoded independently (blocks live in the batch
    dimension), so a block's tokens never depend on a future block and a block's
    reconstruction never depends on a neighbouring block's tokens. The only
    cross-block term is `make_key_features`' one-frame backward difference,
    which is causal.
    """

    @staticmethod
    def _build_bin_centers(max_abs: float, linear_bins: int, tail_bins: int) -> torch.Tensor:
        """Unit-width integer bins over [-linear_bins, linear_bins], then a
        geometric tail out to +/-max_abs. Tail centers are rounded to integers
        (deltas are integer-valued) and forced strictly increasing."""
        linear_bins = int(linear_bins)
        tail_bins = int(tail_bins)
        max_abs = float(max_abs)

        if linear_bins < 1:
            raise ValueError(f"mouse_linear_bins must be >= 1, got {linear_bins}.")
        if max_abs <= linear_bins:
            raise ValueError(
                f"Axis bound {max_abs} must exceed mouse_linear_bins {linear_bins}."
            )

        core = [float(v) for v in range(-linear_bins, linear_bins + 1)]
        tail: list[float] = []
        if tail_bins > 0:
            ratio = (max_abs / linear_bins) ** (1.0 / tail_bins)
            previous = float(linear_bins)
            for step in range(1, tail_bins + 1):
                value = min(max(round(linear_bins * ratio**step), previous + 1.0), max_abs)
                if value > previous:
                    tail.append(value)
                    previous = value

        centers = [-v for v in reversed(tail)] + core + tail
        return torch.tensor(centers, dtype=torch.float32)

    def __init__(self, config: KeyboardMouseTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        # Control frames arrive at 100 Hz, so block_size frames per block must
        # land on the 12.5 Hz action grid. check_rate() raises if it does not.
        self.native_rate = CONTROL_FPS / float(config.block_size)
        self.check_rate()

        if config.mouse_decode not in ("expectation", "argmax"):
            raise ValueError(
                f"mouse_decode must be 'expectation' or 'argmax', got {config.mouse_decode!r}."
            )

        centers_x = self._build_bin_centers(
            config.mouse_max_abs_dx, config.mouse_linear_bins, config.mouse_tail_bins
        )
        centers_y = self._build_bin_centers(
            config.mouse_max_abs_dy, config.mouse_linear_bins, config.mouse_tail_bins
        )
        if centers_x.numel() != centers_y.numel():
            raise ValueError(
                "dx and dy bin schedules must have equal length so the mouse head stays "
                f"a single tensor; got {centers_x.numel()} and {centers_y.numel()}. "
                "Adjust mouse_linear_bins / mouse_tail_bins."
            )
        self.mouse_num_classes = int(centers_x.numel())

        levels = tuple(int(level) for level in config.fsq_levels)
        if len(levels) != int(config.fsq_dim):
            raise ValueError("fsq_levels length must match fsq_dim.")
        if math.prod(levels) != int(config.vocab_size):
            raise ValueError("The product of fsq_levels must equal vocab_size.")

        self.use_semantic_split = bool(config.use_semantic_split)
        self.key_group_indices = self._validate_groups(
            config.key_group_indices, config.mouse_group_indices, int(config.num_tokens)
        )
        self.mouse_group_indices = [int(index) for index in config.mouse_group_indices]

        self.fsq = FSQ(
            levels=list(levels),
            dim=config.fsq_dim,
            force_quantization_f32=True,
        )

        self.key_stem = nn.Sequential(
            nn.Linear(config.key_feature_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
        )

        self.mouse_stem = nn.Sequential(
            nn.Linear(config.mouse_feature_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
        )

        self.frame_fusion = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )

        self.frame_pos = nn.Parameter(torch.zeros(1, config.block_size, config.model_dim))

        if int(config.frame_encoder_layers) > 0:
            self.frame_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.model_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.ffn_dim,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=int(config.frame_encoder_layers),
                enable_nested_tensor=False,
            )
        else:
            self.frame_encoder = None

        self.register_queries = nn.Parameter(torch.zeros(1, config.num_tokens, config.model_dim))
        self.encoder = self._make_attention_stack(config.encoder_layers)

        self.to_fsq = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.fsq_dim),
        )
        self.from_fsq = nn.Sequential(
            nn.Linear(config.fsq_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
        )

        self.token_pos = nn.Parameter(torch.zeros(1, config.num_tokens, config.model_dim))

        # Only build the decoder stack(s) the active mode uses -- an unused
        # parameter is a hard error under DDP/DeepSpeed, not just dead weight.
        if self.use_semantic_split:
            self.key_frame_queries = self._make_frame_queries()
            self.key_frame_pos = self._make_frame_queries()
            self.key_decoder = self._make_attention_stack(config.decoder_layers)

            self.mouse_frame_queries = self._make_frame_queries()
            self.mouse_frame_pos = self._make_frame_queries()
            self.mouse_decoder = self._make_attention_stack(config.decoder_layers)
        else:
            self.decode_frame_queries = self._make_frame_queries()
            self.decode_frame_pos = self._make_frame_queries()
            self.decoder = self._make_attention_stack(config.decoder_layers)

        self.key_head = nn.Linear(config.model_dim, config.num_keys)
        self.mouse_head = nn.Linear(config.model_dim, 2 * self.mouse_num_classes)

        # [2, num_bins] bin centers, and [2, num_bins-1] midpoint edges used to
        # assign a value to its nearest bin.
        centers = torch.stack([centers_x, centers_y], dim=0)
        self.register_buffer("mouse_bin_centers", centers, persistent=False)
        self.register_buffer(
            "mouse_bin_edges", (centers[:, 1:] + centers[:, :-1]) * 0.5, persistent=False
        )
        self.register_buffer(
            "mouse_axis_max",
            torch.tensor(
                [float(config.mouse_max_abs_dx), float(config.mouse_max_abs_dy)],
                dtype=torch.float32,
            ),
            persistent=False,
        )

        for parameter in self._learned_query_parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    @staticmethod
    def _validate_groups(
        key_group_indices: Sequence[int],
        mouse_group_indices: Sequence[int],
        num_tokens: int,
    ) -> list[int]:
        key_groups = [int(index) for index in key_group_indices]
        mouse_groups = [int(index) for index in mouse_group_indices]

        combined = key_groups + mouse_groups
        if len(set(combined)) != len(combined):
            raise ValueError(
                "key_group_indices and mouse_group_indices must be disjoint, got "
                f"key={key_groups}, mouse={mouse_groups}."
            )
        if sorted(combined) != list(range(num_tokens)):
            raise ValueError(
                "key_group_indices and mouse_group_indices must together cover every "
                f"group in [0, {num_tokens}); got key={key_groups}, mouse={mouse_groups}."
            )
        return key_groups

    def _make_frame_queries(self) -> nn.Parameter:
        return nn.Parameter(torch.zeros(1, self.config.block_size, self.config.model_dim))

    def _make_attention_stack(self, num_layers: int) -> nn.ModuleList:
        return nn.ModuleList(
            [
                CrossAttentionBlock(
                    model_dim=self.config.model_dim,
                    num_heads=self.config.num_heads,
                    ffn_dim=self.config.ffn_dim,
                    dropout=self.config.dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def _learned_query_parameters(self):
        yield self.frame_pos
        yield self.register_queries
        yield self.token_pos
        if self.use_semantic_split:
            yield self.key_frame_queries
            yield self.key_frame_pos
            yield self.mouse_frame_queries
            yield self.mouse_frame_pos
        else:
            yield self.decode_frame_queries
            yield self.decode_frame_pos

    def make_key_features(self, key_press: torch.Tensor) -> torch.Tensor:
        key_press = key_press.to(dtype=self.frame_pos.dtype)

        prev = F.pad(key_press[:, :-1], (0, 0, 1, 0), value=0.0)
        key_down = (key_press > prev).to(dtype=key_press.dtype)
        key_up = (key_press < prev).to(dtype=key_press.dtype)

        return torch.cat([key_press, key_down, key_up], dim=-1)

    def clamp_to_axis_bounds(self, mouse_movement: torch.Tensor) -> torch.Tensor:
        bounds = self.mouse_axis_max.to(device=mouse_movement.device, dtype=mouse_movement.dtype)
        return torch.maximum(torch.minimum(mouse_movement, bounds), -bounds)

    def make_mouse_features(self, mouse_movement: torch.Tensor) -> torch.Tensor:
        cfg = self.config

        mouse_movement = self.clamp_to_axis_bounds(mouse_movement.to(dtype=self.frame_pos.dtype))

        dx = mouse_movement[..., 0]
        dy = mouse_movement[..., 1]

        # Linear term keeps fine resolution where ~99% of the mass lives, but is
        # bounded so a 1280 px pointer warp cannot produce a 25-sigma activation.
        # The log terms carry the full range.
        dx_norm = (dx / float(cfg.mouse_norm_scale)).clamp(-4.0, 4.0)
        dy_norm = (dy / float(cfg.mouse_norm_scale)).clamp(-4.0, 4.0)

        dx_log = torch.log1p(dx.abs()) / math.log1p(float(cfg.mouse_max_abs_dx))
        dy_log = torch.log1p(dy.abs()) / math.log1p(float(cfg.mouse_max_abs_dy))

        dx_sign = torch.sign(dx)
        dy_sign = torch.sign(dy)

        magnitude = torch.sqrt(dx.square() + dy.square())
        max_magnitude = math.hypot(float(cfg.mouse_max_abs_dx), float(cfg.mouse_max_abs_dy))
        magnitude = torch.log1p(magnitude) / math.log1p(max_magnitude)

        return torch.stack(
            [dx_norm, dy_norm, dx_log, dy_log, dx_sign, dy_sign, magnitude],
            dim=-1,
        )

    def mouse_values_to_bins(self, values: torch.Tensor) -> torch.Tensor:
        """[..., 2] real deltas -> [..., 2] nearest-bin indices."""
        clamped = self.clamp_to_axis_bounds(values.float())
        edges = self.mouse_bin_edges.to(device=values.device, dtype=torch.float32)
        bins = [
            torch.bucketize(clamped[..., axis].contiguous(), edges[axis])
            for axis in range(2)
        ]
        return torch.stack(bins, dim=-1)

    def mouse_bins_to_values(self, bins: torch.Tensor) -> torch.Tensor:
        centers = self.mouse_bin_centers.to(device=bins.device)
        return torch.stack(
            [centers[axis][bins[..., axis]] for axis in range(2)],
            dim=-1,
        )

    def mouse_expectation(self, mouse_logits: torch.Tensor) -> torch.Tensor:
        """Soft-argmax: the distribution's mean, in pixels. [..., 2, bins] -> [..., 2]."""
        centers = self.mouse_bin_centers.to(device=mouse_logits.device, dtype=torch.float32)
        probabilities = torch.softmax(mouse_logits.float(), dim=-1)
        return (probabilities * centers).sum(dim=-1)

    def blockify(
        self,
        key_press: torch.Tensor,
        mouse_movement: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if key_press.ndim != 3:
            raise ValueError(f"Expected key_press shape [B, T, K], got {tuple(key_press.shape)}.")

        if mouse_movement.ndim != 3:
            raise ValueError(
                f"Expected mouse_movement shape [B, T, 2], got {tuple(mouse_movement.shape)}."
            )

        if (
            key_press.shape[0] != mouse_movement.shape[0]
            or key_press.shape[1] != mouse_movement.shape[1]
        ):
            raise ValueError(
                "key_press and mouse_movement must have matching batch/time dimensions. "
                f"Got key_press={tuple(key_press.shape)}, "
                f"mouse_movement={tuple(mouse_movement.shape)}."
            )

        if key_press.shape[-1] != self.config.num_keys:
            raise ValueError(
                f"Expected key_press last dim {self.config.num_keys}, got {key_press.shape[-1]}."
            )

        if mouse_movement.shape[-1] != 2:
            raise ValueError(f"Expected mouse_movement last dim 2, got {mouse_movement.shape[-1]}.")

        batch_size, original_t, _ = key_press.shape
        block_size = self.config.block_size

        if lengths is None:
            lengths = torch.full(
                (batch_size,),
                fill_value=original_t,
                dtype=torch.long,
                device=key_press.device,
            )
        else:
            lengths = lengths.to(device=key_press.device, dtype=torch.long).clamp(0, original_t)

        num_blocks = math.ceil(original_t / block_size)
        padded_t = num_blocks * block_size
        pad_t = padded_t - original_t

        if pad_t > 0:
            key_press = F.pad(key_press, (0, 0, 0, pad_t), value=0.0)
            mouse_movement = F.pad(mouse_movement, (0, 0, 0, pad_t), value=0.0)

        key_features = self.make_key_features(key_press)
        mouse_features = self.make_mouse_features(mouse_movement)

        key_blocks = key_features.view(batch_size, num_blocks, block_size, -1)
        mouse_blocks = mouse_features.view(batch_size, num_blocks, block_size, -1)

        frame_positions = torch.arange(padded_t, device=key_press.device).unsqueeze(0)
        frame_mask = frame_positions < lengths.unsqueeze(1)
        frame_mask = frame_mask.view(batch_size, num_blocks, block_size)
        block_mask = frame_mask.any(dim=-1)

        return key_blocks, mouse_blocks, frame_mask, block_mask

    def _frames_from_blocks(
        self,
        key_blocks: torch.Tensor,
        mouse_blocks: torch.Tensor,
    ) -> torch.Tensor:
        key_emb = self.key_stem(key_blocks)
        mouse_emb = self.mouse_stem(mouse_blocks)
        frame_emb = self.frame_fusion(torch.cat([key_emb, mouse_emb], dim=-1))
        return frame_emb + self.frame_pos[:, : self.config.block_size]

    def _encode_prequantized(
        self,
        key_press: torch.Tensor,
        mouse_movement: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_blocks, mouse_blocks, frame_mask, block_mask = self.blockify(
            key_press=key_press,
            mouse_movement=mouse_movement,
            lengths=lengths,
        )

        batch_size, num_blocks, block_size, _ = key_blocks.shape
        frames = self._frames_from_blocks(key_blocks, mouse_blocks)
        frames = frames.reshape(batch_size * num_blocks, block_size, self.config.model_dim)

        flat_frame_mask = frame_mask.reshape(batch_size * num_blocks, block_size)
        memory_padding_mask = ~flat_frame_mask.bool()
        empty_blocks = memory_padding_mask.all(dim=-1)
        if empty_blocks.any():
            # A fully-masked attention row is NaN, not zero. Unmask one slot;
            # the block is zeroed out downstream by `block_mask` anyway.
            memory_padding_mask = memory_padding_mask.clone()
            memory_padding_mask[empty_blocks, 0] = False

        if self.frame_encoder is not None:
            frames = self.frame_encoder(frames, src_key_padding_mask=memory_padding_mask)

        registers = self.register_queries.expand(batch_size * num_blocks, -1, -1)
        for layer in self.encoder:
            registers = layer(
                query=registers,
                memory=frames,
                memory_key_padding_mask=memory_padding_mask,
            )

        prequantized = self.to_fsq(registers)
        prequantized = prequantized.view(
            batch_size,
            num_blocks,
            self.config.num_tokens,
            self.config.fsq_dim,
        )
        prequantized = prequantized * block_mask[:, :, None, None].to(prequantized.dtype)

        return prequantized, frame_mask, block_mask

    def encode_tokens(
        self,
        key_press: torch.Tensor,
        mouse_movement: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prequantized, _frame_mask, _block_mask = self._encode_prequantized(
            key_press=key_press,
            mouse_movement=mouse_movement,
            lengths=lengths,
        )
        token_ids, _quantized_codes = self._quantize(prequantized)
        return token_ids

    def _quantize(self, prequantized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_blocks, num_tokens, fsq_dim = prequantized.shape
        flat = prequantized.reshape(
            batch_size * num_blocks,
            num_tokens,
            fsq_dim,
        )
        quantized_codes, token_ids = self.fsq(flat)
        quantized_codes = quantized_codes.view(
            batch_size,
            num_blocks,
            num_tokens,
            fsq_dim,
        ).to(dtype=prequantized.dtype)
        token_ids = token_ids.view(
            batch_size,
            num_blocks,
            num_tokens,
        )
        return token_ids.long(), quantized_codes

    def decode_tokens(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ids.ndim != 3:
            raise ValueError(f"Expected ids shape [B, N, num_tokens], got {tuple(ids.shape)}.")

        batch_size, num_blocks, num_tokens = ids.shape
        if num_tokens != self.config.num_tokens:
            raise ValueError(f"Expected {self.config.num_tokens} tokens, got {num_tokens}.")

        flat_ids = ids.reshape(
            batch_size * num_blocks,
            num_tokens,
        )
        codes = self.fsq.indices_to_codes(flat_ids)
        codes = codes.view(
            batch_size,
            num_blocks,
            num_tokens,
            self.config.fsq_dim,
        ).to(dtype=self.frame_pos.dtype)
        return self.decode_codes(codes)

    def _run_decoder(
        self,
        layers: nn.ModuleList,
        queries: torch.Tensor,
        positions: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        frames = (queries + positions).expand(memory.shape[0], -1, -1)
        for layer in layers:
            frames = layer(query=frames, memory=memory)
        return frames

    def decode_codes(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config

        if codes.ndim != 4:
            raise ValueError(
                "Expected codes shape [B, N, num_tokens, fsq_dim], "
                f"got {tuple(codes.shape)}."
            )

        batch_size, num_blocks, num_tokens, fsq_dim = codes.shape
        if num_tokens != cfg.num_tokens or fsq_dim != cfg.fsq_dim:
            raise ValueError(
                f"Expected last dims ({cfg.num_tokens}, {cfg.fsq_dim}), "
                f"got ({num_tokens}, {fsq_dim})."
            )

        memory = self.from_fsq(codes)
        memory = memory + self.token_pos[:, :num_tokens]
        memory = memory.reshape(batch_size * num_blocks, cfg.num_tokens, cfg.model_dim)

        if self.use_semantic_split:
            key_frames = self._run_decoder(
                self.key_decoder,
                self.key_frame_queries,
                self.key_frame_pos,
                memory[:, self.key_group_indices],
            )
            mouse_frames = self._run_decoder(
                self.mouse_decoder,
                self.mouse_frame_queries,
                self.mouse_frame_pos,
                memory[:, self.mouse_group_indices],
            )
        else:
            key_frames = self._run_decoder(
                self.decoder,
                self.decode_frame_queries,
                self.decode_frame_pos,
                memory,
            )
            mouse_frames = key_frames

        key_frames = key_frames.view(batch_size, num_blocks, cfg.block_size, cfg.model_dim)
        mouse_frames = mouse_frames.view(batch_size, num_blocks, cfg.block_size, cfg.model_dim)

        key_logits = self.key_head(key_frames)
        mouse_logits = self.mouse_head(mouse_frames).view(
            batch_size,
            num_blocks,
            cfg.block_size,
            2,
            self.mouse_num_classes,
        )
        mouse_pred = self.decode_mouse(mouse_logits)

        return key_logits, mouse_logits, mouse_pred

    def decode_mouse_argmax(self, mouse_logits: torch.Tensor) -> torch.Tensor:
        return self.mouse_bins_to_values(mouse_logits.argmax(dim=-1))

    def decode_mouse(self, mouse_logits: torch.Tensor) -> torch.Tensor:
        """Point estimate in pixels, rounded back to the integer grid the targets
        live on. The unrounded expectation is what the loss uses."""
        if self.config.mouse_decode == "argmax":
            return self.decode_mouse_argmax(mouse_logits)
        return self.mouse_expectation(mouse_logits).round()

    def forward(
        self,
        key_press: torch.Tensor,
        mouse_movement: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> KeyboardMouseTokenizerOutput:
        prequantized, frame_mask, block_mask = self._encode_prequantized(
            key_press=key_press,
            mouse_movement=mouse_movement,
            lengths=lengths,
        )
        token_ids, quantized_codes = self._quantize(prequantized)
        key_logits, mouse_logits, mouse_pred = self.decode_codes(quantized_codes)

        return KeyboardMouseTokenizerOutput(
            token_ids=token_ids,
            key_logits=key_logits,
            mouse_logits=mouse_logits,
            mouse_pred=mouse_pred,
            frame_mask=frame_mask,
            block_mask=block_mask,
            prequantized=prequantized,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: Optional[torch.device] = None,
    config: Optional[KeyboardMouseTokenizerConfig] = None,
) -> KeyboardMouseTokenizer:
    """Build the frozen tokenizer and load its weights from a LOCAL checkpoint file, sha256-
    verified. No huggingface_hub: compute nodes may lack network, and the upstream HF repo is
    gated anyway -- the checkpoint is staged locally instead (plaicraft-debug#80)."""
    checkpoint_path = Path(checkpoint_path)
    digest = _sha256(checkpoint_path)
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"km tokenizer checkpoint {checkpoint_path} has sha256 {digest}, "
            f"expected {CHECKPOINT_SHA256}. Refusing to load a corpus-defining "
            "tokenizer from an unverified file."
        )

    model = KeyboardMouseTokenizer(config or KeyboardMouseTokenizerConfig())
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = state_dict.get("state_dict", state_dict)
    prefix = "model."
    if all(key.startswith(prefix) for key in state_dict):
        state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model if device is None else model.to(device)
