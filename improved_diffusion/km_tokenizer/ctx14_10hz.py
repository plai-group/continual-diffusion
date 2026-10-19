"""Vendored, verbatim, from CharlieH512/km_tokenizer_ctx14_10hz_8x1000's
km_tokenizer_ctx14_10hz.py (public HF repo, staged at models/km_ctx14_10hz/,
plaicraft-debug#96). The only addition is load_ctx14_tokenizer at the bottom,
a local-checkpoint-dir loader mirroring km_tokenizer/model.py's load_tokenizer
(compute nodes may lack network at runtime). Key order: the repo's README
lists the same order as keypress_lookup.KEYPRESS_LABELS -- feed
build_action_array_79's raw79 output straight in, no scatter.

Keyboard/mouse tokenizer ctx14 at 10 Hz (8 x 1000).

100 Hz key and mouse frames -> one row of `num_tokens` discrete codes per 100 ms tick
(10 frames), and back. A tick's 10 frames go through attention over those frames, are
concatenated into one tick vector, pass attention across ticks within +-14 ticks (1.4 s,
the same for every tick), and a linear map gives one latent vector per tick. That vector is cut into `num_tokens` fixed slices; each
slice is quantized on its own by rounding every dimension to a fixed number of levels.
Codes of different slices never attend to each other: the decoder reads the quantized
slices concatenated, as plain input channels, applies causal attention across ticks,
expands each tick to its 10 frame slots, applies attention over those slots, and emits
per frame 79 key logits and, per mouse axis, logits over 241 movement bins.

Input:  key_press [B, T, 79], mouse_movement [B, T, 2] at 100 Hz.
Tokens: [B, N, num_tokens], N = ceil(T / block_size), one row per tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Rate of the raw key/mouse frames this tokenizer consumes.
CONTROL_FPS = 100.0

KM_TOKENIZER_CTX14_10HZ_REPO_ID = "CharlieH512/km_tokenizer_ctx14_10hz_8x1000"
_WEIGHTS_FILENAME = "model.safetensors"
_CONFIG_FILENAME = "config.json"


@dataclass
class KmTokenizerCtx14_10HzConfig:
    num_keys: int = 79
    block_size: int = 10                 # frames per tick: 10 x 10 ms = 100 ms

    key_feature_dim: int = 237           # 79 held + 79 pressed-this-frame + 79 released-this-frame
    mouse_feature_dim: int = 7

    model_dim: int = 256
    num_heads: int = 8
    ffn_dim: int = 1024
    frame_layers: int = 2                # attention over a tick's 10 frames
    tick_layers: int = 4                 # attention across ticks: bidirectional to encode, causal to decode
    # Per encoder tick layer, how many ticks each side a tick attends to; the reach adds up over
    # layers, so every tick's code sees exactly +-14 ticks (1.4 s).
    encoder_tick_radius: tuple[int, ...] = (4, 4, 3, 3)

    num_tokens: int = 8                  # codebook groups per tick
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)
    vocab_size: int = 1000               # prod(fsq_levels): codes per group

    mouse_max_abs_dx: int = 1280         # per-frame clamp, px
    mouse_max_abs_dy: int = 720
    mouse_norm_scale: float = 50.0
    mouse_linear_bins: int = 96          # unit-width bins over +-96 px, then a geometric tail
    mouse_tail_bins: int = 24

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)


@dataclass
class KmTokenizerCtx14_10HzOutput:
    token_ids: torch.Tensor      # [B, N, G] long
    key_logits: torch.Tensor     # [B, N, bs, 79]
    mouse_logits: torch.Tensor   # [B, N, bs, 2, bins]
    mouse_pred: torch.Tensor     # [B, N, bs, 2] whole pixels
    frame_mask: torch.Tensor     # [B, N, bs]
    block_mask: torch.Tensor     # [B, N]
    prequantized: torch.Tensor   # [B, N, G, fsq_dim]


# --------------------------------------------------------------------- quantizer


class FSQ(nn.Module):
    """Per-dimension scalar quantization: tanh-bounded, rounded to `levels[i]` values
    (straight-through gradient), and packed into one id per group by mixed radix."""

    def __init__(self, levels: list[int]) -> None:
        super().__init__()
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32), persistent=False)
        self.register_buffer("_basis", torch.cumprod(torch.tensor([1, *levels[:-1]], dtype=torch.int32), dim=0),
                             persistent=False)

    def _bound(self, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        levels = self._levels.to(device=x.device, dtype=x.dtype)
        half_l = (levels - 1.0) * (1.0 + eps) / 2.0
        offset = torch.where(levels.remainder(2) == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels))
        return torch.tanh(x + torch.atanh(offset / half_l)) * half_l - offset

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        half = (self._levels // 2).to(device=codes.device, dtype=codes.dtype)
        idx = (codes * half + half).round().long()
        idx = torch.minimum(idx.clamp_min(0), self._levels.to(codes.device).long() - 1)
        return (idx * self._basis.to(codes.device).long()).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        idx = torch.div(indices.long()[..., None], self._basis.to(indices.device).long(), rounding_mode="floor")
        idx = idx.remainder(self._levels.to(indices.device).long()).float()
        half = (self._levels // 2).to(device=indices.device, dtype=torch.float32)
        return (idx - half) / half

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        bounded = self._bound(x.float())
        quantized = bounded + (bounded.round() - bounded).detach()
        codes = quantized / (self._levels // 2).to(device=x.device, dtype=torch.float32)
        return codes.to(dtype), self.codes_to_indices(codes)


def build_bin_centers(max_abs: float, linear_bins: int, tail_bins: int) -> torch.Tensor:
    """Integer bins over [-linear_bins, linear_bins], then a geometric tail out to
    +-max_abs with integer centers, strictly increasing."""
    core = [float(v) for v in range(-linear_bins, linear_bins + 1)]
    tail: list[float] = []
    ratio, previous = (max_abs / linear_bins) ** (1.0 / tail_bins), float(linear_bins)
    for step in range(1, tail_bins + 1):
        value = min(max(round(linear_bins * ratio ** step), previous + 1.0), max_abs)
        if value > previous:
            tail.append(value)
            previous = value
    return torch.tensor([-v for v in reversed(tail)] + core + tail, dtype=torch.float32)


# -------------------------------------------------------------------- attention


def _rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position over the sequence axis of [B, H, L, D]."""
    L, D = x.shape[-2], x.shape[-1]
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    ang = torch.arange(L, device=x.device, dtype=torch.float32)[:, None] * freqs[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    """Pre-norm transformer block; rotary positions on the ticks axis only."""

    def __init__(self, d: int, heads: int, ffn: int, rope: bool) -> None:
        super().__init__()
        self.heads, self.rope = heads, rope
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, causal: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = _rope(q), _rope(k)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.norm2(x))


def _tick_mask(lengths: torch.Tensor, ticks: int, block_size: int) -> torch.Tensor:
    """[B, ticks]: ticks holding at least one valid frame."""
    valid = torch.div(lengths + block_size - 1, block_size, rounding_mode="floor")
    return torch.arange(ticks, device=lengths.device)[None, :] < valid[:, None]


class FrameTickEncoder(nn.Module):
    """Frames [B, C, T] -> per-tick latent [B, G * fsq_dim, N]."""

    def __init__(self, cfg: KmTokenizerCtx14_10HzConfig, in_channels: int, encoded_dim: int) -> None:
        super().__init__()
        d, bs = int(cfg.model_dim), int(cfg.block_size)
        self.bs = bs
        self.frame_in = nn.Linear(in_channels, d)
        self.frame_pos = nn.Parameter(torch.randn(1, bs, d) * 0.02)
        self.frame_layers = nn.ModuleList(Block(d, cfg.num_heads, cfg.ffn_dim, False) for _ in range(cfg.frame_layers))
        self.to_tick = nn.Linear(bs * d, d)
        self.tick_layers = nn.ModuleList(Block(d, cfg.num_heads, cfg.ffn_dim, True) for _ in range(cfg.tick_layers))
        self.radius = tuple(int(r) for r in cfg.encoder_tick_radius)
        if len(self.radius) != int(cfg.tick_layers) or min(self.radius) < 0:
            raise ValueError(f"encoder_tick_radius needs one radius >= 0 per tick layer ({cfg.tick_layers}), got {self.radius}.")
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, encoded_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, _, T = x.shape
        N = T // self.bs
        h = self.frame_in(x.transpose(1, 2)).view(B * N, self.bs, -1) + self.frame_pos
        for layer in self.frame_layers:
            h = layer(h)
        h = self.to_tick(h.reshape(B, N, -1))
        mask = _tick_mask(lengths, N, self.bs)[:, None, None, :]         # attend to valid ticks only
        idx = torch.arange(N, device=h.device)
        offset = (idx[None, :] - idx[:, None]).abs()
        # A padded tick keeps itself as a key, so no query row is ever fully masked.
        keys = mask | torch.eye(N, dtype=torch.bool, device=h.device)[None, None]
        for layer, r in zip(self.tick_layers, self.radius):
            h = layer(h, attn_mask=keys & (offset <= r)[None, None])
        return self.out(self.norm(h)).transpose(1, 2)


class FrameTickDecoder(nn.Module):
    """Per-tick latent [B, G * fsq_dim, N] -> frame outputs [B, C, N * bs]; causal across ticks."""

    def __init__(self, cfg: KmTokenizerCtx14_10HzConfig, input_dim: int, out_channels: int) -> None:
        super().__init__()
        d, bs = int(cfg.model_dim), int(cfg.block_size)
        self.bs, self.d = bs, d
        self.tick_in = nn.Linear(input_dim, d)
        self.tick_layers = nn.ModuleList(Block(d, cfg.num_heads, cfg.ffn_dim, True) for _ in range(cfg.tick_layers))
        self.to_frames = nn.Linear(d, bs * d)
        self.frame_pos = nn.Parameter(torch.randn(1, bs, d) * 0.02)
        self.frame_layers = nn.ModuleList(Block(d, cfg.num_heads, cfg.ffn_dim, False) for _ in range(cfg.frame_layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, out_channels)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, _, N = x.shape
        h = self.tick_in(x.transpose(1, 2))
        for layer in self.tick_layers:
            h = layer(h, causal=True)
        h = self.to_frames(h).view(B * N, self.bs, self.d) + self.frame_pos
        for layer in self.frame_layers:
            h = layer(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            out = self.out(self.norm(h.float())).view(B, N * self.bs, -1).transpose(1, 2)
        valid = torch.arange(N * self.bs, device=x.device)[None, :] < lengths[:, None]
        return out * valid[:, None, :].to(out.dtype)


# ------------------------------------------------------------------- tokenizer


class KmTokenizerCtx14_10Hz(nn.Module):
    def __init__(self, config: KmTokenizerCtx14_10HzConfig, repo_id: Optional[str] = KM_TOKENIZER_CTX14_10HZ_REPO_ID,
                 local_files_only: bool = True) -> None:
        super().__init__()
        self.config = config
        self.repo_id = repo_id
        self.local_files_only = bool(local_files_only)
        self._weights_loaded = False

        levels = [int(level) for level in config.fsq_levels]
        if math.prod(levels) != int(config.vocab_size):
            raise ValueError(f"vocab_size must be prod(fsq_levels) = {math.prod(levels)}, got {config.vocab_size}.")
        centers = torch.stack([
            build_bin_centers(config.mouse_max_abs_dx, config.mouse_linear_bins, config.mouse_tail_bins),
            build_bin_centers(config.mouse_max_abs_dy, config.mouse_linear_bins, config.mouse_tail_bins),
        ])
        self.register_buffer("mouse_bin_centers", centers, persistent=False)
        self.register_buffer("mouse_bin_edges", (centers[:, 1:] + centers[:, :-1]) * 0.5, persistent=False)
        self.register_buffer("mouse_axis_max", torch.tensor([float(config.mouse_max_abs_dx), float(config.mouse_max_abs_dy)]),
                             persistent=False)
        self.mouse_num_classes = int(centers.shape[1])

        self.fsq = FSQ(levels)
        latent_dim = int(config.num_tokens) * len(levels)
        self.encoder = FrameTickEncoder(config, config.key_feature_dim + config.mouse_feature_dim, latent_dim)
        self.decoder = FrameTickDecoder(config, latent_dim, config.num_keys + 2 * self.mouse_num_classes)

    # -------------------------------------------------------------- features and bins

    def make_key_features(self, key_press: torch.Tensor) -> torch.Tensor:
        prev = F.pad(key_press[:, :-1], (0, 0, 1, 0), value=0.0)
        return torch.cat([key_press, (key_press > prev).to(key_press.dtype), (key_press < prev).to(key_press.dtype)], dim=-1)

    def clamp_to_axis_bounds(self, mouse_movement: torch.Tensor) -> torch.Tensor:
        bounds = self.mouse_axis_max.to(device=mouse_movement.device, dtype=mouse_movement.dtype)
        return torch.maximum(torch.minimum(mouse_movement, bounds), -bounds)

    def make_mouse_features(self, mouse_movement: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        m = self.clamp_to_axis_bounds(mouse_movement)
        dx, dy = m[..., 0], m[..., 1]
        magnitude = torch.log1p(torch.sqrt(dx.square() + dy.square())) / math.log1p(
            math.hypot(float(cfg.mouse_max_abs_dx), float(cfg.mouse_max_abs_dy)))
        return torch.stack([
            (dx / float(cfg.mouse_norm_scale)).clamp(-4.0, 4.0),
            (dy / float(cfg.mouse_norm_scale)).clamp(-4.0, 4.0),
            torch.log1p(dx.abs()) / math.log1p(float(cfg.mouse_max_abs_dx)),
            torch.log1p(dy.abs()) / math.log1p(float(cfg.mouse_max_abs_dy)),
            torch.sign(dx), torch.sign(dy), magnitude,
        ], dim=-1)

    def mouse_values_to_bins(self, values: torch.Tensor) -> torch.Tensor:
        clamped = self.clamp_to_axis_bounds(values.float())
        edges = self.mouse_bin_edges.to(device=values.device, dtype=torch.float32)
        return torch.stack([torch.bucketize(clamped[..., a].contiguous(), edges[a]) for a in range(2)], dim=-1)

    def mouse_expectation(self, mouse_logits: torch.Tensor) -> torch.Tensor:
        """Bin logits [..., 2, bins] -> expected pixels [..., 2]."""
        centers = self.mouse_bin_centers.to(device=mouse_logits.device, dtype=torch.float32)
        return (torch.softmax(mouse_logits.float(), dim=-1) * centers).sum(dim=-1)

    # ------------------------------------------------------------------ encode

    def _pad_to_blocks(self, key_press, mouse_movement, lengths):
        if key_press.ndim != 3 or mouse_movement.ndim != 3 or key_press.shape[:2] != mouse_movement.shape[:2]:
            raise ValueError(f"expected key_press [B,T,79] and mouse_movement [B,T,2], got "
                             f"{tuple(key_press.shape)} and {tuple(mouse_movement.shape)}.")
        if key_press.shape[-1] != self.config.num_keys or mouse_movement.shape[-1] != 2:
            raise ValueError("key_press last dim must be num_keys and mouse_movement last dim 2.")
        B, T, _ = key_press.shape
        bs = int(self.config.block_size)
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=key_press.device)
        lengths = lengths.to(device=key_press.device, dtype=torch.long).clamp(0, T)
        N = math.ceil(T / bs)
        pad = N * bs - T
        if pad:
            key_press = F.pad(key_press, (0, 0, 0, pad))
            mouse_movement = F.pad(mouse_movement, (0, 0, 0, pad))
        frame_mask = (torch.arange(N * bs, device=key_press.device)[None, :] < lengths[:, None]).view(B, N, bs)
        return key_press, mouse_movement, lengths, frame_mask, frame_mask.any(dim=-1)

    def _encode_prequantized(self, key_press, mouse_movement, lengths=None):
        key_press, mouse_movement, lengths, frame_mask, block_mask = self._pad_to_blocks(
            key_press.float(), mouse_movement.float(), lengths)
        frames = torch.cat([self.make_key_features(key_press), self.make_mouse_features(mouse_movement)], dim=-1)
        frames = frames * frame_mask.reshape(frames.shape[0], -1, 1).to(frames.dtype)
        latent = self.encoder(frames.transpose(1, 2), lengths)                      # [B, G*d, N]
        B, _, N = latent.shape
        prequantized = latent.transpose(1, 2).reshape(B, N, int(self.config.num_tokens), int(self.config.fsq_dim))
        return prequantized * block_mask[:, :, None, None].to(prequantized.dtype), frame_mask, block_mask

    def _quantize(self, prequantized: torch.Tensor):
        B, N, G, d = prequantized.shape
        codes, ids = self.fsq(prequantized.reshape(B * N, G, d))
        return ids.view(B, N, G).long(), codes.view(B, N, G, d).to(prequantized.dtype)

    def encode_tokens(self, key_press, mouse_movement, lengths=None) -> torch.Tensor:
        """[B, T, 79], [B, T, 2] at 100 Hz -> ids [B, ceil(T/10), num_tokens]."""
        prequantized, _, _ = self._encode_prequantized(key_press, mouse_movement, lengths)
        return self._quantize(prequantized)[0]

    # ------------------------------------------------------------------ decode

    def decode_codes(self, codes: torch.Tensor, block_lengths: Optional[torch.Tensor] = None):
        """[B, N, G, d] quantized values -> (key_logits [B,N,bs,79], mouse_logits [B,N,bs,2,bins],
        mouse_pred [B,N,bs,2] whole pixels). Tick n reads ticks <= n only."""
        cfg = self.config
        B, N, G, d = codes.shape
        if G != int(cfg.num_tokens) or d != int(cfg.fsq_dim):
            raise ValueError(f"expected codes [B, N, {cfg.num_tokens}, {cfg.fsq_dim}], got {tuple(codes.shape)}.")
        bs = int(cfg.block_size)
        frames = (torch.full((B,), N, device=codes.device, dtype=torch.long) if block_lengths is None
                  else block_lengths.to(codes.device).long()) * bs
        out = self.decoder(codes.reshape(B, N, G * d).transpose(1, 2), frames).transpose(1, 2)   # [B, N*bs, C]
        key_logits = out[..., : cfg.num_keys].reshape(B, N, bs, cfg.num_keys)
        mouse_logits = out[..., cfg.num_keys:].reshape(B, N, bs, 2, self.mouse_num_classes)
        return key_logits, mouse_logits, self.mouse_expectation(mouse_logits).round()

    def decode_tokens(self, ids: torch.Tensor):
        """ids [B, N, num_tokens] -> (key_logits, mouse_logits, mouse_pred) as `decode_codes`."""
        if ids.ndim != 3 or ids.shape[-1] != int(self.config.num_tokens):
            raise ValueError(f"expected ids [B, N, {self.config.num_tokens}], got {tuple(ids.shape)}.")
        B, N, G = ids.shape
        codes = self.fsq.indices_to_codes(ids.reshape(B * N, G)).view(B, N, G, -1)
        return self.decode_codes(codes.to(next(self.parameters()).dtype))

    def forward(self, key_press, mouse_movement, lengths=None) -> KmTokenizerCtx14_10HzOutput:
        prequantized, frame_mask, block_mask = self._encode_prequantized(key_press, mouse_movement, lengths)
        token_ids, codes = self._quantize(prequantized)
        key_logits, mouse_logits, mouse_pred = self.decode_codes(codes, block_mask.sum(-1))
        return KmTokenizerCtx14_10HzOutput(
            token_ids=token_ids, key_logits=key_logits, mouse_logits=mouse_logits, mouse_pred=mouse_pred,
            frame_mask=frame_mask, block_mask=block_mask, prequantized=prequantized,
        )

    # ------------------------------------------------------------- loading

    @classmethod
    def from_pretrained(cls, repo_id: str = KM_TOKENIZER_CTX14_10HZ_REPO_ID, device="cpu", **hub_kwargs) -> "KmTokenizerCtx14_10Hz":
        """Build from the repo's config.json and load its weights; returns the model in eval mode."""
        import json
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg = json.loads(Path(hf_hub_download(repo_id, _CONFIG_FILENAME, **hub_kwargs)).read_text())
        cfg["fsq_levels"] = tuple(cfg["fsq_levels"])
        cfg["encoder_tick_radius"] = tuple(cfg["encoder_tick_radius"])
        model = cls(KmTokenizerCtx14_10HzConfig(**cfg), repo_id=repo_id)
        model.load_state_dict(load_file(hf_hub_download(repo_id, _WEIGHTS_FILENAME, **hub_kwargs)), strict=True)
        model._weights_loaded = True
        return model.to(device).eval()


DEFAULT_CHECKPOINT_DIR = Path("/ubc/cs/research/plai-scratch/ctardy/models/km_ctx14_10hz")
CHECKPOINT_SHA256 = "4d91000d2d34ace0f995b4e483c2a07e51b10778cd8e6b0f5a4b3db40af75925"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ctx14_tokenizer(checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR, device=None) -> "KmTokenizerCtx14_10Hz":
    """Build the frozen tokenizer from a LOCAL checkpoint dir (config.json + model.safetensors),
    weights sha256-verified -- no huggingface_hub at load time, same reasoning as
    km_tokenizer/model.py's load_tokenizer (compute nodes may lack network; plaicraft-debug#96)."""
    import json

    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    weights_path = checkpoint_dir / _WEIGHTS_FILENAME
    digest = _sha256(weights_path)
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"km_ctx14 tokenizer weights {weights_path} have sha256 {digest}, expected "
            f"{CHECKPOINT_SHA256}. Refusing to load a corpus-defining tokenizer from an unverified file."
        )
    cfg = json.loads((checkpoint_dir / _CONFIG_FILENAME).read_text())
    cfg["fsq_levels"] = tuple(cfg["fsq_levels"])
    cfg["encoder_tick_radius"] = tuple(cfg["encoder_tick_radius"])
    model = KmTokenizerCtx14_10Hz(KmTokenizerCtx14_10HzConfig(**cfg), repo_id=None, local_files_only=True)
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    model._weights_loaded = True
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model if device is None else model.to(device)


__all__ = [
    "KmTokenizerCtx14_10Hz", "KmTokenizerCtx14_10HzConfig", "KmTokenizerCtx14_10HzOutput",
    "KM_TOKENIZER_CTX14_10HZ_REPO_ID", "CONTROL_FPS", "DEFAULT_CHECKPOINT_DIR", "CHECKPOINT_SHA256",
    "load_ctx14_tokenizer",
]
