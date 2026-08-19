"""
Originally inspired by impl at https://github.com/facebookresearch/DiT/blob/main/models.py

Modified by Haoyu Lu, for video diffusion transformer
"""
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
#
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange, reduce, repeat


def _expand_cond(v, N):
    """Broadcast a conditioning tensor to (B, T*N, D). v is (B,D) [one vector per video]
    or (B,T,D) [one vector per frame]."""
    if v.ndim == 2:
        return v.unsqueeze(1)
    return v.repeat_interleave(N, dim=1)


def modulate(x, shift, scale, T):

    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
    if scale.ndim == 3:
        shift_e = _expand_cond(shift, N)
        scale_e = _expand_cond(scale, N)
        x = x * (1 + scale_e) + shift_e
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
    return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class ActionEmbedder(nn.Module):
    """
    Embeds per-frame action vectors into per-frame conditioning vectors. Also handles
    action dropout (per whole sample) for classifier-free guidance.
    """
    def __init__(self, action_dim, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.null_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.dropout_prob = dropout_prob

    def forward(self, actions, train, force_drop=None):
        # actions: (B, T, action_dim) -> (B, T, hidden_size)
        embeddings = self.net(actions)
        B = actions.shape[0]

        if force_drop is not None:
            if isinstance(force_drop, bool):
                drop_ids = torch.full((B,), force_drop, dtype=torch.bool, device=actions.device)
            else:
                drop_ids = force_drop.bool().to(actions.device)
        elif train and self.dropout_prob > 0:
            drop_ids = torch.rand(B, device=actions.device) < self.dropout_prob
        else:
            # All-False rather than None so null_embedding STAYS IN THE GRAPH.
            # Skipping the torch.where when dropout_prob==0 leaves it unused, and
            # DDP rejects a parameter that never contributes to the loss
            # ("Expected to have finished reduction in the prior iteration").
            # torch.where keeps it connected at zero gradient, so it simply stays
            # at its init value -- which is what an unused null embedding should
            # do. Cheaper than find_unused_parameters=True on every step.
            drop_ids = torch.zeros(B, dtype=torch.bool, device=actions.device)

        null = self.null_embedding.to(embeddings.dtype).expand_as(embeddings)
        embeddings = torch.where(drop_ids.view(B, 1, 1), null, embeddings)

        return embeddings


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

#################################################################################
#                                 Core VDT Model                                #
#################################################################################

class VDTBlock(nn.Module):
    """
    A VDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, mode='video', num_frames=16, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

        self.mode = mode

        ## Temporal Attention Parameters
        if self.mode == 'video':
            
            self.temporal_norm1 = nn.LayerNorm(hidden_size)
            self.temporal_attn = Attention(
              hidden_size, num_heads=num_heads, qkv_bias=True)
            self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        T = self.num_frames
        K, N, M = x.shape
        B = K // T
        if self.mode == 'video':
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_attn(self.temporal_norm1(x))
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_fc(res_temporal)
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            x = x + res_temporal

        attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, self.num_frames))
        attn = rearrange(attn, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        attn = _expand_cond(gate_msa, N) * attn
        attn = rearrange(attn, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + attn

        mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, self.num_frames))
        mlp = rearrange(mlp, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        mlp = _expand_cond(gate_mlp, N) * mlp
        mlp = rearrange(mlp, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + mlp

        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FinalLayer(nn.Module):
    """
    The final layer of VDT for predicting video.
    """
    def __init__(self, hidden_size, patch_size, out_channels, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x


class ActionHead(nn.Module):
    """
    The output head of VDT for predicting actions.
    """
    def __init__(self, hidden_size, action_dim, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x


class VDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mode='video',
        num_frames=16,
        action_dim=0,
        action_dropout_prob=0.0,
        generate_actions=False,
        action_token_cond=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.generate_actions = generate_actions
        # Token conditioning WITHOUT generation: the action rides in the
        # sequence like a patch, but it is never noised and carries no loss.
        # Separates "an action token in the sequence" from "an action the model
        # must denoise" -- #69 changed both at once and could not tell which of
        # the two cost the video quality.
        self.action_token_cond = bool(action_token_cond) and not generate_actions

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if action_dim > 0:
            if generate_actions or self.action_token_cond:
                self.action_x_embedder = nn.Linear(action_dim, hidden_size, bias=True)
                self.action_pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
                # No head in token-cond mode: there is nothing to read back out.
                self.action_head = (ActionHead(hidden_size, action_dim, num_frames)
                                    if generate_actions else None)
                self.action_embedder = None
            else:
                self.action_x_embedder = None
                self.action_pos_embed = None
                self.action_head = None
                self.action_embedder = ActionEmbedder(action_dim, hidden_size, action_dropout_prob)
        else:
            self.action_x_embedder = None
            self.action_pos_embed = None
            self.action_head = None
            self.action_embedder = None

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.mode = mode
        if self.mode == 'video':
            self.num_frames = num_frames
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
            self.time_drop = nn.Dropout(p=0)
        else:
            self.num_frames = 1

        self.blocks = nn.ModuleList([
            VDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, mode=mode, num_frames=self.num_frames) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.mode == 'video':
            grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
            time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.action_x_embedder is not None:
            nn.init.xavier_uniform_(self.action_x_embedder.weight)
            nn.init.constant_(self.action_x_embedder.bias, 0)

        if self.action_pos_embed is not None:
            nn.init.normal_(self.action_pos_embed, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in VDT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.action_head is not None:
            nn.init.constant_(self.action_head.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.action_head.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.action_head.linear.weight, 0)
            nn.init.constant_(self.action_head.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = self.x_embedder.grid_size[0]
        w = self.x_embedder.grid_size[1]
        p_h = self.x_embedder.patch_size[0]
        p_w = self.x_embedder.patch_size[1]

        x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p_h, w * p_w))
        return imgs

    def forward(self, x, timesteps=None, *, x0=None, frame_indices=None,
                obs_mask=None, latent_mask=None, return_attn_weights=False,
                actions=None, actions0=None, obs_action_mask=None, latent_action_mask=None,
                force_action_drop=None, **kwargs):
        """
        Forward pass of VDT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timesteps: (B,) tensor of diffusion timesteps
        """
        if timesteps is None and 'timesteps' in kwargs:
            timesteps = kwargs['timesteps']

        B, T, C, H, W = x.shape
        if obs_mask is not None and x0 is not None:
            x = x * (1 - obs_mask) + x0 * obs_mask

        x = x.contiguous().view(-1, C, H, W)
        y = torch.zeros(B, dtype=torch.long, device=x.device)
        patch_tokens = self.x_embedder(x) + self.pos_embed  # (B*T, N, D), where N = (H*W) / patch_size ** 2
        N = patch_tokens.shape[1]

        # True whenever the action rides in the sequence, whether or not it is
        # also being denoised. In token-cond mode the caller passes no
        # obs_action_mask/actions0, so `actions` falls through clean.
        use_action_tokens = (self.action_dim > 0 and self.action_x_embedder is not None
                             and actions is not None)
        if use_action_tokens:
            if obs_action_mask is not None and actions0 is not None:
                actions = actions * (1 - obs_action_mask) + actions0 * obs_action_mask
            action_tokens = self.action_x_embedder(actions)  # (B, T, D)
            action_tokens = rearrange(action_tokens, 'b t d -> (b t) 1 d') + self.action_pos_embed  # (B*T, 1, D)
            tokens = torch.cat([patch_tokens, action_tokens], dim=1)  # (B*T, N+1, D)
        else:
            tokens = patch_tokens

        if self.mode == 'video':
            # Temporal embed across T frames: (B*(N+1), T, D)
            tokens = rearrange(tokens, '(b t) n m -> (b n) t m', b=B, t=T)
            tokens = tokens + self.time_embed
            tokens = self.time_drop(tokens)
            tokens = rearrange(tokens, '(b n) t m -> (b t) n m', b=B, t=T)

        t = self.t_embedder(timesteps)           # (B, D)
        y = self.y_embedder(y, self.training)    # (B, D)

        if not self.generate_actions and actions is not None and self.action_embedder is not None:
            # `y` is kept here even though num_classes=0 makes it a learned
            # constant: dropping it leaves y_embedder unreachable by backward,
            # and an orphaned parameter with p.grad None crashed _log_grad_norm
            # on the first optimizer step. It is expressively free -- the action
            # embedder has its own bias -- so this costs nothing.
            c = t.unsqueeze(1) + y.unsqueeze(1) + \
                self.action_embedder(actions, self.training, force_action_drop)  # (B, T, D)
        else:
            c = t + y                         # (B, D)

        for block in self.blocks:
            tokens = block(tokens, c)            # (B*T, N+1, D) or (B*T, N, D)

        video_tokens = tokens[:, :N, :]
        x_out = self.final_layer(video_tokens, c) # (B*T, N, patch_size ** 2 * out_channels)
        x_out = self.unpatchify(x_out)            # (B*T, out_channels, H, W)
        x_out = x_out.view(B, T, x_out.shape[-3], x_out.shape[-2], x_out.shape[-1])

        if use_action_tokens and self.action_head is not None:
            action_tokens_out = tokens[:, N:, :]  # (B*T, 1, D)
            act_out = self.action_head(action_tokens_out, c)  # (B*T, 1, action_dim)
            act_out = act_out.view(B, T, self.action_dim)
            return x_out, act_out

        return x_out, None

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of VDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out, _ = self.forward(combined, timesteps=t)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: grid_size tuple (height, width)
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    h, w = grid_size

    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   VDT Configs                                  #
#################################################################################

def VDT_L_2(**kwargs):
    return VDT(depth=28, hidden_size=1152, num_heads=16, num_classes=0, **kwargs)

def VDT_M_2(**kwargs):
    return VDT(depth=12, hidden_size=1024, num_heads=16, num_classes=0, **kwargs)

def VDT_SM_2(**kwargs):
    return VDT(depth=12, hidden_size=640, num_heads=10, num_classes=0, **kwargs)

def VDT_S_2(**kwargs):
    return VDT(depth=12, hidden_size=384, num_heads=6, num_classes=1000, **kwargs)


VDT_models = {
    'VDT-L/2':  VDT_L_2,
    'VDT-M/2':  VDT_M_2,
    'VDT-S/2':  VDT_S_2,
    'VDT-SM/2': VDT_SM_2,
}
