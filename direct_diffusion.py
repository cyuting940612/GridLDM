"""Direct cross-attention diffusion training for multi-domain 1D time series.

This module trains a DDPM-style diffusion model directly in raw time-series space,
conditioned on precomputed text-token embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainVariant:
    """One dataset/text-embedding pair used as a training source.

    Attributes:
        name: Human-readable variant name.
        data_path: Path to the raw training array ``(N, T_raw)``.
        text_path: Path to the aligned text embedding array ``(N, L, 768)``.
        true_length: Valid sequence length before right padding.
        pad_to: Final padded length used by the diffusion model.
    """

    name: str
    data_path: str
    text_path: str
    true_length: int
    pad_to: int = 96


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level training configuration."""

    batch_size: int = 256
    num_epochs: int = 1000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    p_uncond: float = 0.1
    null_fill_value: float = 0.0
    random_seed: int = 42
    checkpoint_path: str = "direct_diffusion.pt"


@dataclass(frozen=True)
class UNet1DConfig:
    """Architecture parameters for the 1D U-Net denoiser."""

    in_channels: int = 1
    base_channels: int = 128
    channel_mults: Tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    num_heads: int = 4
    t_emb_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class DiffusionConfig:
    """Noise schedule and objective for DDPM training."""

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    objective: str = "eps"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def select_device() -> torch.device:
    """Return the best available torch device.

    Preference order is CUDA, then Apple MPS, then CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set NumPy and PyTorch seeds for reproducible training."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create standard sinusoidal embeddings for diffusion timesteps.

    Args:
        timesteps: Tensor of shape ``(B,)`` containing diffusion step indices.
        dim: Output embedding dimension.

    Returns:
        Tensor of shape ``(B, dim)``.
    """

    half_dim = dim // 2
    device = timesteps.device
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=device).float() / half_dim
    )
    t = timesteps.float()
    args = t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class GELU(nn.Module):
    """Thin wrapper to keep activation style explicit in module definitions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


# -----------------------------------------------------------------------------
# Attention and convolution blocks
# -----------------------------------------------------------------------------


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the temporal dimension.

    Input and output shapes are both ``(B, C, T)``.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, steps = x.shape
        h = self.norm(x)
        q, k, v = torch.chunk(self.qkv(h), 3, dim=1)

        q = q.view(batch_size, self.num_heads, self.head_dim, steps)
        k = k.view(batch_size, self.num_heads, self.head_dim, steps)
        v = v.view(batch_size, self.num_heads, self.head_dim, steps)

        scale = 1.0 / math.sqrt(self.head_dim)
        attention = torch.einsum("b h d t, b h d s -> b h t s", q, k) * scale
        attention = attention.softmax(dim=-1)
        out = torch.einsum("b h t s, b h d s -> b h d t", attention, v)
        out = out.reshape(batch_size, channels, steps)
        return x + self.proj(out)


class CrossAttention1D(nn.Module):
    """Cross-attention from time-series features to text conditioning tokens.

    Args:
        x_channels: Channel dimension of time-series features.
        cond_dim: Feature dimension of the conditioning tokens.
        num_heads: Number of attention heads.

    Inputs:
        x: ``(B, C, T)``
        cond_tokens: ``(B, M, D)``

    Returns:
        Tensor of shape ``(B, C, T)``.
    """

    def __init__(self, x_channels: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        if x_channels % num_heads != 0:
            raise ValueError("x_channels must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = x_channels // num_heads
        self.norm_x = nn.GroupNorm(8, x_channels)
        self.norm_cond = nn.LayerNorm(cond_dim)
        self.to_q = nn.Conv1d(x_channels, x_channels, kernel_size=1)
        self.to_k = nn.Linear(cond_dim, x_channels)
        self.to_v = nn.Linear(cond_dim, x_channels)
        self.proj = nn.Conv1d(x_channels, x_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, channels, steps = x.shape
        _, num_tokens, _ = cond_tokens.shape

        q = self.to_q(self.norm_x(x)).view(
            batch_size, self.num_heads, self.head_dim, steps
        )

        cond_norm = self.norm_cond(cond_tokens)
        k = self.to_k(cond_norm).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v = self.to_v(cond_norm).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attention = torch.einsum("b h d t, b h d m -> b h t m", q, k) * scale
        attention = attention.softmax(dim=-1)
        out = torch.einsum("b h t m, b h d m -> b h d t", attention, v)
        out = out.reshape(batch_size, channels, steps)
        return self.proj(out)


class ResBlock(nn.Module):
    """Residual 1D convolution block with timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act = GELU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(t_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return self.skip(x) + h


class Downsample(nn.Module):
    """Strided temporal downsampling block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Transposed-convolution temporal upsampling block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -----------------------------------------------------------------------------
# Denoiser backbone
# -----------------------------------------------------------------------------


class UNet1D(nn.Module):
    """1D U-Net denoiser with timestep conditioning and cross-attention."""

    def __init__(
        self,
        cfg: UNet1DConfig,
        cond_dim: int,
        use_self_attn: bool = True,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.cfg = cfg

        base_channels = cfg.base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.t_emb_dim, cfg.t_emb_dim * 4),
            GELU(),
            nn.Linear(cfg.t_emb_dim * 4, cfg.t_emb_dim),
        )
        self.in_conv = nn.Conv1d(cfg.in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_channels = base_channels
        skip_channels: List[int] = []

        for stage_index, multiplier in enumerate(cfg.channel_mults):
            out_channels = base_channels * multiplier
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(
                    nn.ModuleDict(
                        {
                            "res": ResBlock(
                                in_channels,
                                out_channels,
                                cfg.t_emb_dim,
                                cfg.dropout,
                            ),
                            "self_attn": SelfAttention1D(out_channels, cfg.num_heads)
                            if use_self_attn
                            else nn.Identity(),
                            "cross_attn": CrossAttention1D(out_channels, cond_dim, cfg.num_heads)
                            if use_cross_attn
                            else nn.Identity(),
                        }
                    )
                )
                in_channels = out_channels
                skip_channels.append(out_channels)

            if stage_index != len(cfg.channel_mults) - 1:
                self.down_blocks.append(nn.ModuleDict({"down": Downsample(in_channels)}))
                skip_channels.append(in_channels)

        self.mid_block1 = ResBlock(in_channels, in_channels, cfg.t_emb_dim, cfg.dropout)
        self.mid_self = SelfAttention1D(in_channels, cfg.num_heads) if use_self_attn else nn.Identity()
        self.mid_cross = CrossAttention1D(in_channels, cond_dim, cfg.num_heads) if use_cross_attn else nn.Identity()
        self.mid_block2 = ResBlock(in_channels, in_channels, cfg.t_emb_dim, cfg.dropout)

        self.up_blocks = nn.ModuleList()
        for stage_index, multiplier in reversed(list(enumerate(cfg.channel_mults))):
            out_channels = base_channels * multiplier
            num_res_blocks_up = cfg.num_res_blocks + (1 if stage_index != 0 else 0)
            for _ in range(num_res_blocks_up):
                skip_ch = skip_channels.pop()
                self.up_blocks.append(
                    nn.ModuleDict(
                        {
                            "res": ResBlock(
                                in_channels + skip_ch,
                                out_channels,
                                cfg.t_emb_dim,
                                cfg.dropout,
                            ),
                            "self_attn": SelfAttention1D(out_channels, cfg.num_heads)
                            if use_self_attn
                            else nn.Identity(),
                            "cross_attn": CrossAttention1D(out_channels, cond_dim, cfg.num_heads)
                            if use_cross_attn
                            else nn.Identity(),
                        }
                    )
                )
                in_channels = out_channels
            if stage_index != 0:
                self.up_blocks.append(nn.ModuleDict({"up": Upsample(in_channels)}))

        self.out_norm = nn.GroupNorm(8, in_channels)
        self.out_act = GELU()
        self.out_conv = nn.Conv1d(in_channels, cfg.in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        """Predict diffusion noise.

        Args:
            x: Noisy series of shape ``(B, C, T)``.
            t: Diffusion step indices of shape ``(B,)``.
            cond_tokens: Conditioning embeddings of shape ``(B, M, D)``.

        Returns:
            Predicted noise tensor with shape ``(B, C, T)``.
        """

        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.cfg.t_emb_dim))
        h = self.in_conv(x)
        skips: List[torch.Tensor] = []

        for block in self.down_blocks:
            if "res" in block:
                h = block["res"](h, t_emb)
                h = block["self_attn"](h)
                if isinstance(block["cross_attn"], CrossAttention1D):
                    h = block["cross_attn"](h, cond_tokens)
                skips.append(h)
            else:
                h = block["down"](h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_self(h)
        h = self.mid_cross(h, cond_tokens)
        h = self.mid_block2(h, t_emb)

        for block in self.up_blocks:
            if "res" in block:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block["res"](h, t_emb)
                h = block["self_attn"](h)
                if isinstance(block["cross_attn"], CrossAttention1D):
                    h = block["cross_attn"](h, cond_tokens)
            else:
                h = block["up"](h)

        return self.out_conv(self.out_act(self.out_norm(h)))


# -----------------------------------------------------------------------------
# Diffusion wrapper
# -----------------------------------------------------------------------------


class GaussianDiffusion(nn.Module):
    """DDPM wrapper around a 1D denoising network."""

    def __init__(self, model: UNet1D, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the forward diffusion process ``q(x_t | x_0)``."""

        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self.sqrt_alphas_cumprod[t][:, None, None] * x0
            + self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise
        )

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the DDPM posterior mean and variance for one reverse step."""

        eps_pred = self.model(x_t, t, cond_tokens)
        beta_t = self.betas[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None]

        x0_pred = (x_t - sqrt_one_minus * eps_pred) / sqrt_recip_alpha

        alpha_t = self.alphas[t][:, None, None]
        alpha_bar_t = self.alphas_cumprod[t][:, None, None]
        alpha_bar_prev = torch.where(
            t[:, None, None] > 0,
            self.alphas_cumprod[(t - 1).clamp(min=0)][:, None, None],
            torch.ones_like(alpha_bar_t),
        )
        posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)
        posterior_mean = (
            beta_t * torch.sqrt(alpha_bar_prev + 1e-8) / (1 - alpha_bar_t + 1e-8) * x0_pred
            + (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t + 1e-8) * x_t
        )
        return posterior_mean, posterior_var

    def loss(
        self,
        x0: torch.Tensor,
        cond_tokens: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute epsilon-prediction loss.

        Args:
            x0: Clean series of shape ``(B, 1, T)``.
            cond_tokens: Conditioning embeddings of shape ``(B, M, D)``.
            valid_mask: Optional mask of shape ``(B, 1, T)`` with ones on valid
                positions and zeros on padded positions.
        """

        batch_size = x0.size(0)
        t = torch.randint(0, self.cfg.timesteps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, t, cond_tokens)

        if valid_mask is None:
            return F.mse_loss(eps_pred, noise)

        mse = (eps_pred - noise) ** 2
        mse = mse * valid_mask
        return mse.sum() / (valid_mask.sum() + 1e-8)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int],
        cond_tokens: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate samples without classifier-free guidance."""

        batch_size, _, _ = shape
        dev = device if device is not None else cond_tokens.device
        x = torch.randn(shape, device=dev)
        for step in reversed(range(self.cfg.timesteps)):
            t = torch.full((batch_size,), step, device=x.device, dtype=torch.long)
            mean, var = self.p_mean_variance(x, t, cond_tokens)
            noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
            x = mean + torch.sqrt(var + 1e-8) * noise
        return x

    @torch.no_grad()
    def sample_cfg(
        self,
        shape: Tuple[int, int, int],
        cond_tokens: torch.Tensor,
        uncond_tokens: torch.Tensor,
        guidance_scale: float = 4.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate samples with classifier-free guidance."""

        batch_size, _, _ = shape
        dev = device if device is not None else cond_tokens.device
        x = torch.randn(shape, device=dev)

        for step in reversed(range(self.cfg.timesteps)):
            t = torch.full((batch_size,), step, device=x.device, dtype=torch.long)
            eps_uncond = self.model(x, t, uncond_tokens)
            eps_cond = self.model(x, t, cond_tokens)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            beta_t = self.betas[t][:, None, None]
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
            sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None]
            alpha_t = self.alphas[t][:, None, None]
            alpha_bar_t = self.alphas_cumprod[t][:, None, None]
            alpha_bar_prev = torch.where(
                t[:, None, None] > 0,
                self.alphas_cumprod[(t - 1).clamp(min=0)][:, None, None],
                torch.ones_like(alpha_bar_t),
            )

            x0_pred = (x - sqrt_one_minus * eps) / sqrt_recip_alpha
            posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-8)
            posterior_mean = (
                beta_t * torch.sqrt(alpha_bar_prev + 1e-8) / (1 - alpha_bar_t + 1e-8) * x0_pred
                + (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t + 1e-8) * x
            )
            noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
            x = posterior_mean + torch.sqrt(posterior_var + 1e-8) * noise
        return x


# -----------------------------------------------------------------------------
# Dataset and preprocessing helpers
# -----------------------------------------------------------------------------


class MultiDomainTimeSeriesTextDataset(Dataset):
    """Dataset containing normalized padded series, lengths, and text embeddings."""

    def __init__(
        self,
        ts_norm: np.ndarray,
        lengths: np.ndarray,
        text_embeds: np.ndarray,
        domain_ids: np.ndarray,
    ):
        super().__init__()
        self.ts = torch.from_numpy(ts_norm).float()
        self.lengths = torch.from_numpy(lengths).long()
        self.text_embeds = torch.from_numpy(text_embeds).float()
        self.domain_ids = torch.from_numpy(domain_ids).long()

    def __len__(self) -> int:
        return self.ts.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ts[idx], self.lengths[idx], self.text_embeds[idx], self.domain_ids[idx]


DEFAULT_VARIANTS: Tuple[DomainVariant, ...] = (
    DomainVariant("ev_style1", "data/ev/ev_train_X.npy", "embedding/ev/ev_train_labels_1_emb.npy", 24),
    DomainVariant("ev_style2", "data/ev/ev_train_X.npy", "embedding/ev/ev_train_labels_2_emb.npy", 24),
    DomainVariant("ev_style3", "data/ev/ev_train_X.npy", "embedding/ev/ev_train_labels_3_emb.npy", 24),
    DomainVariant("commercial_style1", "data/commercial_load/comm_train_X_1.npy", "embedding/commercial_load/comm_train_labels_1_emb.npy", 96),
    DomainVariant("commercial_style2", "data/commercial_load/comm_train_X_1.npy", "embedding/commercial_load/comm_train_labels_2_emb.npy", 96),
    DomainVariant("commercial_style3", "data/commercial_load/comm_train_X_1.npy", "embedding/commercial_load/comm_train_labels_3_emb.npy", 96),
    DomainVariant("wind_style1", "data/wind/wind_train_X.npy", "embedding/wind/wind_train_labels_1_emb.npy", 24),
    DomainVariant("wind_style2", "data/wind/wind_train_X.npy", "embedding/wind/wind_train_labels_2_emb.npy", 24),
    DomainVariant("wind_style3", "data/wind/wind_train_X.npy", "embedding/wind/wind_train_labels_3_emb.npy", 24),
    DomainVariant("solar_style1", "data/solar/solar_train_X.npy", "embedding/solar/solar_train_labels_1_emb.npy", 24),
    DomainVariant("solar_style2", "data/solar/solar_train_X.npy", "embedding/solar/solar_train_labels_2_emb.npy", 24),
    DomainVariant("solar_style3", "data/solar/solar_train_X.npy", "embedding/solar/solar_train_labels_3_emb.npy", 24),
    DomainVariant("transient_style1", "data/transient_voltage/trans_train_X_1.npy", "embedding/transient_voltage/trans_train_labels_1_emb.npy", 81),
    DomainVariant("transient_style2", "data/transient_voltage/trans_train_X_1.npy", "embedding/transient_voltage/trans_train_labels_2_emb.npy", 81),
    DomainVariant("transient_style3", "data/transient_voltage/trans_train_X_1.npy", "embedding/transient_voltage/trans_train_labels_3_emb.npy", 81),
)


def pad_series_to_length(series: np.ndarray, target_length: int) -> np.ndarray:
    """Right-pad a 2D array ``(N, T)`` to ``(N, target_length)`` with zeros."""

    if series.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {series.shape}")
    current_length = series.shape[1]
    if current_length > target_length:
        raise ValueError(
            f"Sequence length {current_length} exceeds target_length {target_length}"
        )
    pad_width = target_length - current_length
    if pad_width == 0:
        return series.astype(np.float32)
    return np.pad(series, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0).astype(np.float32)


def compute_valid_mask(lengths: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Return a float mask of shape ``(N, max_seq_len)`` with ones on valid positions."""

    idxs = torch.arange(max_seq_len).unsqueeze(0).expand(lengths.size(0), max_seq_len)
    return (idxs < lengths.unsqueeze(1)).float()


def normalize_one_variant(
    ts_raw: np.ndarray,
    lengths: np.ndarray,
    max_seq_len: int,
) -> Tuple[np.ndarray, float, float]:
    """Normalize one padded domain/variant over valid positions only.

    Returns:
        ts_norm: Normalized array with padded positions zeroed out.
        mean: Scalar mean computed over valid positions.
        std: Scalar std computed over valid positions.
    """

    ts_t = torch.from_numpy(ts_raw).float()
    len_t = torch.from_numpy(lengths).long()
    valid = compute_valid_mask(len_t, max_seq_len)

    mean = (ts_t * valid).sum() / valid.sum()
    var = (((ts_t - mean) * valid) ** 2).sum() / valid.sum()
    std = torch.sqrt(var + 1e-6)

    ts_norm = ((ts_t - mean) / std) * valid
    return ts_norm.numpy().astype(np.float32), float(mean.item()), float(std.item())


def load_and_prepare_variants(
    variants: Sequence[DomainVariant],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, float], Dict[int, float]]:
    """Load all training variants, pad them, normalize them, and concatenate them."""

    ts_all: List[np.ndarray] = []
    lengths_all: List[np.ndarray] = []
    text_all: List[np.ndarray] = []
    domain_ids_all: List[np.ndarray] = []
    domain_mean: Dict[int, float] = {}
    domain_std: Dict[int, float] = {}

    for domain_idx, variant in enumerate(variants):
        series = np.load(variant.data_path).astype(np.float32)
        text = np.load(variant.text_path).astype(np.float32)
        if series.shape[0] != text.shape[0]:
            raise ValueError(
                f"Mismatched sample count for {variant.name}: "
                f"series={series.shape[0]} vs text={text.shape[0]}"
            )

        padded = pad_series_to_length(series, variant.pad_to)
        lengths = np.full((padded.shape[0],), variant.true_length, dtype=np.int64)
        normalized, mean, std = normalize_one_variant(padded, lengths, variant.pad_to)

        ts_all.append(normalized)
        lengths_all.append(lengths)
        text_all.append(text)
        domain_ids_all.append(np.full((padded.shape[0],), domain_idx, dtype=np.int64))
        domain_mean[domain_idx] = mean
        domain_std[domain_idx] = std

    return (
        np.concatenate(ts_all, axis=0),
        np.concatenate(lengths_all, axis=0),
        np.concatenate(text_all, axis=0),
        np.concatenate(domain_ids_all, axis=0),
        domain_mean,
        domain_std,
    )


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def build_diffusion_raw(x_channels: int = 1, cond_dim: int = 768) -> GaussianDiffusion:
    """Construct the raw-series diffusion model used in the training script."""

    unet_cfg = UNet1DConfig(
        in_channels=x_channels,
        base_channels=128,
        channel_mults=(1, 2, 2),
        num_res_blocks=2,
        num_heads=4,
        t_emb_dim=256,
    )
    unet = UNet1D(unet_cfg, cond_dim=cond_dim, use_self_attn=True, use_cross_attn=True)
    diffusion_cfg = DiffusionConfig(timesteps=1000, beta_start=1e-4, beta_end=2e-2)
    return GaussianDiffusion(unet, diffusion_cfg)


def train_one_epoch(
    model: GaussianDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    p_uncond: float,
    null_fill_value: float,
) -> float:
    """Run one diffusion training epoch and return mean loss."""

    model.train()
    total_loss = 0.0
    total_samples = 0

    for ts_batch, len_batch, txt_batch, _domain_batch in loader:
        ts_batch = ts_batch.to(device)
        len_batch = len_batch.to(device)
        txt_batch = txt_batch.to(device)

        max_len = ts_batch.shape[1]
        valid_mask = (
            torch.arange(max_len, device=device)[None, :] < len_batch[:, None]
        ).float().unsqueeze(1)

        x0 = ts_batch.unsqueeze(1)

        uncond_mask = torch.rand(x0.size(0), device=device) < p_uncond
        cond_tokens = txt_batch.clone()
        if uncond_mask.any():
            cond_tokens[uncond_mask] = null_fill_value

        loss = model.loss(x0, cond_tokens, valid_mask=valid_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x0.size(0)
        total_samples += x0.size(0)

    return total_loss / max(total_samples, 1)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """Train the direct diffusion model on all configured domains."""

    config = TrainingConfig()
    set_seed(config.random_seed)
    device = select_device()
    print(f"Using device: {device}")

    ts_norm, lengths, text_embeds, domain_ids, domain_mean, domain_std = load_and_prepare_variants(
        DEFAULT_VARIANTS
    )

    dataset = MultiDomainTimeSeriesTextDataset(ts_norm, lengths, text_embeds, domain_ids)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    diffusion = build_diffusion_raw(x_channels=1, cond_dim=768).to(device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.num_epochs):
        epoch_loss = train_one_epoch(
            model=diffusion,
            loader=loader,
            optimizer=optimizer,
            device=device,
            p_uncond=config.p_uncond,
            null_fill_value=config.null_fill_value,
        )
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch + 1}] diffusion loss: {epoch_loss:.6f}")

    checkpoint = {
        "model_state_dict": diffusion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": diffusion.cfg.__dict__,
        "training_config": config.__dict__,
        "domain_mean": domain_mean,
        "domain_std": domain_std,
        "variants": [variant.__dict__ for variant in DEFAULT_VARIANTS],
    }
    torch.save(checkpoint, config.checkpoint_path)
    print(f"Saved checkpoint to {config.checkpoint_path}")


if __name__ == "__main__":
    main()
