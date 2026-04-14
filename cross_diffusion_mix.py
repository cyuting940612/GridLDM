"""
Latent diffusion training for multi-domain 1D time-series conditioned on text embeddings.

This module trains a DDPM-style diffusion model in the latent space of a frozen
conditional time-series autoencoder. The intended workflow is:

1. Load per-domain training signals and precomputed text-token embeddings.
2. Pad variable-length signals to a shared maximum sequence length.
3. Normalize each domain using statistics computed only over valid, unpadded values.
4. Encode normalized signals into latent vectors with a frozen autoencoder.
5. Train a 1D U-Net diffusion model on those latent vectors, conditioned on the
   corresponding text embeddings via cross-attention.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from encoder_decoder_conditional import ConditionalTimeSeriesAutoencoder


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class DomainSpec:
    """Configuration for one physical data domain.

    Attributes:
        name: Human-readable domain name used for bookkeeping.
        signal_path: Path to the training time-series array for this domain.
        text_paths: Paths to one or more text-embedding files. The same signal
            array is paired with each of these prompt variants.
        valid_length: Number of real time steps before zero-padding.
        pad_to: Shared padded length used by the autoencoder.
    """

    name: str
    signal_path: str
    text_paths: Sequence[str]
    valid_length: int
    pad_to: int = 96


@dataclass
class TrainingConfig:
    """Hyperparameters and file paths for diffusion training."""

    autoencoder_checkpoint: str = "cond_autoencoder.pt"
    diffusion_checkpoint: str = "gridldm_diffusion.pt"
    max_seq_len: int = 96
    latent_dim: int = 24
    text_token_dim: int = 768
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_epochs: int = 1000
    unconditional_dropout_prob: float = 0.10
    null_condition_value: float = 0.0
    log_every: int = 10

    # Frozen autoencoder configuration.
    ae_d_model: int = 64
    ae_nhead: int = 8
    ae_num_encoder_layers: int = 3
    ae_num_decoder_layers: int = 3
    ae_dim_feedforward: int = 128
    ae_dropout: float = 0.1


DEFAULT_DOMAIN_SPECS: Tuple[DomainSpec, ...] = (
    DomainSpec(
        name="ev",
        signal_path="data/ev/ev_train_X.npy",
        text_paths=(
            "embedding/ev/ev_train_labels_1_emb.npy",
            "embedding/ev/ev_train_labels_2_emb.npy",
            "embedding/ev/ev_train_labels_3_emb.npy",
        ),
        valid_length=24,
    ),
    DomainSpec(
        name="commercial_load",
        signal_path="data/commercial_load/comm_train_X_1.npy",
        text_paths=(
            "embedding/commercial_load/comm_train_labels_1_emb.npy",
            "embedding/commercial_load/comm_train_labels_2_emb.npy",
            "embedding/commercial_load/comm_train_labels_3_emb.npy",
        ),
        valid_length=96,
    ),
    DomainSpec(
        name="wind",
        signal_path="data/wind/wind_train_X.npy",
        text_paths=(
            "embedding/wind/wind_train_labels_1_emb.npy",
            "embedding/wind/wind_train_labels_2_emb.npy",
            "embedding/wind/wind_train_labels_3_emb.npy",
        ),
        valid_length=24,
    ),
    DomainSpec(
        name="solar",
        signal_path="data/solar/solar_train_X.npy",
        text_paths=(
            "embedding/solar/solar_train_labels_1_emb.npy",
            "embedding/solar/solar_train_labels_2_emb.npy",
            "embedding/solar/solar_train_labels_3_emb.npy",
        ),
        valid_length=24,
    ),
    DomainSpec(
        name="transient_voltage",
        signal_path="data/transient_voltage/trans_train_X_1.npy",
        text_paths=(
            "embedding/transient_voltage/trans_train_labels_1_emb.npy",
            "embedding/transient_voltage/trans_train_labels_2_emb.npy",
            "embedding/transient_voltage/trans_train_labels_3_emb.npy",
        ),
        valid_length=81,
    ),
)


# =============================================================================
# Utility functions
# =============================================================================


def get_best_device() -> torch.device:
    """Return the best available execution device.

    Priority order:
    1. CUDA
    2. Apple Metal (MPS)
    3. CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute sinusoidal timestep embeddings.

    Args:
        timesteps: Tensor of shape ``(B,)`` containing integer or floating-point
            diffusion timesteps.
        dim: Output embedding dimension.

    Returns:
        Tensor of shape ``(B, dim)``.
    """

    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half, 1)
    )
    t = timesteps.float()
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class GELU(nn.Module):
    """Thin module wrapper around ``torch.nn.functional.gelu``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


# =============================================================================
# Attention and U-Net building blocks
# =============================================================================


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the temporal axis of a 1D feature map.

    Input shape: ``(B, C, T)``
    Output shape: ``(B, C, T)``
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
        """Apply self-attention and add the residual connection."""

        batch_size, channels, time_steps = x.shape
        h = self.norm(x)
        q, k, v = torch.chunk(self.qkv(h), chunks=3, dim=1)

        q = q.view(batch_size, self.num_heads, self.head_dim, time_steps)
        k = k.view(batch_size, self.num_heads, self.head_dim, time_steps)
        v = v.view(batch_size, self.num_heads, self.head_dim, time_steps)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("b h d t, b h d s -> b h t s", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("b h t s, b h d s -> b h d t", attn, v)
        out = out.reshape(batch_size, channels, time_steps)
        out = self.proj(out)
        return x + out


class CrossAttention1D(nn.Module):
    """Cross-attention from time-series features to condition tokens.

    Query input:
        ``x`` with shape ``(B, C, T)``

    Condition input:
        ``cond_tokens`` with shape ``(B, M, D)``

    Output:
        Tensor of shape ``(B, C, T)``
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
        """Apply cross-attention without an implicit residual connection.

        The enclosing U-Net block decides whether to treat this result as a pure
        transform or as part of a larger residual pathway.
        """

        batch_size, channels, time_steps = x.shape
        _, num_tokens, _ = cond_tokens.shape

        q = self.to_q(self.norm_x(x)).view(
            batch_size, self.num_heads, self.head_dim, time_steps
        )

        cond = self.norm_cond(cond_tokens)
        k = self.to_k(cond).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v = self.to_v(cond).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("b h d t, b h d m -> b h t m", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("b h t m, b h d m -> b h d t", attn, v)
        out = out.reshape(batch_size, channels, time_steps)
        return self.proj(out)


class ResBlock(nn.Module):
    """Residual convolution block with timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = GELU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(t_emb_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape ``(B, C, T)``.
            t_emb: Timestep embedding of shape ``(B, t_emb_dim)``.
        """

        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return self.skip(x) + h


class Downsample(nn.Module):
    """Halve the temporal resolution with a strided 1D convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Double the temporal resolution with a transposed 1D convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =============================================================================
# Diffusion model
# =============================================================================


@dataclass
class UNet1DConfig:
    """Configuration for the 1D U-Net denoiser."""

    in_channels: int = 1
    base_channels: int = 128
    channel_mults: Tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    num_heads: int = 4
    t_emb_dim: int = 256
    dropout: float = 0.0


class UNet1D(nn.Module):
    """1D U-Net with self-attention and cross-attention blocks."""

    def __init__(
        self,
        cfg: UNet1DConfig,
        cond_dim: int,
        use_self_attn: bool = True,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.cfg = cfg

        base = cfg.base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.t_emb_dim, cfg.t_emb_dim * 4),
            GELU(),
            nn.Linear(cfg.t_emb_dim * 4, cfg.t_emb_dim),
        )
        self.in_conv = nn.Conv1d(cfg.in_channels, base, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        channels_for_skips: List[int] = []
        in_ch = base

        for level, mult in enumerate(cfg.channel_mults):
            out_ch = base * mult
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(
                    nn.ModuleDict(
                        {
                            "res": ResBlock(in_ch, out_ch, cfg.t_emb_dim, cfg.dropout),
                            "self_attn": (
                                SelfAttention1D(out_ch, cfg.num_heads)
                                if use_self_attn
                                else nn.Identity()
                            ),
                            "cross_attn": (
                                CrossAttention1D(out_ch, cond_dim, cfg.num_heads)
                                if use_cross_attn
                                else nn.Identity()
                            ),
                        }
                    )
                )
                in_ch = out_ch
                channels_for_skips.append(out_ch)

            if level != len(cfg.channel_mults) - 1:
                self.down_blocks.append(nn.ModuleDict({"down": Downsample(in_ch)}))
                channels_for_skips.append(in_ch)

        self.mid_block1 = ResBlock(in_ch, in_ch, cfg.t_emb_dim, cfg.dropout)
        self.mid_self = (
            SelfAttention1D(in_ch, cfg.num_heads) if use_self_attn else nn.Identity()
        )
        self.mid_cross = (
            CrossAttention1D(in_ch, cond_dim, cfg.num_heads)
            if use_cross_attn
            else nn.Identity()
        )
        self.mid_block2 = ResBlock(in_ch, in_ch, cfg.t_emb_dim, cfg.dropout)

        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(cfg.channel_mults))):
            out_ch = base * mult
            num_res_blocks_up = cfg.num_res_blocks + (1 if level != 0 else 0)
            for _ in range(num_res_blocks_up):
                skip_ch = channels_for_skips.pop()
                self.up_blocks.append(
                    nn.ModuleDict(
                        {
                            "res": ResBlock(
                                in_ch + skip_ch,
                                out_ch,
                                cfg.t_emb_dim,
                                cfg.dropout,
                            ),
                            "self_attn": (
                                SelfAttention1D(out_ch, cfg.num_heads)
                                if use_self_attn
                                else nn.Identity()
                            ),
                            "cross_attn": (
                                CrossAttention1D(out_ch, cond_dim, cfg.num_heads)
                                if use_cross_attn
                                else nn.Identity()
                            ),
                        }
                    )
                )
                in_ch = out_ch

            if level != 0:
                self.up_blocks.append(nn.ModuleDict({"up": Upsample(in_ch)}))

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act = GELU()
        self.out_conv = nn.Conv1d(in_ch, cfg.in_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise for a noisy latent trajectory.

        Args:
            x: Noisy latent tensor with shape ``(B, C, T)``.
            t: Diffusion timestep tensor with shape ``(B,)``.
            cond_tokens: Text conditioning tokens with shape ``(B, M, D)``.
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
        if isinstance(self.mid_cross, CrossAttention1D):
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


@dataclass
class DiffusionConfig:
    """Noise schedule configuration for DDPM training and sampling."""

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    objective: str = "eps"


class GaussianDiffusion(nn.Module):
    """DDPM wrapper around the 1D U-Net denoiser."""

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
        """Diffuse a clean sample ``x0`` to timestep ``t``."""

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
        """Compute the DDPM posterior mean and variance for sampling."""

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

    def loss(self, x0: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        """DDPM epsilon-prediction objective."""

        batch_size = x0.size(0)
        t = torch.randint(0, self.cfg.timesteps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, t, cond_tokens)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int],
        cond_tokens: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate samples by ancestral DDPM sampling."""

        batch_size, _, _ = shape
        sample_device = device if device is not None else cond_tokens.device
        x = torch.randn(shape, device=sample_device)

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
        """Classifier-free guidance sampling using precomputed condition tokens."""

        batch_size, _, _ = shape
        sample_device = device if device is not None else cond_tokens.device
        x = torch.randn(shape, device=sample_device)

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


# =============================================================================
# Dataset utilities
# =============================================================================


class MultiDomainTimeSeriesTextDataset(Dataset):
    """Multi-domain dataset of padded time series, lengths, and text embeddings.

    Attributes:
        ts: Tensor of shape ``(N, T_max)`` containing normalized padded signals.
        lengths: Tensor of shape ``(N,)`` containing valid lengths.
        text_embeds: Tensor of shape ``(N, L_text, D_text)``.
        domain_ids: Tensor of shape ``(N,)`` identifying the physical domain.
    """

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
        return (
            self.ts[idx],
            self.lengths[idx],
            self.text_embeds[idx],
            self.domain_ids[idx],
        )


@dataclass
class DomainNormalizationStats:
    """Per-domain normalization statistics computed over valid signal values only."""

    mean: float
    std: float


@dataclass
class PreparedDataset:
    """Container for the concatenated training dataset and normalization stats."""

    dataset: MultiDomainTimeSeriesTextDataset
    domain_stats: Dict[str, DomainNormalizationStats] = field(default_factory=dict)


def load_array(path: str) -> np.ndarray:
    """Load a NumPy array from disk with a clear error message."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {file_path}")
    return np.load(file_path)


def pad_timeseries(array: np.ndarray, target_len: int) -> np.ndarray:
    """Pad 2D time-series data on the trailing time axis.

    Args:
        array: Input array of shape ``(N, T)``.
        target_len: Target temporal length after padding.

    Returns:
        Array of shape ``(N, target_len)``.
    """

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array of shape (N, T), got {array.shape}")

    current_len = array.shape[1]
    if current_len > target_len:
        raise ValueError(
            f"Cannot pad from length {current_len} down to shorter target {target_len}"
        )
    if current_len == target_len:
        return array.astype(np.float32, copy=False)

    pad_width = target_len - current_len
    return np.pad(array, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)


def compute_domain_normalization(
    ts_raw: np.ndarray,
    lengths: np.ndarray,
    max_seq_len: int,
) -> Tuple[np.ndarray, DomainNormalizationStats]:
    """Normalize one domain using only valid, non-padded signal values.

    Args:
        ts_raw: Padded array of shape ``(N, T_max)``.
        lengths: Valid lengths of shape ``(N,)``.
        max_seq_len: Shared padded length.

    Returns:
        A tuple ``(ts_norm, stats)`` where ``ts_norm`` has the same shape as
        ``ts_raw`` and padded positions are forced to zero.
    """

    ts_tensor = torch.from_numpy(ts_raw).float()
    length_tensor = torch.from_numpy(lengths).long()

    num_samples = ts_tensor.size(0)
    idxs = torch.arange(max_seq_len).unsqueeze(0).expand(num_samples, max_seq_len)
    valid_mask = (idxs < length_tensor.unsqueeze(1)).float()

    mean = (ts_tensor * valid_mask).sum() / valid_mask.sum()
    var = (((ts_tensor - mean) * valid_mask) ** 2).sum() / valid_mask.sum()
    std = torch.sqrt(var + 1e-6)

    ts_norm = ((ts_tensor - mean) / std) * valid_mask
    stats = DomainNormalizationStats(mean=mean.item(), std=std.item())
    return ts_norm.numpy().astype(np.float32), stats


def prepare_multidomain_dataset(
    domain_specs: Sequence[DomainSpec],
    max_seq_len: int,
) -> PreparedDataset:
    """Load, normalize, and concatenate all configured domains.

    For each domain, the same signal array is paired with each prompt-style
    embedding file listed in ``text_paths``.
    """

    ts_all: List[np.ndarray] = []
    lengths_all: List[np.ndarray] = []
    text_all: List[np.ndarray] = []
    domain_ids_all: List[np.ndarray] = []
    domain_stats: Dict[str, DomainNormalizationStats] = {}

    for domain_idx, spec in enumerate(domain_specs):
        ts_raw = load_array(spec.signal_path)
        ts_padded = pad_timeseries(ts_raw, target_len=max_seq_len)
        lengths = np.full(ts_padded.shape[0], spec.valid_length, dtype=np.int64)

        ts_norm, stats = compute_domain_normalization(ts_padded, lengths, max_seq_len)
        domain_stats[spec.name] = stats

        for text_path in spec.text_paths:
            text_embeds = load_array(text_path).astype(np.float32)
            if text_embeds.shape[0] != ts_norm.shape[0]:
                raise ValueError(
                    f"Sample count mismatch for domain '{spec.name}': "
                    f"signals have {ts_norm.shape[0]} rows but text embeddings from "
                    f"{text_path} have {text_embeds.shape[0]} rows."
                )

            ts_all.append(ts_norm)
            lengths_all.append(lengths)
            text_all.append(text_embeds)
            domain_ids_all.append(
                np.full(ts_norm.shape[0], domain_idx, dtype=np.int64)
            )

    dataset = MultiDomainTimeSeriesTextDataset(
        ts_norm=np.concatenate(ts_all, axis=0),
        lengths=np.concatenate(lengths_all, axis=0),
        text_embeds=np.concatenate(text_all, axis=0),
        domain_ids=np.concatenate(domain_ids_all, axis=0),
    )
    return PreparedDataset(dataset=dataset, domain_stats=domain_stats)


# =============================================================================
# Model builders and training
# =============================================================================


def build_diffusion(cond_dim: int = 768) -> GaussianDiffusion:
    """Construct the latent-space diffusion model used in the original script."""

    unet_cfg = UNet1DConfig(
        in_channels=1,
        base_channels=128,
        channel_mults=(1, 2, 2),
        num_res_blocks=2,
        num_heads=4,
        t_emb_dim=256,
    )
    unet = UNet1D(unet_cfg, cond_dim=cond_dim, use_self_attn=True, use_cross_attn=True)
    diff_cfg = DiffusionConfig(timesteps=1000, beta_start=1e-4, beta_end=2e-2)
    return GaussianDiffusion(unet, diff_cfg)


def load_frozen_autoencoder(
    cfg: TrainingConfig,
    device: torch.device,
) -> ConditionalTimeSeriesAutoencoder:
    """Load the pretrained conditional autoencoder and freeze its parameters."""

    model = ConditionalTimeSeriesAutoencoder(
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.ae_d_model,
        latent_dim=cfg.latent_dim,
        nhead=cfg.ae_nhead,
        num_encoder_layers=cfg.ae_num_encoder_layers,
        num_decoder_layers=cfg.ae_num_decoder_layers,
        dim_feedforward=cfg.ae_dim_feedforward,
        dropout=cfg.ae_dropout,
        text_token_dim=cfg.text_token_dim,
    ).to(device)

    checkpoint = torch.load(cfg.autoencoder_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@dataclass
class TrainingHistory:
    """Simple container for training losses."""

    losses: List[float] = field(default_factory=list)


def train_diffusion(
    diffusion: GaussianDiffusion,
    autoencoder: ConditionalTimeSeriesAutoencoder,
    train_loader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
) -> TrainingHistory:
    """Train the latent diffusion model.

    During each iteration:
    1. Encode normalized time series into latent vectors with the frozen AE.
    2. Treat each latent vector as a 1D signal of shape ``(B, 1, latent_dim)``.
    3. Randomly drop a subset of text conditions for classifier-free guidance.
    4. Optimize the DDPM epsilon-prediction loss.
    """

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    history = TrainingHistory()

    for epoch in range(cfg.num_epochs):
        diffusion.train()
        epoch_loss = 0.0
        num_samples = 0

        for ts_batch, len_batch, txt_batch, _domain_batch in train_loader:
            ts_batch = ts_batch.to(device)
            len_batch = len_batch.to(device)
            txt_batch = txt_batch.to(device)

            with torch.no_grad():
                latents = autoencoder.encode(ts_batch, len_batch, txt_batch)

            latent_signal = latents.unsqueeze(1)

            uncond_mask = (
                torch.rand(latent_signal.size(0), device=device)
                < cfg.unconditional_dropout_prob
            )
            cond_tokens = txt_batch.clone()
            if uncond_mask.any():
                cond_tokens[uncond_mask] = cfg.null_condition_value

            loss = diffusion.loss(latent_signal, cond_tokens)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = latent_signal.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

        mean_epoch_loss = epoch_loss / max(num_samples, 1)
        history.losses.append(mean_epoch_loss)

        if (epoch + 1) % cfg.log_every == 0:
            print(f"[Epoch {epoch + 1:04d}] diffusion loss: {mean_epoch_loss:.6f}")

    checkpoint = {
        "model_state_dict": diffusion.state_dict(),
        "config": asdict(cfg),
        "diffusion_config": asdict(diffusion.cfg),
        "training_loss_history": history.losses,
    }
    torch.save(checkpoint, cfg.diffusion_checkpoint)
    print(f"Saved diffusion checkpoint to: {cfg.diffusion_checkpoint}")

    return history


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """Run end-to-end latent diffusion training using the default file layout."""

    cfg = TrainingConfig()
    device = get_best_device()
    print(f"Using device: {device}")

    autoencoder = load_frozen_autoencoder(cfg, device)
    diffusion = build_diffusion(cond_dim=cfg.text_token_dim).to(device)

    prepared = prepare_multidomain_dataset(
        domain_specs=DEFAULT_DOMAIN_SPECS,
        max_seq_len=cfg.max_seq_len,
    )

    print("Loaded domains:")
    for domain_name, stats in prepared.domain_stats.items():
        print(
            f"  - {domain_name}: mean={stats.mean:.6f}, std={stats.std:.6f}"
        )

    train_loader = DataLoader(
        prepared.dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    train_diffusion(
        diffusion=diffusion,
        autoencoder=autoencoder,
        train_loader=train_loader,
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    main()
