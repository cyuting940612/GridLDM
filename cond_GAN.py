"""
Early-fusion conditional GAN for multi-domain 1D time-series generation.

Model summary
-------------
- Generator:
    z + pooled text embeddings -> fused vector -> transposed-convolution decoder -> x_hat
- Discriminator:
    x + pooled text embeddings -> FiLM modulation on first feature map -> score
- Conditioning:
    text embeddings are assumed to be precomputed token embeddings of shape (B, M, 768)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class DomainVariant:
    """One training source: a numeric series file plus a matching text-embedding file."""

    name: str
    signal_path: str
    text_path: str
    true_length: int
    pad_to: int = 96


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level training configuration."""

    batch_size: int = 256
    z_dim: int = 128
    cond_dim: int = 768
    t_out: int = 96
    num_epochs: int = 100
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    d_steps: int = 1
    train_fraction: float = 0.8
    shuffle_seed: int = 42
    checkpoint_path: str = "cgan.pt"


# =============================================================================
# Small utility modules
# =============================================================================


class GELU(nn.Module):
    """Thin module wrapper around GELU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class CondPool(nn.Module):
    """Pool condition tokens from shape (B, M, D) to (B, D) with LayerNorm + mean."""

    def __init__(self, cond_dim: int = 768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cond_dim)

    def forward(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(cond_tokens).mean(dim=1)


class MLP(nn.Module):
    """Simple two-layer MLP."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Early-fusion conditioning modules
# =============================================================================


class GeneratorFuse(nn.Module):
    """
    Early fusion for the generator.

    Inputs
    ------
    z : (B, z_dim)
        Noise vector.
    cond_tokens : (B, M, cond_dim)
        Text-conditioning token embeddings.

    Returns
    -------
    fused : (B, out_dim)
        Joint representation of noise and conditioning.
    """

    def __init__(self, z_dim: int, cond_dim: int = 768, out_dim: int = 512, hidden: int = 1024) -> None:
        super().__init__()
        self.pool = CondPool(cond_dim)
        self.mlp = MLP(z_dim + cond_dim, out_dim, hidden=hidden)

    def forward(self, z: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        pooled_cond = self.pool(cond_tokens)
        return self.mlp(torch.cat([z, pooled_cond], dim=1))


class DiscriminatorFuseFiLM(nn.Module):
    """
    Produce FiLM parameters for the discriminator from pooled condition tokens.

    Returns gamma and beta with shape (B, C), which are applied as:
        h <- h * (1 + gamma) + beta
    """

    def __init__(self, cond_dim: int = 768, channels: int = 128, hidden: int = 512) -> None:
        super().__init__()
        self.pool = CondPool(cond_dim)
        self.to_gamma_beta = MLP(cond_dim, out_dim=2 * channels, hidden=hidden)

    def forward(self, cond_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_cond = self.pool(cond_tokens)
        gamma_beta = self.to_gamma_beta(pooled_cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return gamma, beta


# =============================================================================
# Generator
# =============================================================================


class GenBlock1D(nn.Module):
    """A generator upsampling block for 1D signals."""

    def __init__(self, in_ch: int, out_ch: int, upsample: bool) -> None:
        super().__init__()
        if upsample:
            self.proj = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm = nn.GroupNorm(8, out_ch)
        self.act = GELU()
        self.conv = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.act(self.norm(x))
        return self.conv(x)


class ConditionalGenerator1D(nn.Module):
    """
    Early-fusion conditional generator.

    Architecture
    ------------
    z + pooled text -> fused vector -> linear projection to short grid ->
    three upsampling blocks -> output signal of shape (B, 1, 96)
    """

    def __init__(
        self,
        z_dim: int = 128,
        cond_dim: int = 768,
        t_out: int = 96,
        base_ch: int = 128,
        fuse_dim: int = 512,
    ) -> None:
        super().__init__()
        if t_out != 96:
            raise ValueError("This implementation assumes t_out=96.")

        self.fuse = GeneratorFuse(z_dim=z_dim, cond_dim=cond_dim, out_dim=fuse_dim, hidden=1024)

        # Temporal growth: 12 -> 24 -> 48 -> 96
        self.init_t = 12
        self.fc = nn.Linear(fuse_dim, base_ch * self.init_t)

        self.block1 = GenBlock1D(base_ch, base_ch, upsample=True)
        self.block2 = GenBlock1D(base_ch, base_ch // 2, upsample=True)
        self.block3 = GenBlock1D(base_ch // 2, base_ch // 4, upsample=True)

        self.out_norm = nn.GroupNorm(8, base_ch // 4)
        self.out_act = GELU()
        self.out_conv = nn.Conv1d(base_ch // 4, 1, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(z, cond_tokens)
        h = self.fc(fused).view(z.size(0), -1, self.init_t)

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)

        return self.out_conv(self.out_act(self.out_norm(h)))


# =============================================================================
# Discriminator
# =============================================================================


class DiscBlock1D(nn.Module):
    """A discriminator block with optional strided downsampling."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool) -> None:
        super().__init__()
        if downsample:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ConditionalDiscriminator1D(nn.Module):
    """
    Early-fusion conditional discriminator.

    Conditioning is injected by FiLM modulation on the first feature map.
    """

    def __init__(self, cond_dim: int = 768, base_ch: int = 128) -> None:
        super().__init__()
        self.in_conv = nn.Conv1d(1, base_ch, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.film = DiscriminatorFuseFiLM(cond_dim=cond_dim, channels=base_ch, hidden=512)

        # Temporal reduction: 96 -> 48 -> 24 -> 12
        self.block1 = DiscBlock1D(base_ch, base_ch, downsample=True)
        self.block2 = DiscBlock1D(base_ch, base_ch * 2, downsample=True)
        self.block3 = DiscBlock1D(base_ch * 2, base_ch * 4, downsample=True)

        self.final = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(base_ch * 4, 1),
        )

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_conv(x))

        gamma, beta = self.film(cond_tokens)
        h = h * (1.0 + gamma[:, :, None]) + beta[:, :, None]

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)

        return self.final(h)


# =============================================================================
# Dataset
# =============================================================================


class MultiDomainTimeSeriesTextDataset(Dataset):
    """
    Dataset of normalized padded time-series, true lengths, text embeddings, and domain IDs.

    Parameters
    ----------
    ts_norm : np.ndarray
        Shape (N, T)
    lengths : np.ndarray
        Shape (N,)
    text_embeds : np.ndarray
        Shape (N, M, D)
    domain_ids : np.ndarray
        Shape (N,)
    """

    def __init__(
        self,
        ts_norm: np.ndarray,
        lengths: np.ndarray,
        text_embeds: np.ndarray,
        domain_ids: np.ndarray,
    ) -> None:
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


# =============================================================================
# Losses and masks
# =============================================================================


def d_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for the discriminator."""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_hinge_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for the generator."""
    return (-d_fake).mean()


def make_valid_mask(lengths: torch.Tensor, t_out: int, device: torch.device) -> torch.Tensor:
    """
    Build a mask of shape (B, 1, T) with 1 on valid positions and 0 on padded positions.
    """
    idx = torch.arange(t_out, device=device)[None, :]
    mask = (idx < lengths[:, None]).float()
    return mask.unsqueeze(1)


# =============================================================================
# Data loading and preprocessing
# =============================================================================


def pad_to_length(x: np.ndarray, target_length: int) -> np.ndarray:
    """Right-pad a 2D array of shape (N, T) to (N, target_length)."""
    n, t = x.shape
    if t > target_length:
        raise ValueError(f"Input length {t} exceeds target length {target_length}.")
    if t == target_length:
        return x.astype(np.float32)
    pad_width = ((0, 0), (0, target_length - t))
    return np.pad(x, pad_width, mode="constant", constant_values=0).astype(np.float32)


def load_variant(variant: DomainVariant) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one training source.

    Returns
    -------
    signal : (N, 96)
    lengths : (N,)
    text : (N, M, D)
    """
    signal = np.load(variant.signal_path).astype(np.float32)
    text = np.load(variant.text_path).astype(np.float32)
    signal = pad_to_length(signal, variant.pad_to)

    if signal.shape[0] != text.shape[0]:
        raise ValueError(
            f"Row mismatch for {variant.name}: "
            f"signal has {signal.shape[0]} rows, text has {text.shape[0]} rows."
        )

    lengths = np.full(signal.shape[0], variant.true_length, dtype=np.int64)
    return signal, lengths, text


def normalize_variant(
    signal: np.ndarray,
    lengths: np.ndarray,
    variant_id: int,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """
    Normalize a padded time-series variant using valid positions only.

    Returns
    -------
    signal_norm : (N, T)
    stats : dict
        Contains variant_id, mean, std
    valid_mask : (N, T)
    """
    signal_t = torch.from_numpy(signal).float()
    lengths_t = torch.from_numpy(lengths).long()

    n, t = signal_t.shape
    idx = torch.arange(t).unsqueeze(0).expand(n, t)
    valid = (idx < lengths_t.unsqueeze(1)).float()

    mean = (signal_t * valid).sum() / valid.sum()
    var = (((signal_t - mean) * valid) ** 2).sum() / valid.sum()
    std = torch.sqrt(var + 1e-6)

    signal_norm = ((signal_t - mean) / std) * valid

    stats = {
        "variant_id": float(variant_id),
        "mean": float(mean.item()),
        "std": float(std.item()),
    }
    return signal_norm.numpy().astype(np.float32), stats, valid.numpy().astype(np.float32)


def build_training_arrays(
    variants: List[DomainVariant],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Load, normalize, and concatenate all configured variants.

    Returns
    -------
    ts_norm_all : (N_total, T)
    lengths_all : (N_total,)
    text_all : (N_total, M, D)
    domain_ids_all : (N_total,)
    norm_stats : dict[int, dict]
    """
    ts_norm_list: List[np.ndarray] = []
    lengths_list: List[np.ndarray] = []
    text_list: List[np.ndarray] = []
    domain_ids_list: List[np.ndarray] = []
    norm_stats: Dict[int, Dict[str, float]] = {}

    for variant_id, variant in enumerate(variants):
        signal, lengths, text = load_variant(variant)
        signal_norm, stats, _ = normalize_variant(signal, lengths, variant_id)

        ts_norm_list.append(signal_norm)
        lengths_list.append(lengths)
        text_list.append(text)
        domain_ids_list.append(np.full(lengths.shape[0], variant_id, dtype=np.int64))
        norm_stats[variant_id] = stats | {"name": variant.name}

    return (
        np.concatenate(ts_norm_list, axis=0),
        np.concatenate(lengths_list, axis=0),
        np.concatenate(text_list, axis=0),
        np.concatenate(domain_ids_list, axis=0),
        norm_stats,
    )


def make_train_loader(
    ts_norm_all: np.ndarray,
    lengths_all: np.ndarray,
    text_all: np.ndarray,
    domain_ids_all: np.ndarray,
    batch_size: int,
    train_fraction: float,
    seed: int,
) -> DataLoader:
    """
    Build a shuffled training loader from the first train_fraction of a random permutation.
    """
    n_total = ts_norm_all.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    train_size = int(train_fraction * n_total)
    train_idx = perm[:train_size]

    dataset = MultiDomainTimeSeriesTextDataset(
        ts_norm_all[train_idx],
        lengths_all[train_idx],
        text_all[train_idx],
        domain_ids_all[train_idx],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# =============================================================================
# Training
# =============================================================================


def train_cgan(
    train_loader: DataLoader,
    generator: ConditionalGenerator1D,
    discriminator: ConditionalDiscriminator1D,
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[ConditionalGenerator1D, ConditionalDiscriminator1D]:
    """
    Train the conditional GAN with hinge losses.

    Notes
    -----
    - Real and fake signals are masked over padded regions before they are sent to D.
    - The current implementation trains on the training loader only and does not use a
      validation set, matching the behavior of the original script.
    """
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_g, betas=config.betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=config.betas)

    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()

        loss_g_meter = 0.0
        loss_d_meter = 0.0
        n_samples = 0

        for ts_batch, len_batch, txt_batch, _domain_batch in train_loader:
            ts_batch = ts_batch.to(device)
            len_batch = len_batch.to(device)
            cond_tokens = txt_batch.to(device)

            x_real = ts_batch.unsqueeze(1)
            valid_mask = make_valid_mask(len_batch, config.t_out, device)
            x_real = x_real * valid_mask

            batch_size = x_real.size(0)

            # -------------------------
            # Train discriminator
            # -------------------------
            for _ in range(config.d_steps):
                z = torch.randn(batch_size, config.z_dim, device=device)
                with torch.no_grad():
                    x_fake = generator(z, cond_tokens) * valid_mask

                d_real = discriminator(x_real, cond_tokens)
                d_fake = discriminator(x_fake, cond_tokens)
                loss_d = d_hinge_loss(d_real, d_fake)

                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            # -------------------------
            # Train generator
            # -------------------------
            z = torch.randn(batch_size, config.z_dim, device=device)
            x_fake = generator(z, cond_tokens) * valid_mask
            d_fake = discriminator(x_fake, cond_tokens)
            loss_g = g_hinge_loss(d_fake)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            loss_g_meter += loss_g.item() * batch_size
            loss_d_meter += loss_d.item() * batch_size
            n_samples += batch_size

        print(
            f"[Epoch {epoch + 1:03d}] "
            f"loss_d={loss_d_meter / max(n_samples, 1):.4f} "
            f"loss_g={loss_g_meter / max(n_samples, 1):.4f}"
        )

    return generator, discriminator


# =============================================================================
# Runtime helpers
# =============================================================================


def select_device() -> torch.device:
    """Choose CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_variants() -> List[DomainVariant]:
    """
    Default training sources.

    The original script repeated the same numeric array three times for each physical domain,
    pairing it with three different prompt-style embedding files. This preserves that behavior.
    """
    return [
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
    ]


def save_checkpoint(
    generator: ConditionalGenerator1D,
    discriminator: ConditionalDiscriminator1D,
    norm_stats: Dict[int, Dict[str, float]],
    config: TrainingConfig,
) -> None:
    """Save the trained GAN and normalization metadata."""
    checkpoint = {
        "G_state_dict": generator.state_dict(),
        "D_state_dict": discriminator.state_dict(),
        "z_dim": config.z_dim,
        "cond_dim": config.cond_dim,
        "T_out": config.t_out,
        "normalization_stats": norm_stats,
        "training_config": {
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "lr_g": config.lr_g,
            "lr_d": config.lr_d,
            "betas": config.betas,
            "d_steps": config.d_steps,
            "train_fraction": config.train_fraction,
            "shuffle_seed": config.shuffle_seed,
        },
    }
    torch.save(checkpoint, config.checkpoint_path)
    print(f"Saved checkpoint to {config.checkpoint_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    config = TrainingConfig()
    device = select_device()
    print(f"Using device: {device}")

    variants = default_variants()
    ts_norm_all, lengths_all, text_all, domain_ids_all, norm_stats = build_training_arrays(variants)

    train_loader = make_train_loader(
        ts_norm_all=ts_norm_all,
        lengths_all=lengths_all,
        text_all=text_all,
        domain_ids_all=domain_ids_all,
        batch_size=config.batch_size,
        train_fraction=config.train_fraction,
        seed=config.shuffle_seed,
    )

    generator = ConditionalGenerator1D(
        z_dim=config.z_dim,
        cond_dim=config.cond_dim,
        t_out=config.t_out,
    )
    discriminator = ConditionalDiscriminator1D(cond_dim=config.cond_dim)

    generator, discriminator = train_cgan(
        train_loader=train_loader,
        generator=generator,
        discriminator=discriminator,
        config=config,
        device=device,
    )

    save_checkpoint(generator, discriminator, norm_stats, config)


if __name__ == "__main__":
    main()
