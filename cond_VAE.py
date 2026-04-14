"""
Early-fusion conditional VAE for multi-domain 1D time-series generation.

Model summary
-------------
- Encoder:
    x + pooled text embeddings -> hidden state -> mu, logvar
- Decoder:
    z + pooled text embeddings -> hidden state -> transposed-convolution decoder -> x_hat
- Conditioning:
    text embeddings are assumed to be precomputed token embeddings of shape (B, M, 768)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    """Top-level cVAE training configuration."""

    batch_size: int = 256
    cond_dim: int = 768
    x_feat_dim: int = 256
    z_dim: int = 16
    hidden_dim: int = 512
    decoder_base_channels: int = 128
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    beta: float = 2.0
    save_path: str = "cvae.pt"
    save_every: int = 10
    t_out: int = 96
    train_fraction: float = 0.8
    shuffle_seed: int = 42


# =============================================================================
# Dataset
# =============================================================================


class MultiDomainTimeSeriesTextDataset(Dataset):
    """
    Dataset of normalized padded time-series, true lengths, text embeddings, and domain IDs.

    Parameters
    ----------
    ts_norm : np.ndarray
        Shape (N, T).
    lengths : np.ndarray
        Shape (N,).
    text_embeds : np.ndarray
        Shape (N, M, D).
    domain_ids : np.ndarray
        Shape (N,).
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
# Small utility modules
# =============================================================================


class GELU(nn.Module):
    """Thin module wrapper around GELU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


def make_valid_mask(lengths: torch.Tensor, t_out: int, device: torch.device) -> torch.Tensor:
    """
    Build a mask of shape (B, 1, T) with 1 on valid positions and 0 on padded positions.
    """
    idx = torch.arange(t_out, device=device)[None, :]
    mask = (idx < lengths[:, None]).float()
    return mask.unsqueeze(1)


class CondPool(nn.Module):
    """
    Pool condition tokens from shape (B, M, D) to (B, D) with LayerNorm + mean pooling.
    """

    def __init__(self, cond_dim: int = 768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cond_dim)

    def forward(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(cond_tokens).mean(dim=1)


# =============================================================================
# Encoder / decoder modules
# =============================================================================


class XEncoder1D(nn.Module):
    """
    Encode an input signal x of shape (B, 1, 96) into a compact feature vector.

    Temporal resolution:
        96 -> 48 -> 24 -> 12 -> pooled feature
    """

    def __init__(self, x_channels: int = 1, x_feat_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(x_channels, 64, kernel_size=3, padding=1),
            GELU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            GELU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            GELU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),
            GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
        )
        self.proj = nn.Linear(256, x_feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


class XDecoder1D(nn.Module):
    """
    Decode a hidden vector h of shape (B, in_dim) into x_hat of shape (B, 1, 96).

    Temporal growth:
        12 -> 24 -> 48 -> 96
    """

    def __init__(self, in_dim: int, base_ch: int = 128) -> None:
        super().__init__()
        self.init_t = 12
        self.fc = nn.Linear(in_dim, base_ch * self.init_t)

        self.up1 = nn.ConvTranspose1d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(base_ch, base_ch // 2, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose1d(base_ch // 2, base_ch // 4, kernel_size=4, stride=2, padding=1)

        self.norm1 = nn.GroupNorm(8, base_ch)
        self.norm2 = nn.GroupNorm(8, base_ch // 2)
        self.norm3 = nn.GroupNorm(8, base_ch // 4)
        self.act = GELU()

        self.out = nn.Conv1d(base_ch // 4, 1, kernel_size=3, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size = h.size(0)
        x = self.fc(h).view(batch_size, -1, self.init_t)

        x = self.act(self.norm1(self.up1(x)))
        x = self.act(self.norm2(self.up2(x)))
        x = self.act(self.norm3(self.up3(x)))

        return self.out(x)


# =============================================================================
# Conditional VAE
# =============================================================================


class ConditionalVAE1D(nn.Module):
    """
    Early-fusion conditional VAE for length-96 1D signals.

    Encoder
    -------
    x, cond_tokens -> pooled condition + encoded x -> mu, logvar

    Decoder
    -------
    z, cond_tokens -> pooled condition + latent -> x_hat
    """

    def __init__(
        self,
        x_channels: int = 1,
        cond_dim: int = 768,
        x_feat_dim: int = 256,
        z_dim: int = 64,
        hidden_dim: int = 512,
        decoder_base_channels: int = 128,
    ) -> None:
        super().__init__()
        self.pool = CondPool(cond_dim)
        self.x_encoder = XEncoder1D(x_channels=x_channels, x_feat_dim=x_feat_dim)

        self.encoder_mlp = nn.Sequential(
            nn.Linear(x_feat_dim + cond_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            GELU(),
        )
        self.to_mu = nn.Linear(hidden_dim, z_dim)
        self.to_logvar = nn.Linear(hidden_dim, z_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            GELU(),
        )
        self.decoder = XDecoder1D(in_dim=hidden_dim, base_ch=decoder_base_channels)

    def encode(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_cond = self.pool(cond_tokens)
        x_features = self.x_encoder(x)
        hidden = self.encoder_mlp(torch.cat([x_features, pooled_cond], dim=1))
        mu = self.to_mu(hidden)
        logvar = self.to_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        pooled_cond = self.pool(cond_tokens)
        hidden = self.decoder_input(torch.cat([z, pooled_cond], dim=1))
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, cond_tokens)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, cond_tokens)
        return x_hat, mu, logvar


# =============================================================================
# Loss
# =============================================================================


def cvae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute masked reconstruction loss and KL divergence.

    Parameters
    ----------
    x_hat : torch.Tensor
        Reconstructed signal, shape (B, 1, T).
    x : torch.Tensor
        Target signal, shape (B, 1, T).
    mu : torch.Tensor
        Latent mean, shape (B, z_dim).
    logvar : torch.Tensor
        Latent log-variance, shape (B, z_dim).
    valid_mask : torch.Tensor | None
        Shape (B, 1, T). Ones on valid positions, zeros on padded positions.
    beta : float
        Weight of the KL term.

    Returns
    -------
    total_loss, recon_loss, kl_loss
    """
    if valid_mask is None:
        recon_loss = F.mse_loss(x_hat, x)
    else:
        mse = (x_hat - x) ** 2
        mse = mse * valid_mask
        recon_loss = mse.sum() / (valid_mask.sum() + 1e-8)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


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
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize one padded time-series variant using valid positions only.

    Returns
    -------
    signal_norm : (N, T)
    stats : dict
        Contains variant_id, mean, std
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
    return signal_norm.numpy().astype(np.float32), stats


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
        signal_norm, stats = normalize_variant(signal, lengths, variant_id)

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
    model: ConditionalVAE1D,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    """Save a cVAE checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "opt_state_dict": optimizer.state_dict(),
        "beta": config.beta,
    }
    if extra_metadata is not None:
        checkpoint.update(extra_metadata)
    torch.save(checkpoint, config.save_path)
    print(f"Saved checkpoint: {config.save_path}")


# =============================================================================
# Training
# =============================================================================


def train_cvae(
    train_loader: DataLoader,
    model: ConditionalVAE1D,
    config: TrainingConfig,
    device: torch.device,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Tuple[ConditionalVAE1D, torch.optim.Optimizer]:
    """
    Train the cVAE on the provided loader.

    Notes
    -----
    - Padded positions are masked in the reconstruction loss.
    - The training loop matches the original script and does not use a validation set.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_samples = 0

        for ts_batch, len_batch, txt_batch, _domain_batch in train_loader:
            ts_batch = ts_batch.to(device)
            len_batch = len_batch.to(device)
            cond_tokens = txt_batch.to(device)

            x = ts_batch.unsqueeze(1)
            valid_mask = make_valid_mask(len_batch, config.t_out, device)
            x = x * valid_mask

            x_hat, mu, logvar = model(x, cond_tokens)
            loss, recon, kl = cvae_loss(
                x_hat,
                x,
                mu,
                logvar,
                valid_mask=valid_mask,
                beta=config.beta,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon += recon.item() * batch_size
            total_kl += kl.item() * batch_size
            n_samples += batch_size

        print(
            f"[Epoch {epoch + 1:03d}] "
            f"loss={total_loss / max(n_samples, 1):.6f} "
            f"recon={total_recon / max(n_samples, 1):.6f} "
            f"kl={total_kl / max(n_samples, 1):.6f}"
        )

        if config.save_every > 0 and ((epoch + 1) % config.save_every == 0):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch + 1,
                extra_metadata=extra_metadata,
            )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=config.num_epochs,
        extra_metadata=extra_metadata,
    )
    return model, optimizer


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

    model = ConditionalVAE1D(
        x_channels=1,
        cond_dim=config.cond_dim,
        x_feat_dim=config.x_feat_dim,
        z_dim=config.z_dim,
        hidden_dim=config.hidden_dim,
        decoder_base_channels=config.decoder_base_channels,
    )

    metadata = {
        "normalization_stats": norm_stats,
        "training_config": {
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "beta": config.beta,
            "save_every": config.save_every,
            "train_fraction": config.train_fraction,
            "shuffle_seed": config.shuffle_seed,
        },
    }

    train_cvae(
        train_loader=train_loader,
        model=model,
        config=config,
        device=device,
        extra_metadata=metadata,
    )


if __name__ == "__main__":
    main()
