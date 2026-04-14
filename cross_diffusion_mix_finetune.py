from __future__ import annotations

"""Fine-tune a pretrained latent diffusion model on a new time-series domain.

This script continues training a pretrained 1D latent diffusion model using:

1. A frozen conditional time-series autoencoder (AE) that maps padded time series
   into latent vectors.
2. Precomputed text-token embeddings used as conditioning signals.
3. A pretrained diffusion model operating in the AE latent space.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from encoder_decoder_conditional import ConditionalTimeSeriesAutoencoder
from cross_diffusion_mix import (
    DiffusionConfig,
    GaussianDiffusion,
    UNet1D,
    UNet1DConfig,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class AutoencoderConfig:
    """Hyperparameters used to rebuild the frozen conditional autoencoder."""

    max_seq_len: int = 96
    d_model: int = 64
    latent_dim: int = 24
    nhead: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    text_token_dim: int = 768


@dataclass
class UNetConfig:
    """Hyperparameters used to rebuild the pretrained diffusion U-Net."""

    in_channels: int = 1
    base_channels: int = 128
    channel_mults: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    num_heads: int = 4
    t_emb_dim: int = 256
    dropout: float = 0.0
    cond_dim: int = 768


@dataclass
class FinetuneConfig:
    """End-to-end configuration for diffusion fine-tuning."""

    ae_checkpoint: str = "cond_autoencoder.pt"
    diffusion_checkpoint: str = "gridldm_diffusion.pt"

    new_data_path: str = "data/transient_voltage/trans_labels_fine_tune.npy"
    new_text_path: str = "embedding/transient_voltage/trans_labels_fine_tune_emb.npy"

    output_checkpoint: str = "gridldm_diffusion_trans_finetuned.pt"

    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100

    subset_fraction: float = 0.10
    seed: int = 42

    p_uncond: float = 0.10
    null_fill_value: float = 0.0


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class NewDomainDataset(Dataset):
    """Dataset of normalized time series and their text embeddings.

    Args:
        ts_norm: Array of shape ``(N, max_seq_len)`` containing normalized and
            padded time-series samples.
        lengths: Array of shape ``(N,)`` containing the true unpadded lengths.
        text_embeds: Array of shape ``(N, L, D)`` containing text-token
            embeddings compatible with the pretrained model.
    """

    def __init__(
        self,
        ts_norm: np.ndarray,
        lengths: np.ndarray,
        text_embeds: np.ndarray,
    ) -> None:
        super().__init__()
        self.ts = torch.from_numpy(ts_norm).float()
        self.lengths = torch.from_numpy(lengths).long()
        self.text_embeds = torch.from_numpy(text_embeds).float()

    def __len__(self) -> int:
        return self.ts.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ts[idx], self.lengths[idx], self.text_embeds[idx]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def select_device() -> torch.device:
    """Return the best available device.

    Preference order:
    1. CUDA
    2. Apple MPS
    3. CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set NumPy and PyTorch seeds for reproducibility."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_numpy_array(path: str) -> np.ndarray:
    """Load a NumPy array and cast it to float32 when appropriate."""

    array = np.load(path)
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    return array


def pad_time_series(
    x: np.ndarray,
    max_seq_len: int,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a batch of variable-length but same-width samples to ``max_seq_len``.

    Args:
        x: Input array of shape ``(N, T)``.
        max_seq_len: Target padded length expected by the autoencoder.
        pad_value: Constant value used for right-padding.

    Returns:
        A tuple ``(x_padded, lengths)`` where:
        - ``x_padded`` has shape ``(N, max_seq_len)``
        - ``lengths`` has shape ``(N,)`` and stores the original length ``T``
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x to have shape (N, T), got {x.shape}.")

    n_samples, seq_len = x.shape
    if seq_len > max_seq_len:
        raise ValueError(
            f"Input sequence length {seq_len} exceeds max_seq_len={max_seq_len}."
        )

    pad_len = max_seq_len - seq_len
    x_padded = np.pad(
        x,
        pad_width=((0, 0), (0, pad_len)),
        mode="constant",
        constant_values=pad_value,
    ).astype(np.float32)
    lengths = np.full((n_samples,), seq_len, dtype=np.int64)
    return x_padded, lengths


def normalize_valid_region(
    x_padded: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Normalize only the valid, unpadded region of each sequence.

    The mean and standard deviation are computed over all valid entries in the
    new domain. Padded positions remain zero after normalization.

    Args:
        x_padded: Array of shape ``(N, max_seq_len)``.
        lengths: Array of shape ``(N,)`` with true sequence lengths.

    Returns:
        A tuple ``(x_norm, mean, std)``.
    """

    ts_t = torch.from_numpy(x_padded).float()
    len_t = torch.from_numpy(lengths).long()

    n_samples, max_seq_len = ts_t.shape
    idxs = torch.arange(max_seq_len).unsqueeze(0).expand(n_samples, max_seq_len)
    valid = (idxs < len_t.unsqueeze(1)).float()

    mean = (ts_t * valid).sum() / valid.sum().clamp_min(1.0)
    var = (((ts_t - mean) * valid) ** 2).sum() / valid.sum().clamp_min(1.0)
    std = torch.sqrt(var + 1e-6)

    x_norm = ((ts_t - mean) / std) * valid
    return x_norm.numpy().astype(np.float32), float(mean.item()), float(std.item())


def subsample_dataset(
    ts_norm: np.ndarray,
    lengths: np.ndarray,
    text_embeds: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly keep a fraction of samples from a dataset.

    At least one sample is kept when the dataset is non-empty.
    """

    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"subset_fraction must be in (0, 1], got {fraction}.")

    n_total = ts_norm.shape[0]
    if n_total == 0:
        raise ValueError("Cannot subsample an empty dataset.")

    if fraction == 1.0:
        return ts_norm, lengths, text_embeds

    rng = np.random.default_rng(seed)
    k = max(1, int(round(fraction * n_total)))
    indices = rng.choice(n_total, size=k, replace=False)
    return ts_norm[indices], lengths[indices], text_embeds[indices]


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def build_autoencoder(cfg: AutoencoderConfig) -> ConditionalTimeSeriesAutoencoder:
    """Instantiate the conditional autoencoder from configuration."""

    return ConditionalTimeSeriesAutoencoder(
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        latent_dim=cfg.latent_dim,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        text_token_dim=cfg.text_token_dim,
    )


def load_frozen_autoencoder(
    cfg: AutoencoderConfig,
    checkpoint_path: str,
    device: torch.device,
) -> ConditionalTimeSeriesAutoencoder:
    """Load the pretrained AE checkpoint and freeze all AE parameters."""

    model = build_autoencoder(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def build_unet(cfg: UNetConfig) -> UNet1D:
    """Instantiate the U-Net backbone used by the diffusion model."""

    unet_cfg = UNet1DConfig(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        num_heads=cfg.num_heads,
        t_emb_dim=cfg.t_emb_dim,
        dropout=cfg.dropout,
    )
    return UNet1D(
        unet_cfg,
        cond_dim=cfg.cond_dim,
        use_self_attn=True,
        use_cross_attn=True,
    )


def load_pretrained_diffusion(
    diffusion_checkpoint: str,
    unet_cfg: UNetConfig,
    device: torch.device,
) -> tuple[GaussianDiffusion, dict]:
    """Rebuild the diffusion model from checkpoint config and load weights."""

    checkpoint = torch.load(diffusion_checkpoint, map_location=device)
    cfg_dict = checkpoint["config"]

    diff_cfg = DiffusionConfig(
        timesteps=cfg_dict["timesteps"],
        beta_start=cfg_dict["beta_start"],
        beta_end=cfg_dict["beta_end"],
        objective=cfg_dict.get("objective", "eps"),
    )

    unet = build_unet(unet_cfg).to(device)
    diffusion = GaussianDiffusion(unet, diff_cfg).to(device)
    diffusion.load_state_dict(checkpoint["model_state_dict"])
    return diffusion, checkpoint


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_one_epoch(
    diffusion: GaussianDiffusion,
    autoencoder: ConditionalTimeSeriesAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    p_uncond: float,
    null_fill_value: float,
) -> float:
    """Train the diffusion model for one epoch.

    Conditioning dropout is applied to implement classifier-free guidance
    training. The autoencoder remains frozen and is used only to encode latents.
    """

    diffusion.train()
    running_loss = 0.0
    n_samples = 0

    for ts_batch, len_batch, txt_batch in loader:
        ts_batch = ts_batch.to(device)
        len_batch = len_batch.to(device)
        txt_batch = txt_batch.to(device)

        with torch.no_grad():
            z = autoencoder.encode(ts_batch, len_batch, txt_batch)  # (B, latent_dim)

        x0 = z.unsqueeze(1)  # (B, 1, latent_dim)

        uncond_mask = torch.rand(x0.size(0), device=device) < p_uncond
        cond_tokens = txt_batch.clone()
        if uncond_mask.any():
            cond_tokens[uncond_mask] = null_fill_value

        loss = diffusion.loss(x0, cond_tokens)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = x0.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / max(n_samples, 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(
    cfg: Optional[FinetuneConfig] = None,
    ae_cfg: Optional[AutoencoderConfig] = None,
    unet_cfg: Optional[UNetConfig] = None,
) -> None:
    """Run latent diffusion fine-tuning on a new domain."""

    cfg = cfg or FinetuneConfig()
    ae_cfg = ae_cfg or AutoencoderConfig()
    unet_cfg = unet_cfg or UNetConfig(cond_dim=ae_cfg.text_token_dim)

    set_seed(cfg.seed)
    device = select_device()
    print(f"Using device: {device}")

    # Load models.
    autoencoder = load_frozen_autoencoder(ae_cfg, cfg.ae_checkpoint, device)
    print(f"Loaded and froze AE from: {cfg.ae_checkpoint}")

    diffusion, base_checkpoint = load_pretrained_diffusion(
        cfg.diffusion_checkpoint,
        unet_cfg,
        device,
    )
    print(f"Loaded diffusion weights from: {cfg.diffusion_checkpoint}")

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Load and preprocess new-domain training pairs.
    x_new = load_numpy_array(cfg.new_data_path).astype(np.float32)
    text_new = load_numpy_array(cfg.new_text_path).astype(np.float32)

    if x_new.shape[0] != text_new.shape[0]:
        raise ValueError(
            "Time-series and text arrays must have the same number of samples: "
            f"got {x_new.shape[0]} and {text_new.shape[0]}."
        )

    x_padded, lengths = pad_time_series(x_new, ae_cfg.max_seq_len)
    ts_norm, mean_new, std_new = normalize_valid_region(x_padded, lengths)

    print(f"New domain data shape: {x_new.shape}")
    print(f"Text embedding shape:   {text_new.shape}")
    print(f"New domain mean: {mean_new:.6f}")
    print(f"New domain std:  {std_new:.6f}")

    ts_sub, len_sub, text_sub = subsample_dataset(
        ts_norm,
        lengths,
        text_new,
        fraction=cfg.subset_fraction,
        seed=cfg.seed,
    )
    print(f"Using {ts_sub.shape[0]} / {ts_norm.shape[0]} samples for fine-tuning.")

    dataset = NewDomainDataset(ts_sub, len_sub, text_sub)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Fine-tune diffusion.
    for epoch in range(cfg.num_epochs):
        epoch_loss = train_one_epoch(
            diffusion=diffusion,
            autoencoder=autoencoder,
            loader=loader,
            optimizer=optimizer,
            device=device,
            p_uncond=cfg.p_uncond,
            null_fill_value=cfg.null_fill_value,
        )
        print(f"[FT Epoch {epoch + 1}/{cfg.num_epochs}] loss: {epoch_loss:.6f}")

    # Save checkpoint.
    output_path = Path(cfg.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": diffusion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": diffusion.cfg.__dict__,
            "finetune_config": asdict(cfg),
            "base_checkpoint": cfg.diffusion_checkpoint,
            "mean_new": mean_new,
            "std_new": std_new,
            "subset_fraction": cfg.subset_fraction,
            "source_data_path": cfg.new_data_path,
            "source_text_path": cfg.new_text_path,
            "pretrained_diffusion_config": base_checkpoint.get("config", {}),
        },
        output_path,
    )
    print(f"Saved fine-tuned diffusion checkpoint to: {output_path}")


if __name__ == "__main__":
    main()
