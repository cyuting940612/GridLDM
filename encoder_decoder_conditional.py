"""
Conditional Transformer Autoencoder for Multi-Domain Time-Series Reconstruction.

This script trains a transformer-based autoencoder on multiple time-series domains
(EV, commercial load, wind, solar, transient voltage) while conditioning on BERT
text embeddings associated with each sample.

Main features
-------------
1. Variable-length time-series support with padding + masking.
2. Conditioning on token-level text embeddings of shape (B, L_text, C_text).
3. Per-domain normalization over valid time steps only.
4. Unified dataset builder for multiple domains and prompt styles.
5. Train / evaluation loop with checkpoint saving.

Expected input files
--------------------
Time-series arrays:
    data/ev/ev_train_X.npy
    data/commercial_load/comm_train_X_1.npy
    data/wind/wind_train_X.npy
    data/solar/solar_train_X.npy
    data/transient_voltage/trans_train_X_1.npy

Text embedding arrays:
    embedding/ev/ev_train_labels_1_emb.npy
    embedding/ev/ev_train_labels_2_emb.npy
    embedding/ev/ev_train_labels_3_emb.npy
    embedding/commercial_load/comm_train_labels_1_emb.npy
    embedding/commercial_load/comm_train_labels_2_emb.npy
    embedding/commercial_load/comm_train_labels_3_emb.npy
    embedding/wind/wind_train_labels_1_emb.npy
    embedding/wind/wind_train_labels_2_emb.npy
    embedding/wind/wind_train_labels_3_emb.npy
    embedding/solar/solar_train_labels_1_emb.npy
    embedding/solar/solar_train_labels_2_emb.npy
    embedding/solar/solar_train_labels_3_emb.npy
    embedding/transient_voltage/trans_train_labels_1_emb.npy
    embedding/transient_voltage/trans_train_labels_2_emb.npy
    embedding/transient_voltage/trans_train_labels_3_emb.npy
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the conditional autoencoder."""

    max_seq_len: int = 96
    d_model: int = 64
    latent_dim: int = 24
    nhead: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    text_token_dim: int = 768


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    train_fraction: float = 0.8
    random_seed: int = 42
    checkpoint_path: str = "cond_autoencoder.pt"


@dataclass(frozen=True)
class DomainSpec:
    """
    Describes one real data domain.

    Each domain has one time-series array and one or more prompt-style embedding files.
    The same raw time-series array is paired with each prompt-style embedding set.
    """

    name: str
    signal_path: str
    valid_length: int
    target_length: int
    embedding_paths: Tuple[str, ...]


DOMAIN_SPECS: Tuple[DomainSpec, ...] = (
    DomainSpec(
        name="ev",
        signal_path="data/ev/ev_train_X.npy",
        valid_length=24,
        target_length=96,
        embedding_paths=(
            "embedding/ev/ev_train_labels_1_emb.npy",
            "embedding/ev/ev_train_labels_2_emb.npy",
            "embedding/ev/ev_train_labels_3_emb.npy",
        ),
    ),
    DomainSpec(
        name="commercial_load",
        signal_path="data/commercial_load/comm_train_X_1.npy",
        valid_length=96,
        target_length=96,
        embedding_paths=(
            "embedding/commercial_load/comm_train_labels_1_emb.npy",
            "embedding/commercial_load/comm_train_labels_2_emb.npy",
            "embedding/commercial_load/comm_train_labels_3_emb.npy",
        ),
    ),
    DomainSpec(
        name="wind",
        signal_path="data/wind/wind_train_X.npy",
        valid_length=24,
        target_length=96,
        embedding_paths=(
            "embedding/wind/wind_train_labels_1_emb.npy",
            "embedding/wind/wind_train_labels_2_emb.npy",
            "embedding/wind/wind_train_labels_3_emb.npy",
        ),
    ),
    DomainSpec(
        name="solar",
        signal_path="data/solar/solar_train_X.npy",
        valid_length=24,
        target_length=96,
        embedding_paths=(
            "embedding/solar/solar_train_labels_1_emb.npy",
            "embedding/solar/solar_train_labels_2_emb.npy",
            "embedding/solar/solar_train_labels_3_emb.npy",
        ),
    ),
    DomainSpec(
        name="transient_voltage",
        signal_path="data/transient_voltage/trans_train_X_1.npy",
        valid_length=81,
        target_length=96,
        embedding_paths=(
            "embedding/transient_voltage/trans_train_labels_1_emb.npy",
            "embedding/transient_voltage/trans_train_labels_2_emb.npy",
            "embedding/transient_voltage/trans_train_labels_3_emb.npy",
        ),
    ),
)


# =============================================================================
# Model components
# =============================================================================


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings.

        Args:
            x: Tensor of shape (T, B, d_model).

        Returns:
            Tensor of shape (T, B, d_model).
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class ConditionalTimeSeriesAutoencoder(nn.Module):
    """
    Transformer autoencoder for variable-length time-series with text conditioning.

    The text input is a sequence of token embeddings `(B, L_text, C_text)`.
    The sequence is mean-pooled and projected to the model dimension. The resulting
    conditioning vector is injected into both the encoder and decoder.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.d_model = config.d_model
        self.latent_dim = config.latent_dim
        self.text_token_dim = config.text_token_dim

        self.input_proj = nn.Linear(1, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.max_seq_len)
        self.decoder_pos = PositionalEncoding(config.d_model, max_len=config.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
        )

        self.to_latent = nn.Linear(config.d_model, config.latent_dim)
        self.latent_to_dmodel = nn.Linear(config.latent_dim, config.d_model)
        self.cond_pool = nn.Linear(config.text_token_dim, config.d_model)
        self.output_proj = nn.Linear(config.d_model, 1)
        self.query_tokens = nn.Parameter(torch.zeros(config.max_seq_len, 1, config.d_model))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def pool_condition(self, cond_seq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Mean-pool text token embeddings and project them to model space.

        Args:
            cond_seq: Optional tensor of shape (B, L_text, C_text).

        Returns:
            Tensor of shape (B, d_model), or None.
        """
        if cond_seq is None:
            return None
        pooled = cond_seq.mean(dim=1)
        return self.cond_pool(pooled)

    def encode(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        cond_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a padded time-series batch into latent codes.

        Args:
            x: Tensor of shape (B, T) or (B, T, 1).
            lengths: True sequence lengths, shape (B,).
            cond_seq: Optional text embeddings, shape (B, L_text, C_text).

        Returns:
            Latent tensor of shape (B, latent_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        x = self.input_proj(x)          # (B, T, d_model)
        x = x.transpose(0, 1)           # (T, B, d_model)
        x = self.pos_encoder(x)

        device = x.device
        idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        src_key_padding_mask = idx >= lengths.unsqueeze(1)  # True at padded positions

        cond_vec = self.pool_condition(cond_seq)
        if cond_vec is not None:
            x = x + cond_vec.unsqueeze(0)

        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        valid_mask = (~src_key_padding_mask).to(memory.dtype)
        valid_mask = valid_mask.transpose(0, 1).unsqueeze(-1)  # (T, B, 1)

        pooled = (memory * valid_mask).sum(dim=0) / lengths.to(memory.dtype).unsqueeze(-1)
        return self.to_latent(pooled)

    def decode(
        self,
        z: torch.Tensor,
        cond_seq: Optional[torch.Tensor] = None,
        target_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode latent vectors back to padded time-series.

        Args:
            z: Latent tensor of shape (B, latent_dim).
            cond_seq: Optional text embeddings, shape (B, L_text, C_text).
            target_len: Reconstructed sequence length. Defaults to max_seq_len.

        Returns:
            Reconstructed time-series of shape (B, target_len).
        """
        batch_size = z.size(0)
        target_len = target_len or self.max_seq_len
        if target_len > self.max_seq_len:
            raise ValueError(
                f"target_len={target_len} exceeds max_seq_len={self.max_seq_len}."
            )

        memory = self.latent_to_dmodel(z)
        cond_vec = self.pool_condition(cond_seq)
        if cond_vec is not None:
            memory = memory + cond_vec
        memory = memory.unsqueeze(0)  # (1, B, d_model)

        query = self.query_tokens[:target_len].expand(target_len, batch_size, self.d_model)
        query = self.decoder_pos(query)
        if cond_vec is not None:
            query = query + cond_vec.unsqueeze(0)

        decoded = self.decoder(tgt=query, memory=memory)
        decoded = self.output_proj(decoded)              # (T, B, 1)
        return decoded.transpose(0, 1).squeeze(-1)      # (B, T)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        cond_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full autoencoder forward pass.

        Returns:
            recon: Reconstructed padded time-series of shape (B, T).
            z: Latent tensor of shape (B, latent_dim).
        """
        target_len = x.shape[1]
        z = self.encode(x, lengths, cond_seq)
        recon = self.decode(z, cond_seq, target_len=target_len)
        return recon, z


# =============================================================================
# Dataset and losses
# =============================================================================


class MultiDomainTimeSeriesTextDataset(Dataset):
    """Simple dataset wrapper for normalized time-series and text embeddings."""

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.ts[index],
            self.lengths[index],
            self.text_embeds[index],
            self.domain_ids[index],
        )


def masked_mse_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared error computed only on valid time steps.

    Args:
        recon: Reconstructed signals, shape (B, T).
        target: Target signals, shape (B, T).
        lengths: True lengths, shape (B,).
    """
    batch_size, seq_len = target.shape
    idx = torch.arange(seq_len, device=target.device).unsqueeze(0).expand(batch_size, seq_len)
    valid_mask = (idx < lengths.unsqueeze(1)).to(target.dtype)

    squared_error = (recon - target) ** 2
    return (squared_error * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


# =============================================================================
# Data preparation helpers
# =============================================================================


def pad_array_to_length(array: np.ndarray, target_length: int) -> np.ndarray:
    """Right-pad a 2D array `(N, T)` with zeros until `target_length`."""
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}.")

    current_length = array.shape[1]
    if current_length > target_length:
        raise ValueError(
            f"Array length {current_length} exceeds target_length={target_length}."
        )
    if current_length == target_length:
        return array

    pad_width = target_length - current_length
    return np.pad(array, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)


def load_npy(path: str) -> np.ndarray:
    """Load a `.npy` file and fail early if it is missing."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    return np.load(file_path)


def normalize_domain(
    ts_raw: np.ndarray,
    lengths: np.ndarray,
    max_seq_len: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize one domain over valid time steps only.

    Args:
        ts_raw: Padded time-series array, shape (N, max_seq_len).
        lengths: True lengths, shape (N,).
        max_seq_len: Global padded length.

    Returns:
        ts_norm: Normalized array, same shape as input.
        mean: Domain mean over valid positions.
        std: Domain std over valid positions.
    """
    ts_tensor = torch.from_numpy(ts_raw).float()
    len_tensor = torch.from_numpy(lengths).long()

    idx = torch.arange(max_seq_len).unsqueeze(0).expand(ts_tensor.size(0), max_seq_len)
    valid = (idx < len_tensor.unsqueeze(1)).float()

    mean = (ts_tensor * valid).sum() / valid.sum()
    var = (((ts_tensor - mean) * valid) ** 2).sum() / valid.sum()
    std = torch.sqrt(var + 1e-6)

    ts_norm = ((ts_tensor - mean) / std) * valid
    return ts_norm.numpy().astype(np.float32), float(mean.item()), float(std.item())


def build_multidomain_arrays(
    domain_specs: Sequence[DomainSpec],
    max_seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, float], Dict[int, float], Dict[int, str]]:
    """
    Build concatenated arrays across all real domains and prompt styles.

    For each real domain, the same normalized signal array is paired with multiple
    prompt-style embedding arrays.

    Returns:
        ts_all:       (N_total, max_seq_len)
        lengths_all:  (N_total,)
        text_all:     (N_total, L_text, C_text)
        domain_ids:   (N_total,) with one integer per real domain
        domain_mean:  normalization mean keyed by domain id
        domain_std:   normalization std keyed by domain id
        domain_name:  mapping domain id -> domain string
    """
    ts_blocks: List[np.ndarray] = []
    length_blocks: List[np.ndarray] = []
    text_blocks: List[np.ndarray] = []
    domain_id_blocks: List[np.ndarray] = []

    domain_mean: Dict[int, float] = {}
    domain_std: Dict[int, float] = {}
    domain_name: Dict[int, str] = {}

    for domain_id, spec in enumerate(domain_specs):
        ts_raw = load_npy(spec.signal_path).astype(np.float32)
        ts_raw = pad_array_to_length(ts_raw, spec.target_length)

        lengths = np.full(ts_raw.shape[0], spec.valid_length, dtype=np.int64)
        ts_norm, mean, std = normalize_domain(ts_raw, lengths, max_seq_len)

        domain_mean[domain_id] = mean
        domain_std[domain_id] = std
        domain_name[domain_id] = spec.name

        for embedding_path in spec.embedding_paths:
            text_embeds = load_npy(embedding_path).astype(np.float32)
            if text_embeds.shape[0] != ts_norm.shape[0]:
                raise ValueError(
                    f"Sample count mismatch for domain '{spec.name}': "
                    f"signals={ts_norm.shape[0]}, text={text_embeds.shape[0]} "
                    f"from {embedding_path}"
                )

            ts_blocks.append(ts_norm)
            length_blocks.append(lengths)
            text_blocks.append(text_embeds)
            domain_id_blocks.append(np.full(lengths.shape[0], domain_id, dtype=np.int64))

    ts_all = np.concatenate(ts_blocks, axis=0)
    lengths_all = np.concatenate(length_blocks, axis=0)
    text_all = np.concatenate(text_blocks, axis=0)
    domain_ids = np.concatenate(domain_id_blocks, axis=0)

    return ts_all, lengths_all, text_all, domain_ids, domain_mean, domain_std, domain_name


def split_dataset(
    ts_all: np.ndarray,
    lengths_all: np.ndarray,
    text_all: np.ndarray,
    domain_ids: np.ndarray,
    train_fraction: float,
    seed: int,
) -> Tuple[MultiDomainTimeSeriesTextDataset, MultiDomainTimeSeriesTextDataset]:
    """Create reproducible train/test splits using a random permutation."""
    rng = np.random.default_rng(seed)
    num_samples = ts_all.shape[0]
    permutation = rng.permutation(num_samples)

    split_idx = int(train_fraction * num_samples)
    train_idx = permutation[:split_idx]
    test_idx = permutation[split_idx:]

    train_dataset = MultiDomainTimeSeriesTextDataset(
        ts_all[train_idx],
        lengths_all[train_idx],
        text_all[train_idx],
        domain_ids[train_idx],
    )
    test_dataset = MultiDomainTimeSeriesTextDataset(
        ts_all[test_idx],
        lengths_all[test_idx],
        text_all[test_idx],
        domain_ids[test_idx],
    )
    return train_dataset, test_dataset


# =============================================================================
# Training helpers
# =============================================================================


def get_device() -> torch.device:
    """Choose CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: ConditionalTimeSeriesAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average masked MSE."""
    model.train()
    total_loss = 0.0
    total_count = 0

    for ts_batch, len_batch, txt_batch, _ in loader:
        ts_batch = ts_batch.to(device)
        len_batch = len_batch.to(device)
        txt_batch = txt_batch.to(device)

        recon, _ = model(ts_batch, len_batch, txt_batch)
        loss = masked_mse_loss(recon, ts_batch, len_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = ts_batch.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: ConditionalTimeSeriesAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate average masked MSE on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_count = 0

    for ts_batch, len_batch, txt_batch, _ in loader:
        ts_batch = ts_batch.to(device)
        len_batch = len_batch.to(device)
        txt_batch = txt_batch.to(device)

        recon, _ = model(ts_batch, len_batch, txt_batch)
        loss = masked_mse_loss(recon, ts_batch, len_batch)

        batch_size = ts_batch.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def save_checkpoint(
    path: str,
    model: ConditionalTimeSeriesAutoencoder,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    domain_mean: Dict[int, float],
    domain_std: Dict[int, float],
    domain_name: Dict[int, str],
) -> None:
    """Save model weights, optimizer state, and normalization metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "domain_mean": domain_mean,
        "domain_std": domain_std,
        "domain_name": domain_name,
        "config": {
            "max_seq_len": model_config.max_seq_len,
            "d_model": model_config.d_model,
            "latent_dim": model_config.latent_dim,
            "nhead": model_config.nhead,
            "num_encoder_layers": model_config.num_encoder_layers,
            "num_decoder_layers": model_config.num_decoder_layers,
            "dim_feedforward": model_config.dim_feedforward,
            "dropout": model_config.dropout,
            "text_token_dim": model_config.text_token_dim,
        },
    }
    torch.save(checkpoint, path)


# =============================================================================
# Main script
# =============================================================================


def main() -> None:
    """Train the conditional autoencoder on all configured domains."""
    model_config = ModelConfig()
    train_config = TrainConfig()

    torch.manual_seed(train_config.random_seed)
    np.random.seed(train_config.random_seed)

    (
        ts_all,
        lengths_all,
        text_all,
        domain_ids,
        domain_mean,
        domain_std,
        domain_name,
    ) = build_multidomain_arrays(DOMAIN_SPECS, max_seq_len=model_config.max_seq_len)

    train_dataset, test_dataset = split_dataset(
        ts_all,
        lengths_all,
        text_all,
        domain_ids,
        train_fraction=train_config.train_fraction,
        seed=train_config.random_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
    )

    device = get_device()
    print(f"Using device: {device}")
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    print("Domain mapping:")
    for domain_id, name in domain_name.items():
        print(f"  {domain_id}: {name} | mean={domain_mean[domain_id]:.6f}, std={domain_std[domain_id]:.6f}")

    model = ConditionalTimeSeriesAutoencoder(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_test_loss = float("inf")
    for epoch in range(1, train_config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_loss = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f}"
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_checkpoint(
                path=train_config.checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                domain_mean=domain_mean,
                domain_std=domain_std,
                domain_name=domain_name,
            )
            print(f"  -> Saved improved checkpoint to {train_config.checkpoint_path}")

    print(f"Best test MSE: {best_test_loss:.6f}")


if __name__ == "__main__":
    main()
