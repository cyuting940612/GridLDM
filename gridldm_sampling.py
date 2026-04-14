"""
Unified multi-domain sampling script for text-conditioned time-series generation.

This script combines the separate domain-specific sampling scripts for:
- wind
- solar
- ev
- commercial_load
- transient_voltage

What it does
------------
1. Loads one shared conditional autoencoder and one shared diffusion model.
2. Uses domain-specific regex parsers to read prompt JSONL files.
3. Computes normalization statistics from each domain's training data.
4. Generates synthetic samples from text embeddings with classifier-free guidance.
5. Saves metadata and generated CSVs in domain/style-specific output folders.

Notes
-----
- EV, wind, and solar generate one CSV per style with all conditions repeated
  `n_samples_per_condition` times.
- transient_voltage also expands metadata to keep one row per generated sample.
- commercial_load preserves the original subset-based behavior:
  weekday / weekend / summer / winter / per-user.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from cross_diffusion_mix import DiffusionConfig, GaussianDiffusion, UNet1D, UNet1DConfig
from encoder_decoder_conditional import ConditionalTimeSeriesAutoencoder


# =============================================================================
# Global constants
# =============================================================================

LATENT_DIM = 24
LATENT_CHANNELS = 1
OUTPUT_LENGTH = 96
TEXT_DIM = 768


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class DomainConfig:
    """Configuration for one domain."""

    name: str
    style_config: Dict[int, Dict[str, str]]
    output_root: str
    train_x_path: str
    true_length: int
    n_samples_per_condition: int = 10
    lengths_path: Optional[str] = None


def default_domain_configs() -> Dict[str, DomainConfig]:
    """Return the default per-domain configuration used by the unified sampler."""
    return {
        "wind": DomainConfig(
            name="wind",
            style_config={
                1: {
                    "labels_path": "prompts/wind/wind_test_labels_1.jsonl",
                    "emb_path": "embedding/wind/wind_test_labels_1_emb.npy",
                },
                2: {
                    "labels_path": "prompts/wind/wind_test_labels_2.jsonl",
                    "emb_path": "embedding/wind/wind_test_labels_2_emb.npy",
                },
                3: {
                    "labels_path": "prompts/wind/wind_test_labels_3.jsonl",
                    "emb_path": "embedding/wind/wind_test_labels_3_emb.npy",
                },
            },
            output_root="samples_wind",
            train_x_path="data/wind/wind_train_X.npy",
            true_length=24,
        ),
        "solar": DomainConfig(
            name="solar",
            style_config={
                1: {
                    "labels_path": "prompts/solar/solar_test_labels_1.jsonl",
                    "emb_path": "embedding/solar/solar_test_labels_1_emb.npy",
                },
                2: {
                    "labels_path": "prompts/solar/solar_test_labels_2.jsonl",
                    "emb_path": "embedding/solar/solar_test_labels_2_emb.npy",
                },
                3: {
                    "labels_path": "prompts/solar/solar_test_labels_3.jsonl",
                    "emb_path": "embedding/solar/solar_test_labels_3_emb.npy",
                },
            },
            output_root="samples_solar",
            train_x_path="data/solar/solar_train_X.npy",
            true_length=24,
        ),
        "ev": DomainConfig(
            name="ev",
            style_config={
                1: {
                    "labels_path": "prompts/ev/ev_test_labels_1.jsonl",
                    "emb_path": "embedding/ev/ev_test_labels_1_emb.npy",
                },
                2: {
                    "labels_path": "prompts/ev/ev_test_labels_2.jsonl",
                    "emb_path": "embedding/ev/ev_test_labels_2_emb.npy",
                },
                3: {
                    "labels_path": "prompts/ev/ev_test_labels_3.jsonl",
                    "emb_path": "embedding/ev/ev_test_labels_3_emb.npy",
                },
            },
            output_root="samples_ev",
            train_x_path="data/ev/ev_train_X.npy",
            true_length=24,
        ),
        "commercial_load": DomainConfig(
            name="commercial_load",
            style_config={
                1: {
                    "labels_path": "prompts/commercial_load/comm_test_labels_1.jsonl",
                    "emb_path": "embedding/commercial_load/comm_test_labels_1_emb.npy",
                },
                2: {
                    "labels_path": "prompts/commercial_load/comm_test_labels_2.jsonl",
                    "emb_path": "embedding/commercial_load/comm_test_labels_2_emb.npy",
                },
                3: {
                    "labels_path": "prompts/commercial_load/comm_test_labels_3.jsonl",
                    "emb_path": "embedding/commercial_load/comm_test_labels_3_emb.npy",
                },
            },
            output_root="samples_comm",
            train_x_path="data/commercial_load/comm_train_X.npy",
            true_length=96,
        ),
        "transient_voltage": DomainConfig(
            name="transient_voltage",
            style_config={
                1: {
                    "labels_path": "prompts/transient_voltage/trans_test_labels_1.jsonl",
                    "emb_path": "embedding/transient_voltage/trans_test_labels_1_emb.npy",
                },
                2: {
                    "labels_path": "prompts/transient_voltage/trans_test_labels_2.jsonl",
                    "emb_path": "embedding/transient_voltage/trans_test_labels_2_emb.npy",
                },
                3: {
                    "labels_path": "prompts/transient_voltage/trans_test_labels_3.jsonl",
                    "emb_path": "embedding/transient_voltage/trans_test_labels_3_emb.npy",
                },
            },
            output_root="samples_trans",
            train_x_path="data/transient_voltage/trans_train_X_1.npy",
            true_length=81,
        ),
    }


# =============================================================================
# Regex patterns
# =============================================================================

FLOAT_RE = r"[-+]?\d+(?:\.\d+)?"

PAT_WIND_BY_STYLE = {
    1: re.compile(
        rf"Daily wind power for (?P<zone>.+?) on (?P<date>\d{{4}}-\d{{2}}-\d{{2}}) "
        rf"with wind speed stats: min (?P<min_ws>{FLOAT_RE}), "
        rf"max (?P<max_ws>{FLOAT_RE}), median (?P<median_ws>{FLOAT_RE})\. "
        rf"trend:(?P<trend_text>[^.]+) and (?P<peak_text>[^.]+)\.?"
    ),
    2: re.compile(
        rf"For (?P<zone>.+?) on (?P<date>\d{{4}}-\d{{2}}-\d{{2}}), "
        rf"wind speed ranges from (?P<min_ws>{FLOAT_RE}) to (?P<max_ws>{FLOAT_RE}) "
        rf"with median (?P<median_ws>{FLOAT_RE}), showing "
        rf"(?P<trend_text>[^,]+) and (?P<peak_text>[^.]+)\.?"
    ),
    3: re.compile(
        rf"(?P<trend_text>[^,]+) and (?P<peak_text>[^,]+) "
        rf"in daily wind power for (?P<zone>.+?) on (?P<date>\d{{4}}-\d{{2}}-\d{{2}}), "
        rf"with wind speed min (?P<min_ws>{FLOAT_RE}), max (?P<max_ws>{FLOAT_RE}), "
        rf"median (?P<median_ws>{FLOAT_RE})\.?"
    ),
}

PAT_SOLAR_BY_STYLE = {
    1: re.compile(
        rf"Daily solar power for (?P<zone>.+?) on (?P<date>\d{{4}}-\d{{2}}-\d{{2}}) "
        rf"with hourly GHI pattern: Min (?P<min_ghi>{FLOAT_RE}), max (?P<max_ghi>{FLOAT_RE}), "
        rf"mean (?P<mean_ghi>{FLOAT_RE})\. "
        rf"Brightness (?P<brightness>[^,]+), variability (?P<variability>[^,]+), "
        rf"peak (?P<peak_period>[^,]+), daylight (?P<sunrise_hour>\d{{2}})-(?P<sunset_hour>\d{{2}})\.?"
    ),
    2: re.compile(
        rf"For solar generation in (?P<zone>.+?) on (?P<date>\d{{4}}-\d{{2}}-\d{{2}}), "
        rf"hourly GHI ranges from (?P<min_ghi>{FLOAT_RE}) to (?P<max_ghi>{FLOAT_RE}) "
        rf"with mean (?P<mean_ghi>{FLOAT_RE}), brightness (?P<brightness>[^,]+), "
        rf"variability (?P<variability>[^,]+), peak (?P<peak_period>[^,]+), "
        rf"daylight (?P<sunrise_hour>\d{{2}})-(?P<sunset_hour>\d{{2}})\.?"
    ),
    3: re.compile(
        rf"(?P<brightness>[^,]+) and (?P<variability>[^,]+) hourly GHI "
        rf"with peak (?P<peak_period>[^ ]+) and daylight (?P<sunrise_hour>\d{{2}})-(?P<sunset_hour>\d{{2}}), "
        rf"with min (?P<min_ghi>{FLOAT_RE}), max (?P<max_ghi>{FLOAT_RE}), "
        rf"mean (?P<mean_ghi>{FLOAT_RE}) for daily solar power in (?P<zone>.+?) "
        rf"on (?P<date>\d{{4}}-\d{{2}}-\d{{2}})\.?"
    ),
}

PAT_EV_BY_STYLE = {
    1: re.compile(
        r"EV charging profile on (?P<date>\d{4}-\d{2}-\d{2}) "
        r"\((?P<week_type>weekday|weekend)\) for "
        r"\((?P<load_type>[a-zA-Z0-9_ ]+) charging in allocation (?P<allocation>\d+)\)\.?"
    ),
    2: re.compile(
        r"On (?P<date>\d{4}-\d{2}-\d{2}) "
        r"\((?P<week_type>weekday|weekend)\), "
        r"(?P<load_type>[a-zA-Z0-9_ ]+) charging in allocation (?P<allocation>\d+) "
        r"shows the daily EV charging pattern\.?"
    ),
    3: re.compile(
        r"(?P<load_type>[a-zA-Z0-9_ ]+) charging in allocation (?P<allocation>\d+) "
        r"load curve on (?P<date>\d{4}-\d{2}-\d{2}) "
        r"\((?P<week_type>weekday|weekend)\)\.?"
    ),
}

PAT_COMMERCIAL_BY_STYLE = {
    1: re.compile(
        r"Energy usage profile of user (?P<user_id>\d+) on "
        r"(?P<date>\d{4}-\d{2}-\d{2}) "
        r"\((?P<day_type>weekday|weekend)\)\."
    ),
    2: re.compile(
        r"User (?P<user_id>\d+)'s energy usage on "
        r"(?P<date>\d{4}-\d{2}-\d{2}) "
        r"\((?P<day_type>weekday|weekend)\)\."
    ),
    3: re.compile(
        r"Load curve for user (?P<user_id>\d+) on "
        r"(?P<date>\d{4}-\d{2}-\d{2}) "
        r"\((?P<day_type>weekday|weekend)\)\."
    ),
}

PAT_TRANSIENT_BY_STYLE = {
    1: re.compile(
        r"Transient-state voltage signal under a (?P<event_type>.+?) event at bus "
        r"(?P<bus>\d+) in the (?P<network>.+?)\.?$",
        flags=re.IGNORECASE,
    ),
    2: re.compile(
        r"Voltage transient in the (?P<network>.+?) caused by a (?P<event_type>.+?) "
        r"at bus (?P<bus>\d+)\.?$",
        flags=re.IGNORECASE,
    ),
    3: re.compile(
        r"In the (?P<network>.+?), this sample shows a voltage transient following "
        r"a (?P<event_type>.+?) at bus (?P<bus>\d+)\.?$",
        flags=re.IGNORECASE,
    ),
}


# =============================================================================
# Small utilities
# =============================================================================


def select_device() -> torch.device:
    """Choose CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bytes_to_mb(x: int) -> str:
    """Convert bytes to a human-readable megabytes string."""
    return f"{x / 1024 / 1024:.1f} MB"


def load_labels_from_jsonl(path: str) -> List[str]:
    """
    Load text labels from a JSONL file.

    Each line may be:
    - a JSON string
    - a JSON object with key 'label' or 'text'
    - raw text
    """
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    labels.append(obj)
                elif isinstance(obj, dict):
                    if "label" in obj:
                        labels.append(obj["label"])
                    elif "text" in obj:
                        labels.append(obj["text"])
                    else:
                        found = False
                        for value in obj.values():
                            if isinstance(value, str):
                                labels.append(value)
                                found = True
                                break
                        if not found:
                            raise ValueError(f"Cannot find label text in JSON object: {obj}")
                else:
                    labels.append(str(obj))
            except json.JSONDecodeError:
                labels.append(line)
    return labels


def pad_to_length(x: np.ndarray, target_length: int) -> np.ndarray:
    """Right-pad a 2D array of shape (N, T) to (N, target_length)."""
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}")
    n, t = x.shape
    if t > target_length:
        raise ValueError(f"Input length {t} exceeds target length {target_length}.")
    if t == target_length:
        return x.astype(np.float32)
    return np.pad(x, ((0, 0), (0, target_length - t)), mode="constant", constant_values=0).astype(np.float32)


def compute_masked_stats(train_x_path: str, true_length: int, max_seq_len: int = OUTPUT_LENGTH) -> Tuple[float, float]:
    """
    Compute mean and std over valid positions only for padded domains.
    """
    x = np.load(train_x_path).astype(np.float32)
    x = pad_to_length(x, max_seq_len)
    lengths = np.full(x.shape[0], true_length, dtype=np.int64)

    x_t = torch.from_numpy(x).float()
    len_t = torch.from_numpy(lengths).long()
    idx = torch.arange(max_seq_len).unsqueeze(0).expand(x_t.size(0), max_seq_len)
    valid = (idx < len_t.unsqueeze(1)).float()

    mean = (x_t * valid).sum() / valid.sum()
    var = (((x_t - mean) * valid) ** 2).sum() / valid.sum()
    std = torch.sqrt(var + 1e-6)
    return mean.item(), std.item()


def compute_unmasked_stats(train_x_path: str) -> Tuple[float, float]:
    """Compute mean and std over all positions for full-length domains."""
    x = np.load(train_x_path).astype(np.float32)
    x_t = torch.from_numpy(x).float()
    return x_t.mean().item(), x_t.std().item()


def compute_domain_stats(config: DomainConfig) -> Tuple[float, float]:
    """Compute normalization statistics for one domain."""
    if config.true_length < OUTPUT_LENGTH:
        return compute_masked_stats(config.train_x_path, true_length=config.true_length, max_seq_len=OUTPUT_LENGTH)
    return compute_unmasked_stats(config.train_x_path)


# =============================================================================
# Label parsing and metadata
# =============================================================================


def parse_wind_label_by_style(label: str, style_id: int) -> Optional[Dict[str, object]]:
    match = PAT_WIND_BY_STYLE[style_id].fullmatch(label.strip())
    if match is None:
        return None
    record = match.groupdict()
    record["zone"] = record["zone"].strip()
    record["trend_text"] = record["trend_text"].strip()
    record["peak_text"] = record["peak_text"].strip()
    record["min_ws"] = float(record["min_ws"])
    record["max_ws"] = float(record["max_ws"])
    record["median_ws"] = float(record["median_ws"])
    record["date"] = pd.to_datetime(record["date"])
    return record


def parse_solar_label_by_style(label: str, style_id: int) -> Optional[Dict[str, object]]:
    match = PAT_SOLAR_BY_STYLE[style_id].fullmatch(label.strip())
    if match is None:
        return None
    record = match.groupdict()
    record["zone"] = record["zone"].strip()
    record["brightness"] = record["brightness"].strip()
    record["variability"] = record["variability"].strip()
    record["peak_period"] = record["peak_period"].strip()
    record["min_ghi"] = float(record["min_ghi"])
    record["max_ghi"] = float(record["max_ghi"])
    record["mean_ghi"] = float(record["mean_ghi"])
    record["sunrise_hour"] = int(record["sunrise_hour"])
    record["sunset_hour"] = int(record["sunset_hour"])
    record["date"] = pd.to_datetime(record["date"])
    return record


def parse_ev_label_by_style(label: str, style_id: int) -> Optional[Dict[str, object]]:
    match = PAT_EV_BY_STYLE[style_id].fullmatch(label.strip())
    if match is None:
        return None
    record = match.groupdict()
    record["date"] = pd.to_datetime(record["date"])
    record["week_type"] = record["week_type"].strip().lower()
    record["load_type"] = record["load_type"].strip().lower()
    record["allocation"] = int(record["allocation"])
    return record


def parse_commercial_label_by_style(label: str, style_id: int) -> Optional[Dict[str, object]]:
    match = PAT_COMMERCIAL_BY_STYLE[style_id].fullmatch(label.strip())
    if match is None:
        return None
    record = match.groupdict()
    record["user_id"] = int(record["user_id"])
    record["date"] = pd.to_datetime(record["date"])
    record["day_type"] = record["day_type"].strip().lower()
    return record


def parse_transient_label_by_style(label: str, style_id: int) -> Optional[Dict[str, object]]:
    match = PAT_TRANSIENT_BY_STYLE[style_id].fullmatch(label.strip())
    if match is None:
        return None
    record = match.groupdict()
    record["event_type"] = re.sub(r"\s+", " ", record["event_type"]).strip().lower()
    record["network"] = re.sub(r"\s+", " ", record["network"]).strip()
    record["bus"] = int(record["bus"])
    record["prompt_style"] = style_id
    network_num = re.search(r"IEEE\s*(\d+)", record["network"], flags=re.IGNORECASE)
    record["network_size"] = int(network_num.group(1)) if network_num else None
    return record


def build_metadata(labels_path: str, style_id: int, parser: Callable[[str, int], Optional[Dict[str, object]]], outdir: str) -> pd.DataFrame:
    """Parse labels for one domain/style and save metadata CSV."""
    rows = load_labels_from_jsonl(labels_path)
    records = []
    for idx, label_str in enumerate(rows):
        info = parser(label_str, style_id)
        if info is None:
            continue
        info["idx"] = idx
        records.append(info)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No labels matched style {style_id} in {labels_path}")

    meta_path = os.path.join(outdir, f"metadata_style{style_id}.csv")
    df.sort_values("idx").to_csv(meta_path, index=False)
    print(f"[style {style_id}] Saved metadata to {meta_path} with shape {df.shape}")
    return df.sort_values("idx").reset_index(drop=True)


def build_commercial_index_for_style(labels_path: str, style_id: int, outdir: str) -> Dict[str, object]:
    """
    Parse commercial labels and build subset indices for evaluation.
    """
    df = build_metadata(labels_path, style_id, parse_commercial_label_by_style, outdir=outdir)
    df["month"] = df["date"].dt.month
    df["is_summer"] = df["month"].between(4, 9)

    weekday_idx = df.loc[df["day_type"] == "weekday", "idx"].tolist()
    weekend_idx = df.loc[df["day_type"] == "weekend", "idx"].tolist()
    summer_idx = df.loc[df["is_summer"], "idx"].tolist()
    winter_idx = df.loc[~df["is_summer"], "idx"].tolist()
    idx_by_user = {uid: group["idx"].tolist() for uid, group in df.groupby("user_id")}

    return {
        "df": df,
        "weekday_idx": weekday_idx,
        "weekend_idx": weekend_idx,
        "summer_idx": summer_idx,
        "winter_idx": winter_idx,
        "idx_by_user": idx_by_user,
    }


def expand_metadata_for_sampling(df_meta: pd.DataFrame, n_samples_per_condition: int) -> pd.DataFrame:
    """Repeat metadata rows to match repeated sampling per condition."""
    expanded = df_meta.loc[df_meta.index.repeat(n_samples_per_condition)].copy().reset_index(drop=True)
    expanded["rep_id"] = np.tile(np.arange(n_samples_per_condition), len(df_meta))
    expanded["sample_idx"] = np.arange(len(expanded))
    return expanded


# =============================================================================
# Model loading and sampling
# =============================================================================


def load_models(
    device: torch.device,
    ae_ckpt_path: str = "cond_autoencoder.pt",
    diffusion_ckpt_path: str = "gridldm_diffusion.pt",
) -> Tuple[ConditionalTimeSeriesAutoencoder, GaussianDiffusion]:
    """Load the shared AE and diffusion checkpoints."""
    ae = ConditionalTimeSeriesAutoencoder(
        max_seq_len=OUTPUT_LENGTH,
        d_model=64,
        latent_dim=LATENT_DIM,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        text_token_dim=TEXT_DIM,
    ).to(device)

    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae.load_state_dict(ae_ckpt["model_state_dict"])
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False

    diff_ckpt = torch.load(diffusion_ckpt_path, map_location=device)
    diff_cfg_dict = diff_ckpt["config"]
    diff_cfg = DiffusionConfig(
        timesteps=diff_cfg_dict["timesteps"],
        beta_start=diff_cfg_dict["beta_start"],
        beta_end=diff_cfg_dict["beta_end"],
        objective=diff_cfg_dict.get("objective", "eps"),
    )

    unet_cfg = UNet1DConfig(
        in_channels=LATENT_CHANNELS,
        base_channels=128,
        channel_mults=(1, 2, 2),
        num_res_blocks=2,
        num_heads=4,
        t_emb_dim=256,
        dropout=0.0,
    )
    unet = UNet1D(
        unet_cfg,
        cond_dim=TEXT_DIM,
        use_self_attn=True,
        use_cross_attn=True,
    ).to(device)

    diffusion = GaussianDiffusion(unet, diff_cfg).to(device)
    diffusion.load_state_dict(diff_ckpt["model_state_dict"])
    diffusion.eval()

    return ae, diffusion


def sample_cfg_batched(
    name: str,
    cond_tokens: torch.Tensor,
    diffusion: GaussianDiffusion,
    decode_model: ConditionalTimeSeriesAutoencoder,
    mean: float,
    std: float,
    device: torch.device,
    guidance_scale: float,
    chunk_size: int,
    outdir: str,
    verbose: bool = True,
) -> None:
    """
    Generate synthetic time series with classifier-free guidance and save one CSV.
    """
    batch_total = cond_tokens.shape[0]
    if batch_total == 0:
        if verbose:
            print(f"[{name}] Nothing to do.")
        return

    Path(outdir).mkdir(parents=True, exist_ok=True)
    rows = []
    n_chunks = math.ceil(batch_total / chunk_size)

    if verbose:
        print(
            f"[{name}] Start sampling | B={batch_total}, chunk_size={chunk_size}, "
            f"n_chunks={n_chunks}, guidance_scale={guidance_scale}, device={device}"
        )

    overall_t0 = time.perf_counter()
    with torch.no_grad():
        for chunk_idx, start in enumerate(range(0, batch_total, chunk_size), 1):
            end = min(start + chunk_size, batch_total)
            current = cond_tokens[start:end].to(device, non_blocking=True)
            uncond = torch.zeros_like(current)

            t0 = time.perf_counter()
            latent = diffusion.sample_cfg(
                shape=(current.shape[0], LATENT_CHANNELS, LATENT_DIM),
                cond_tokens=current,
                uncond_tokens=uncond,
                guidance_scale=guidance_scale,
                device=device,
            )
            recon_norm = decode_model.decode(
                latent.squeeze(1),
                cond_seq=current,
                target_len=OUTPUT_LENGTH,
            )
            recon = recon_norm * std + mean
            rows.append(recon.detach().cpu())

            del current, uncond, latent, recon
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            if verbose:
                t1 = time.perf_counter()
                message = f"[{name}] Chunk {chunk_idx}/{n_chunks} | idx[{start}:{end}) | {t1 - t0:.2f}s"
                if torch.cuda.is_available():
                    message += (
                        f" | CUDA alloc={bytes_to_mb(torch.cuda.memory_allocated())}"
                        f", reserved={bytes_to_mb(torch.cuda.memory_reserved())}"
                    )
                print(message)

    arr = torch.cat(rows, dim=0).numpy()
    out_path = os.path.join(outdir, f"syn_{name}.csv")
    columns = [f"t{j}" for j in range(OUTPUT_LENGTH)]
    pd.DataFrame(arr, columns=columns).to_csv(out_path, index=False)

    if verbose:
        total_time = time.perf_counter() - overall_t0
        print(f"[{name}] Done. Wrote {arr.shape} to {out_path} in {total_time:.2f}s.")


# =============================================================================
# Domain runners
# =============================================================================


def run_standard_domain(
    domain: DomainConfig,
    parser: Callable[[str, int], Optional[Dict[str, object]]],
    diffusion: GaussianDiffusion,
    decode_model: ConditionalTimeSeriesAutoencoder,
    mean: float,
    std: float,
    device: torch.device,
    guidance_scale: float,
    chunk_size: int,
) -> None:
    """
    Run EV / wind / solar style generation.

    Generates one CSV per style with all conditions repeated n_samples_per_condition times.
    """
    for style_id, cfg in domain.style_config.items():
        labels_path = cfg["labels_path"]
        emb_path = cfg["emb_path"]

        if not os.path.exists(labels_path):
            print(f"[WARN][{domain.name}] Missing labels file: {labels_path}")
            continue
        if not os.path.exists(emb_path):
            print(f"[WARN][{domain.name}] Missing embedding file: {emb_path}")
            continue

        print(f"\n==== {domain.name.upper()} STYLE {style_id} ====")
        outdir = os.path.join(domain.output_root, f"style_{style_id}")
        Path(outdir).mkdir(parents=True, exist_ok=True)

        df_meta = build_metadata(labels_path, style_id, parser, outdir)
        cond_np = np.load(emb_path).astype(np.float32)

        if cond_np.shape[0] != len(df_meta):
            raise ValueError(
                f"[{domain.name} style {style_id}] embeddings count ({cond_np.shape[0]}) "
                f"does not match parsed label count ({len(df_meta)})."
            )

        cond_all = torch.from_numpy(cond_np).repeat_interleave(domain.n_samples_per_condition, dim=0)
        print(f"[style {style_id}] cond_all shape after repeat: {tuple(cond_all.shape)}")

        sample_cfg_batched(
            name=f"style{style_id}_all",
            cond_tokens=cond_all,
            diffusion=diffusion,
            decode_model=decode_model,
            mean=mean,
            std=std,
            device=device,
            guidance_scale=guidance_scale,
            chunk_size=chunk_size,
            outdir=outdir,
            verbose=True,
        )


def run_transient_domain(
    domain: DomainConfig,
    diffusion: GaussianDiffusion,
    decode_model: ConditionalTimeSeriesAutoencoder,
    mean: float,
    std: float,
    device: torch.device,
    guidance_scale: float,
    chunk_size: int,
) -> None:
    """
    Run transient style generation.

    Also writes expanded metadata to keep one metadata row per sampled series.
    """
    for style_id, cfg in domain.style_config.items():
        labels_path = cfg["labels_path"]
        emb_path = cfg["emb_path"]

        if not os.path.exists(labels_path):
            print(f"[WARN][{domain.name}] Missing labels file: {labels_path}")
            continue
        if not os.path.exists(emb_path):
            print(f"[WARN][{domain.name}] Missing embedding file: {emb_path}")
            continue

        print(f"\n==== TRANSIENT STYLE {style_id} ====")
        outdir = os.path.join(domain.output_root, f"style_{style_id}")
        Path(outdir).mkdir(parents=True, exist_ok=True)

        df_meta = build_metadata(labels_path, style_id, parse_transient_label_by_style, outdir)
        cond_np = np.load(emb_path).astype(np.float32)

        if cond_np.shape[0] != len(df_meta):
            raise ValueError(
                f"[transient style {style_id}] embeddings count ({cond_np.shape[0]}) "
                f"does not match parsed label count ({len(df_meta)})."
            )

        cond_all = torch.from_numpy(cond_np).repeat_interleave(domain.n_samples_per_condition, dim=0)
        df_expanded = expand_metadata_for_sampling(df_meta, domain.n_samples_per_condition)
        expanded_path = os.path.join(outdir, f"metadata_style{style_id}_expanded.csv")
        df_expanded.to_csv(expanded_path, index=False)
        print(f"[style {style_id}] Saved expanded metadata to {expanded_path} with shape {df_expanded.shape}")

        sample_cfg_batched(
            name=f"style{style_id}_all",
            cond_tokens=cond_all,
            diffusion=diffusion,
            decode_model=decode_model,
            mean=mean,
            std=std,
            device=device,
            guidance_scale=guidance_scale,
            chunk_size=chunk_size,
            outdir=outdir,
            verbose=True,
        )


def run_commercial_domain(
    domain: DomainConfig,
    diffusion: GaussianDiffusion,
    decode_model: ConditionalTimeSeriesAutoencoder,
    mean: float,
    std: float,
    device: torch.device,
    guidance_scale: float,
    chunk_size: int,
) -> None:
    """
    Run commercial-load generation.

    Preserves the original subset-based output behavior:
    weekday, weekend, summer, winter, and one subset per user_id.
    """
    for style_id, cfg in domain.style_config.items():
        labels_path = cfg["labels_path"]
        emb_path = cfg["emb_path"]

        if not os.path.exists(labels_path):
            print(f"[WARN][{domain.name}] Missing labels file: {labels_path}")
            continue
        if not os.path.exists(emb_path):
            print(f"[WARN][{domain.name}] Missing embedding file: {emb_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"COMMERCIAL STYLE {style_id}")
        print(f"{'=' * 60}")

        outdir = os.path.join(domain.output_root, f"style_{style_id}")
        Path(outdir).mkdir(parents=True, exist_ok=True)

        idx_info = build_commercial_index_for_style(labels_path, style_id, outdir)
        cond_all = torch.from_numpy(np.load(emb_path).astype(np.float32))

        def gather(idx_list: List[int]) -> Optional[torch.Tensor]:
            if len(idx_list) == 0:
                return None
            return cond_all[idx_list]

        subset_specs = {
            "weekday": idx_info["weekday_idx"],
            "weekend": idx_info["weekend_idx"],
            "summer": idx_info["summer_idx"],
            "winter": idx_info["winter_idx"],
        }
        for user_id in sorted(idx_info["idx_by_user"].keys()):
            subset_specs[f"user_{user_id}"] = idx_info["idx_by_user"][user_id]

        for subset_name, idx_list in subset_specs.items():
            cond_subset = gather(idx_list)
            if cond_subset is None:
                print(f"[style {style_id} / {subset_name}] No samples, skipping.")
                continue

            cond_subset = cond_subset.repeat_interleave(domain.n_samples_per_condition, dim=0)
            sample_cfg_batched(
                name=f"style{style_id}_{subset_name}",
                cond_tokens=cond_subset,
                diffusion=diffusion,
                decode_model=decode_model,
                mean=mean,
                std=std,
                device=device,
                guidance_scale=guidance_scale,
                chunk_size=chunk_size,
                outdir=outdir,
                verbose=True,
            )


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified multi-domain sampling script.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["all"],
        choices=["all", "wind", "solar", "ev", "commercial_load", "transient_voltage"],
        help="Domains to run. Default: all",
    )
    parser.add_argument("--ae-ckpt", default="cond_autoencoder.pt", help="Path to AE checkpoint.")
    parser.add_argument("--diff-ckpt", default="gridldm_diffusion.pt", help="Path to diffusion checkpoint.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument("--chunk-size", type=int, default=64, help="Sampling chunk size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"Using device: {device}")

    all_domains = default_domain_configs()
    selected_domains = list(all_domains.keys()) if "all" in args.domains else args.domains

    print("Loading models...")
    ae, diffusion = load_models(
        device=device,
        ae_ckpt_path=args.ae_ckpt,
        diffusion_ckpt_path=args.diff_ckpt,
    )
    print("✓ Models loaded")

    with torch.no_grad():
        for domain_name in selected_domains:
            domain = all_domains[domain_name]
            print(f"\n{'#' * 72}")
            print(f"Domain: {domain_name}")
            print(f"{'#' * 72}")

            mean, std = compute_domain_stats(domain)
            print(f"Normalization stats | mean={mean:.6f}, std={std:.6f}")

            if domain_name == "wind":
                run_standard_domain(
                    domain=domain,
                    parser=parse_wind_label_by_style,
                    diffusion=diffusion,
                    decode_model=ae,
                    mean=mean,
                    std=std,
                    device=device,
                    guidance_scale=args.guidance_scale,
                    chunk_size=args.chunk_size,
                )
            elif domain_name == "solar":
                run_standard_domain(
                    domain=domain,
                    parser=parse_solar_label_by_style,
                    diffusion=diffusion,
                    decode_model=ae,
                    mean=mean,
                    std=std,
                    device=device,
                    guidance_scale=args.guidance_scale,
                    chunk_size=args.chunk_size,
                )
            elif domain_name == "ev":
                run_standard_domain(
                    domain=domain,
                    parser=parse_ev_label_by_style,
                    diffusion=diffusion,
                    decode_model=ae,
                    mean=mean,
                    std=std,
                    device=device,
                    guidance_scale=args.guidance_scale,
                    chunk_size=args.chunk_size,
                )
            elif domain_name == "commercial_load":
                run_commercial_domain(
                    domain=domain,
                    diffusion=diffusion,
                    decode_model=ae,
                    mean=mean,
                    std=std,
                    device=device,
                    guidance_scale=args.guidance_scale,
                    chunk_size=args.chunk_size,
                )
            elif domain_name == "transient_voltage":
                run_transient_domain(
                    domain=domain,
                    diffusion=diffusion,
                    decode_model=ae,
                    mean=mean,
                    std=std,
                    device=device,
                    guidance_scale=args.guidance_scale,
                    chunk_size=args.chunk_size,
                )
            else:
                raise ValueError(f"Unsupported domain: {domain_name}")

    print("\nAll requested domains are done.")


if __name__ == "__main__":
    main()
