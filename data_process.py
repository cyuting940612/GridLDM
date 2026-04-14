#!/usr/bin/env python3
"""
Convert prompt JSONL files into BERT token embeddings.

This script scans prompt folders like:
    prompts/wind
    prompts/solar
    prompts/ev
    prompts/commercial_load
    prompts/transient_voltage

It recursively finds all `.jsonl` files, reads each line as a JSON string,
encodes the texts with BERT, and saves embeddings under a mirrored structure in:
    embedding/<domain>/...

Example:
    prompts/wind/train/wind_train_labels_1.jsonl
becomes:
    embedding/wind/train/wind_train_labels_1_emb.npy
    embedding/wind/train/wind_train_labels_1_mask.npy

The implementation follows the same pattern used in the user's previous scripts:
- read JSONL with `json.loads`
- `AutoTokenizer.from_pretrained`
- add `WS_0` ... `WS_200` special tokens
- `AutoModel.from_pretrained`
- save `last_hidden_state` as NumPy arrays
- optionally save attention masks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DOMAIN_DIRS: Dict[str, str] = {
    "wind": "wind",
    "solar": "solar",
    "ev": "ev",
    "commercial_load": "commercial_load",
    "transient_voltage": "transient_voltage",
}

SPECIAL_WS_TOKENS = [f"WS_{i}" for i in range(0, 201)]


def read_jsonl_texts(path: Path) -> List[str]:
    """Read texts from a JSONL file where each line is a JSON string."""
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc

            if not isinstance(item, str):
                raise ValueError(
                    f"Expected each JSONL line to decode to a string in {path}, line {line_no}; "
                    f"got {type(item).__name__}."
                )
            texts.append(item)
    return texts


def select_device(prefer_mps: bool = True) -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BertEmbedder:
    """Reusable BERT embedding wrapper."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 50,
        batch_size: int = 64,
        out_dtype: str = "float32",
        prefer_mps: bool = True,
    ) -> None:
        if out_dtype not in {"float32", "float16"}:
            raise ValueError("out_dtype must be 'float32' or 'float16'.")

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.out_dtype = out_dtype
        self.device = select_device(prefer_mps=prefer_mps)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(SPECIAL_WS_TOKENS)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device).eval()

    def encode(self, texts: List[str], save_mask: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """Encode texts into token-level embeddings and optional attention masks."""
        hidden_size = self.model.config.hidden_size
        n = len(texts)
        dtype = np.float16 if self.out_dtype == "float16" else np.float32

        embeddings = np.empty((n, self.max_length, hidden_size), dtype=dtype)
        masks = np.empty((n, self.max_length), dtype=np.uint8) if save_mask else None

        with torch.no_grad():
            idx = 0
            while idx < n:
                batch = texts[idx : idx + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                hs = out.last_hidden_state.detach().cpu().numpy()
                if self.out_dtype == "float16":
                    hs = hs.astype(np.float16)
                bsz = hs.shape[0]
                embeddings[idx : idx + bsz] = hs
                if masks is not None:
                    masks[idx : idx + bsz] = enc["attention_mask"].detach().cpu().numpy().astype(np.uint8)
                idx += bsz

        return embeddings, masks


def find_jsonl_files(domain_prompt_dir: Path) -> List[Path]:
    """Find all JSONL files recursively under a domain prompt directory."""
    return sorted(p for p in domain_prompt_dir.rglob("*.jsonl") if p.is_file())


def output_paths(prompt_file: Path, domain_prompt_dir: Path, domain_embed_dir: Path) -> tuple[Path, Path]:
    """Build mirrored embedding/mask paths for a prompt JSONL file."""
    relative = prompt_file.relative_to(domain_prompt_dir)
    target_dir = domain_embed_dir / relative.parent
    stem = relative.stem
    emb_path = target_dir / f"{stem}_emb.npy"
    mask_path = target_dir / f"{stem}_mask.npy"
    return emb_path, mask_path


def convert_domain(
    domain_name: str,
    prompt_root: Path,
    embedding_root: Path,
    embedder: BertEmbedder,
    overwrite: bool = False,
    save_mask: bool = True,
) -> int:
    """Convert all JSONL prompt files for one domain."""
    prompt_dir = prompt_root / domain_name
    if not prompt_dir.exists():
        print(f"[skip] {domain_name}: prompt directory not found: {prompt_dir}")
        return 0

    files = find_jsonl_files(prompt_dir)
    if not files:
        print(f"[skip] {domain_name}: no .jsonl files found under {prompt_dir}")
        return 0

    embed_dir = embedding_root / domain_name
    converted = 0

    for prompt_file in files:
        emb_path, mask_path = output_paths(prompt_file, prompt_dir, embed_dir)
        if emb_path.exists() and (not save_mask or mask_path.exists()) and not overwrite:
            print(f"[exists] {prompt_file} -> {emb_path}")
            continue

        texts = read_jsonl_texts(prompt_file)
        if not texts:
            print(f"[skip] empty file: {prompt_file}")
            continue

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings, masks = embedder.encode(texts, save_mask=save_mask)
        np.save(emb_path, embeddings)
        if save_mask and masks is not None:
            np.save(mask_path, masks)

        print(f"[done] {prompt_file} -> {emb_path} | shape={embeddings.shape}")
        converted += 1

    return converted


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert prompt JSONL files to BERT token embeddings."
    )
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=Path("prompts"),
        help="Root folder containing domain prompt directories.",
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path("embedding"),
        help="Root folder to save embeddings.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=list(DOMAIN_DIRS.keys()),
        choices=list(DOMAIN_DIRS.keys()),
        help="Domains to process.",
    )
    parser.add_argument(
        "--model-name",
        default="bert-base-uncased",
        help="Transformer model name.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum token length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Embedding output dtype.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Do not save attention masks.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU even if MPS is available.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    embedder = BertEmbedder(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        out_dtype=args.out_dtype,
        prefer_mps=not args.cpu_only,
    )

    print(f"Using device: {embedder.device}")
    total = 0
    for domain in args.domains:
        total += convert_domain(
            domain_name=DOMAIN_DIRS[domain],
            prompt_root=args.prompt_root,
            embedding_root=args.embedding_root,
            embedder=embedder,
            overwrite=args.overwrite,
            save_mask=not args.no_mask,
        )

    print(f"Finished. Converted {total} JSONL file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
