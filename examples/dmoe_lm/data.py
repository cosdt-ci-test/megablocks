# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Prepare the wikitext-2 token stream used by the dMoE example.

The text is tokenized with the GPT-2 BPE vocabulary and stored as a flat
uint16 token stream (``tokens_train.bin`` / ``tokens_val.bin``), the same
one-dimensional layout Megatron's ``.bin`` files use. The tokenizer is saved
alongside so generation can decode with the exact same vocabulary::

    python data.py prepare --data-dir data --dataset-source hf

Set ``HF_ENDPOINT=https://hf-mirror.com`` first when huggingface.co is not
reachable. ``--dataset-source modelscope`` loads the text through the
ModelScope SDK instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

HF_DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1")
HF_TOKENIZER = "gpt2"
MODELSCOPE_DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


class DataMeta(TypedDict):
    """Metadata written to ``meta.json`` next to the token streams."""

    dataset: str
    source: str
    tokenizer: str
    vocab_size: int
    train_tokens: int
    val_tokens: int


def _load_tokenizer() -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HF_TOKENIZER)


def _download_hf() -> tuple[str, str]:
    from datasets import load_dataset  # pyrefly: ignore[missing-import]

    dataset = load_dataset(*HF_DATASET)
    train_text = "".join(row["text"] for row in dataset["train"])
    val_text = "".join(row["text"] for row in dataset["validation"])
    return train_text, val_text


def _parquet_text(paths: list[Path]) -> str:
    """Concatenate the ``text`` column of parquet shards into one string."""
    import pyarrow.parquet as pq

    table = pq.read_table(paths, columns=["text"])
    return "".join(str(value) for value in table.column("text").to_pylist())


def _download_modelscope() -> tuple[str, str]:
    """Download the wikitext parquet shards from the ModelScope hub mirror."""
    try:
        from modelscope.hub.snapshot_download import (
            snapshot_download,
        )  # pyrefly: ignore[missing-import]
    except ImportError as error:
        raise RuntimeError(
            "pip install modelscope pyarrow to use --dataset-source modelscope",
        ) from error
    name, subset = MODELSCOPE_DATASET
    try:
        repo_dir = Path(snapshot_download(name, repo_type="dataset"))
        train_text = _parquet_text(sorted((repo_dir / subset).glob("train-*.parquet")))
        val_text = _parquet_text(
            sorted((repo_dir / subset).glob("validation-*.parquet")),
        )
    except Exception as error:
        raise RuntimeError(
            f"failed to load {name}/{subset} from ModelScope; the dataset id may have changed. "
            "Use --dataset-source hf (with HF_ENDPOINT=https://hf-mirror.com if needed).",
        ) from error
    return train_text, val_text


def _write_tokens(path: Path, tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Encode ``text`` to uint16 token ids and write them flat to ``path``."""
    ids = np.asarray(tokenizer.encode(text), dtype=np.int64)
    if ids.size and (int(ids.max()) >= 2**16 or int(ids.min()) < 0):
        raise ValueError(
            "token id out of uint16 range; this example stores tokens as uint16",
        )
    tokens = ids.astype(np.uint16)
    tokens.tofile(path)
    return int(tokens.size)


def prepare(data_dir: Path, source: str) -> DataMeta:
    """Download, tokenize, and write the uint16 token streams. Returns the metadata."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer()
    if source == "hf":
        train_text, val_text = _download_hf()
        dataset = "/".join(HF_DATASET)
    elif source == "modelscope":
        train_text, val_text = _download_modelscope()
        dataset = "/".join(MODELSCOPE_DATASET)
    else:
        raise ValueError(
            f"unknown dataset source {source!r}; expected hf or modelscope",
        )

    train_tokens = _write_tokens(data_dir / "tokens_train.bin", tokenizer, train_text)
    val_tokens = _write_tokens(data_dir / "tokens_val.bin", tokenizer, val_text)
    tokenizer.save_pretrained(data_dir / "tokenizer")

    meta = DataMeta(
        dataset=dataset,
        source=source,
        tokenizer=HF_TOKENIZER,
        vocab_size=len(tokenizer),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
    )
    (data_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(
        f"wrote {train_tokens} train / {val_tokens} val tokens to {data_dir} (vocab {len(tokenizer)})",
    )
    return meta


class TokenStream:
    """Yields ``(inputs, targets)`` batches from a uint16 token file."""

    data: np.memmap
    batch_size: int
    seq_length: int
    rng: np.random.Generator

    def __init__(
        self, path: Path | str, batch_size: int, seq_length: int, seed: int = 0,
    ) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if self.data.size < seq_length + 1:
            raise ValueError(
                f"token file {path} holds too few tokens ({self.data.size}) for a single {seq_length}-token step",
            )
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.rng = np.random.default_rng(seed)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Exclusive high is size - seq_length so the last token can be a target.
        offsets = self.rng.integers(
            0, self.data.size - self.seq_length, size=self.batch_size,
        )
        idx = offsets[:, None] + np.arange(self.seq_length)[None, :]
        inputs = np.asarray(self.data[idx], dtype=np.int64)
        targets = np.asarray(self.data[idx + 1], dtype=np.int64)
        return torch.from_numpy(inputs), torch.from_numpy(targets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the token stream for the dMoE example.",
    )
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--dataset-source", choices=("hf", "modelscope"), default="hf")
    args = parser.parse_args()
    prepare(Path(args.data_dir), args.dataset_source)


if __name__ == "__main__":
    main()
