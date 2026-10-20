# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for the examples/dmoe_lm entry points on NPU.

Runs train.py and generate.py in subprocesses on a tiny random token stream
(no network access). The suite skips cleanly when no NPU is present.
"""

from __future__ import annotations

import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.npu

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "dmoe_lm"


def _load_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_token_stream(path: Path, num_tokens: int, vocab_size: int) -> None:
    rng = np.random.default_rng(0)
    rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16).tofile(path)


def _write_tiny_tokenizer(data_dir: Path, vocab_size: int) -> None:
    """Build a byte-level BPE tokenizer offline so the smoke test needs no network."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {
        char: index
        for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    assert len(vocab) == vocab_size
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(data_dir)


def _run(script: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLE_DIR / script), *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _tiny_train_args(
    data_dir: Path, token_path: Path, checkpoint_dir: Path, vocab_size: int,
) -> list[str]:
    return [
        "--data-path",
        str(token_path),
        "--data-dir",
        str(data_dir),
        "--vocab-size",
        str(vocab_size),
        "--hidden-size",
        "128",
        "--ffn-hidden-size",
        "256",
        "--num-attention-heads",
        "4",
        "--num-layers",
        "2",
        "--num-experts",
        "4",
        "--top-k",
        "2",
        "--seq-length",
        "64",
        "--micro-batch-size",
        "4",
        "--steps",
        "6",
        "--warmup-steps",
        "2",
        "--lr",
        "1e-3",
        "--log-interval",
        "1",
        "--eval-interval",
        "999999",
        "--save",
        str(checkpoint_dir),
        "--save-interval",
        "999999",
    ]


def _assert_finite_step_losses(output: str, steps: int) -> None:
    losses = [
        float(value) for value in re.findall(r"(?m)^step=\d+ lm_loss=([0-9.]+)", output)
    ]
    assert len(losses) == steps, output
    assert all(math.isfinite(loss) for loss in losses), losses


def _assert_generated_text(output: str) -> None:
    match = re.search(r"(?m)^generated:(.*)$", output)
    assert match is not None and match.group(1).strip(), output


def test_train_and_generate_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    pytest.importorskip("transformers")

    vocab_size = 256
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    token_path = tmp_path / "tokens.bin"
    _write_token_stream(token_path, 32768, vocab_size)
    _write_tiny_tokenizer(data_dir, vocab_size)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint = checkpoint_dir / "checkpoint.pt"

    train = _run(
        "train.py",
        _tiny_train_args(data_dir, token_path, checkpoint_dir, vocab_size),
        tmp_path,
    )
    train_out = _output(train)
    assert train.returncode == 0, train_out
    assert "device: npu" in train_out, train_out
    _assert_finite_step_losses(train_out, 6)
    assert "summary:" in train_out, train_out
    assert checkpoint.is_file()

    generate = _run(
        "generate.py",
        [
            "--checkpoint",
            str(checkpoint),
            "--tokenizer-path",
            str(data_dir),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "8",
            "--greedy",
        ],
        tmp_path,
    )
    generate_out = _output(generate)
    assert generate.returncode == 0, generate_out
    assert "device: npu" in generate_out, generate_out
    _assert_generated_text(generate_out)

    # CLI architecture flags disagree with the checkpoint on purpose.
    resume = _run(
        "train.py",
        [
            "--data-path",
            str(token_path),
            "--data-dir",
            str(data_dir),
            "--vocab-size",
            "32",
            "--hidden-size",
            "64",
            "--ffn-hidden-size",
            "128",
            "--num-attention-heads",
            "2",
            "--num-layers",
            "6",
            "--num-experts",
            "8",
            "--top-k",
            "1",
            "--seq-length",
            "32",
            "--steps",
            "6",
            "--load",
            str(checkpoint_dir),
            "--eval-interval",
            "999999",
        ],
        tmp_path,
    )
    resume_out = _output(resume)
    assert resume.returncode == 0, resume_out
    assert "model: 2 layers, hidden 128, 4 experts x top-2" in resume_out, resume_out
    assert "nothing to train: resumed at step 6 >= --steps 6" in resume_out, resume_out


def test_missing_explicit_data_path_does_not_prepare(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"
    result = _run("train.py", ["--data-path", str(missing), "--steps", "1"], tmp_path)
    output = _output(result)
    assert result.returncode != 0, output
    assert "token file not found" in output, output
    assert "preparing the" not in output, output


def test_token_stream_includes_final_token(tmp_path: Path) -> None:
    data = _load_module("dmoe_lm_data", "data.py")
    seq_length = 4
    path = tmp_path / "tokens.bin"
    np.arange(seq_length + 1, dtype=np.uint16).tofile(path)
    stream = data.TokenStream(path, batch_size=2, seq_length=seq_length, seed=0)
    inputs, targets = stream.next_batch()
    assert inputs.shape == (2, seq_length)
    assert targets.shape == (2, seq_length)
    assert int(targets[0, -1]) == seq_length


def test_causal_attention_keeps_batch_and_heads_apart() -> None:
    torch = pytest.importorskip("torch")
    model = _load_module("dmoe_lm_model", "model.py")
    num_heads, batch, seq_length, head_dim = 4, 3, 5, 8
    hidden = num_heads * head_dim
    attention = model.CausalSelfAttention(
        hidden, num_heads, torch.device("cpu"), torch.float32,
    )
    tokens = torch.randn(seq_length, batch, hidden)
    qkv = attention.qkv(tokens).view(seq_length, batch, 3, num_heads, head_dim)
    query, key, value = qkv.permute(2, 1, 3, 0, 4).unbind(0)  # each [bs, nh, sl, hd]
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
    causal = torch.triu(
        torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1,
    )
    probs = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
    mixed = (probs @ value).permute(2, 0, 1, 3).reshape(seq_length, batch, hidden)
    expected = attention.proj(mixed)
    torch.testing.assert_close(attention(tokens), expected)
