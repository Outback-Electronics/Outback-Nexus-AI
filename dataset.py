"""Dataset + batcher.

Loads the corpus, hands it to the tokenizer, and serves random (x, y) pairs
where y is x shifted by one — the standard next-token-prediction setup.

We do NOT pad. A "sample" is a random `block_size`-long slice from the
flat token stream; loss is computed on every position. This is the cheapest
sensible setup for from-scratch training on a single corpus.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch

from .tokenizer import CharTokenizer


def load_corpus(path: Path | str) -> str:
    """Read a UTF-8 text file. The corpus already encodes turns with the
    role tokens (<user>, <nexus>, <end>) so we don't transform it here."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Corpus at {path} is empty")
    return text


class CorpusDataset:
    """Tokenizes the corpus once and serves random batches.

    Split is deterministic given the seed: the last 5% of tokens are held
    out as a tiny validation set. With a small hand-written corpus, val
    loss is more of a smoke signal than a real measure — useful for
    catching overfitting collapses, not for model selection.
    """

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        block_size: int,
        val_fraction: float = 0.05,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size

        ids = tokenizer.encode(text)
        if len(ids) < block_size + 2:
            raise ValueError(
                f"Corpus is too short: {len(ids)} tokens, need at least "
                f"{block_size + 2} for block_size={block_size}"
            )

        data = torch.tensor(ids, dtype=torch.long)
        split = int(len(data) * (1.0 - val_fraction))
        self.train_data = data[:split]
        self.val_data = data[split:]
        self.n_train = len(self.train_data)
        self.n_val = len(self.val_data)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample `batch_size` random starting offsets, slice out
        `block_size` tokens for x, and `block_size` tokens shifted by one
        for y. Both are (B, T) long tensors.
        """
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(f"split={split} has too few tokens ({len(data)})")

        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])

        # pin to device. non_blocking is a hint only — harmless on CPU.
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y
