"""BPE tokenizer for Nexus AI.

Uses sentencepiece for subword tokenization, which is more efficient than
character-level tokenization and allows for larger vocabularies.
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import sentencepiece as spm


# Role tokens - these will be added as special tokens to the BPE vocab
USER_TOKEN = "<user>"
NEXUS_TOKEN = "<nexus>"
END_TOKEN = "<end>"
SPECIAL_TOKENS: tuple[str, ...] = (USER_TOKEN, NEXUS_TOKEN, END_TOKEN)


class BPETokenizer:
    """BPE tokenizer using sentencepiece.

    The vocabulary is built from `fit()` over the training corpus using
    sentencepiece's BPE algorithm. Special tokens are added to the vocab
    and assigned the lowest ids (0, 1, 2, ...) for stability.
    """

    def __init__(self, model_path: str | None = None, vocab_size: int = 1000) -> None:
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.sp = None

        if model_path and Path(model_path).exists():
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        """Load a trained sentencepiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def fit(self, text: str, model_prefix: str = "nexus_sp") -> None:
        """Train a BPE tokenizer on the corpus.

        Args:
            text: Training corpus
            model_prefix: Prefix for saved model files
        """
        # Write corpus to temp file for sentencepiece
        temp_file = Path(model_prefix + ".txt")
        temp_file.write_text(text, encoding="utf-8")

        # Train sentencepiece model
        import argparse
        args = argparse.Namespace(
            input=str(temp_file),
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            max_sentence_length=4096,
            shuffle_input_sentence=True,
            user_defined_symbols=list(SPECIAL_TOKENS),
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )

        spm.SentencePieceTrainer.train(
            input=str(temp_file),
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            max_sentence_length=4096,
            shuffle_input_sentence=True,
            user_defined_symbols=list(SPECIAL_TOKENS),
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )

        # Load the trained model
        self.model_path = model_prefix + ".model"
        self._load(self.model_path)

        # Clean up temp file
        temp_file.unlink()

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call fit() first.")
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text."""
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call fit() first.")
        return self.sp.decode(ids)

    @property
    def user_id(self) -> int:
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call fit() first.")
        return self.sp.piece_to_id(USER_TOKEN)

    @property
    def nexus_id(self) -> int:
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call fit() first.")
        return self.sp.piece_to_id(NEXUS_TOKEN)

    @property
    def end_id(self) -> int:
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call fit() first.")
        return self.sp.piece_to_id(END_TOKEN)

    def __len__(self) -> int:
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call fit() first.")
        return self.sp.get_piece_size()

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={len(self) if self.sp else 0})"
