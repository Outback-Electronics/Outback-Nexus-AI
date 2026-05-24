"""Hyperparameters and filesystem paths for Nexus AI.

A single dataclass holds every knob so train.py, generate.py, and cli.py can
share one source of truth. Tweak NexusConfig() and re-run training.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


# ─── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.txt"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "nexus.pt"


@dataclass
class NexusConfig:
    """Hyperparameters for the model + trainer.

    The defaults below define a ~3.5M-parameter model that trains in 5-15
    minutes on a modern CPU and is small enough to sample interactively on
    any laptop. Bump n_layer / n_embd / block_size if you have a GPU.
    """

    # ─── model architecture ──────────────────────────────────────────
    vocab_size: int = 0          # filled in after tokenizer fit
    block_size: int = 512        # max context length, in tokens
    n_layer: int = 12            # transformer blocks (scaled for 1% production)
    n_head: int = 8              # attention heads per block (512/8=64)
    n_embd: int = 512            # residual stream dimension (scaled for 1% production)
    dropout: float = 0.2         # moderate dropout
    tie_weights: bool = True     # share embedding and output head weights

    # ─── optimization ────────────────────────────────────────────────
    batch_size: int = 16  # reduced batch size for larger model
    max_iters: int = 8000  # more iterations for larger model
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.2
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ─── eval / logging ──────────────────────────────────────────────
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 50

    # ─── sampling ────────────────────────────────────────────────────
    sample_temperature: float = 0.8
    sample_top_k: int = 40
    sample_max_new_tokens: int = 400

    # ─── runtime ─────────────────────────────────────────────────────
    seed: int = 1337
    device: str = "auto"         # "auto" | "cpu" | "cuda" | "mps"
    use_amp: bool = True         # use automatic mixed precision (FP16/BF16)

    # tokenizer state — populated at training time, persisted in the
    # checkpoint, never edited by hand.
    tokenizer_vocab: list = field(default_factory=list)
    tokenizer_type: str = "bpe"  # "char" or "bpe"
    tokenizer_model_path: str = ""  # path to sentencepiece model for BPE

    # ─── helpers ─────────────────────────────────────────────────────
    def resolve_device(self) -> str:
        import torch
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NexusConfig":
        # keep only keys the dataclass actually defines, so newer checkpoints
        # with extra keys still load.
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)

    def summary(self) -> str:
        approx_params = (
            self.vocab_size * self.n_embd                                # tok emb
            + self.block_size * self.n_embd                              # pos emb
            + self.n_layer * (
                4 * self.n_embd * self.n_embd                            # qkv + proj
                + 8 * self.n_embd * self.n_embd                          # mlp 4x
                + 4 * self.n_embd                                        # layernorms
            )
            + (0 if self.tie_weights else self.vocab_size * self.n_embd) # head
        )
        return (
            f"NexusGPT · {self.n_layer}L · {self.n_head}H · d={self.n_embd}\n"
            f"  vocab={self.vocab_size}  block={self.block_size}  "
            f"params≈{approx_params/1e6:.2f}M  device={self.resolve_device()}"
        )
