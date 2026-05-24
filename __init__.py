"""Nexus AI — a from-scratch transformer language model for electronics nerds.

Architecture: decoder-only transformer (GPT-style), trained from scratch on a
hand-written corpus of electronics Q&A. No external API, no pretrained weights.

Modules:
    config      — hyperparameters and paths
    tokenizer   — character-level tokenizer with special role tokens
    model       — NexusGPT: multi-head causal self-attention transformer
    dataset     — corpus loader + batcher
    train       — training loop (AdamW + cosine LR schedule)
    generate    — autoregressive sampling with top-k / temperature
    cli         — interactive chat REPL

Entry points:
    python -m nexus.train   # train a fresh model on data/corpus.txt
    python -m nexus.cli     # chat with the trained model
"""

__version__ = "0.1.0"
__all__ = ["config", "tokenizer", "model", "dataset", "train", "generate"]
