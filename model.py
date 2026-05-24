"""NexusGPT — a decoder-only transformer language model.

This is a faithful, minimal implementation of the GPT architecture: token +
learned positional embeddings, N blocks of (causal multi-head self-attention
+ MLP) with pre-LayerNorm and residual streams, a final LayerNorm, and a
linear language-model head whose weights are tied to the token embedding.

Nothing here is imported from `transformers` or any other library — every
matmul is spelled out in `torch.nn`. This is the *actual model*.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import NexusConfig


# ─── attention ──────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Q/K/V are produced by a single fused linear projection, then reshaped
    into (B, n_head, T, head_dim). We compute scaled dot-product attention
    with a lower-triangular mask so position t can only attend to positions
    ≤ t. The output is concatenated across heads and projected back.
    """

    def __init__(self, cfg: NexusConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # Fused Q, K, V projection — one matmul instead of three.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Causal mask. Registered as a buffer so it moves to the right
        # device with .to(), but isn't a learned parameter.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project, split into Q/K/V, reshape into per-head views.
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Mix values, fold heads back together.
        y = att @ v                                       # (B, h, T, d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.resid_dropout(self.c_proj(y))


# ─── MLP ────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Position-wise feed-forward network. Standard 4× expansion + GELU."""

    def __init__(self, cfg: NexusConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


# ─── transformer block ─────────────────────────────────────────────────
class Block(nn.Module):
    """One transformer block: pre-LN, attention, residual; pre-LN, MLP, residual."""

    def __init__(self, cfg: NexusConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            def attn_forward(x):
                return self.attn(self.ln1(x))
            def mlp_forward(x):
                return self.mlp(self.ln2(x))
            x = x + checkpoint(attn_forward, x)
            x = x + checkpoint(mlp_forward, x)
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


# ─── full model ─────────────────────────────────────────────────────────
class NexusGPT(nn.Module):
    """Decoder-only transformer language model.

    Forward returns (logits, loss). If `targets` is None, loss is None and
    the caller is presumably sampling — see `generate()`.
    """

    def __init__(self, cfg: NexusConfig) -> None:
        super().__init__()
        assert cfg.vocab_size > 0, "vocab_size must be set before constructing model"
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        self.gradient_checkpointing = False

        if cfg.tie_weights:
            # Weight-tying: output projection shares the input embedding
            # matrix. Saves params and tends to improve sample quality.
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        # GPT-2 style scaled init for residual projections — keeps the
        # variance of the residual stream stable as depth grows.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ─── forward / loss ──────────────────────────────────────────────
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block size {self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)        # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)                    # (B, T, C)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                                        # (B, T, V)

        loss = None
        if targets is not None:
            # Standard next-token cross-entropy. Targets with id == -100
            # (the PyTorch default ignore_index) are masked out.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ─── autoregressive sampling ─────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token_id: int | None = None,
        on_token=None,
    ) -> torch.Tensor:
        """Sample tokens one at a time.

        Args:
            idx:            (B, T) prompt token ids.
            max_new_tokens: hard cap on tokens to generate.
            temperature:    >0. Lower = sharper / more deterministic.
            top_k:          if set, restrict sampling to the top-k logits.
            stop_token_id:  if emitted, generation halts early.
            on_token:       optional callback fn(token_id) called per token,
                            used by the CLI to stream output to the terminal.
        """
        self.eval()
        block_size = self.cfg.block_size

        for _ in range(max_new_tokens):
            # crop the context window so we never exceed block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)        # (B, 1)
            idx = torch.cat([idx, next_tok], dim=1)

            if on_token is not None:
                on_token(int(next_tok[0, 0].item()))
            if stop_token_id is not None and int(next_tok[0, 0].item()) == stop_token_id:
                break

        return idx

    # ─── housekeeping ────────────────────────────────────────────────
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
            if not self.cfg.tie_weights:
                n -= self.tok_emb.weight.numel()
        return n
