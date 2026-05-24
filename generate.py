"""Generation / sampling helpers built on top of NexusGPT.generate().

The wire format is set by tokenizer.py:

    <user>{prompt}<nexus>{reply}<end>

chat() takes a raw user string, frames it correctly, and decodes only
the new tokens (skipping the prompt echo).
"""

from __future__ import annotations
from typing import Callable

import torch

from .config import NexusConfig
from .model import NexusGPT
from .tokenizer import CharTokenizer, USER_TOKEN, NEXUS_TOKEN, END_TOKEN
from .tokenizer_bpe import BPETokenizer


def chat(
    prompt: str,
    model: NexusGPT,
    tokenizer: CharTokenizer | BPETokenizer,
    cfg: NexusConfig,
    on_text: Callable[[str], None] | None = None,
) -> str:
    """Send one user message; return the model's reply as a string.

    If on_text is provided, it's called with each generated character as
    soon as the token is emitted — used by the CLI for streaming output.
    """
    # Input validation
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if len(prompt) > 10000:
        raise ValueError("prompt too long (max 10000 characters)")
    # Basic safety check - filter harmful content
    harmful_keywords = ['hack', 'exploit', 'malware', 'virus', 'attack', 'bypass', 'inject']
    if any(keyword in prompt.lower() for keyword in harmful_keywords):
        raise ValueError("prompt contains potentially harmful content")
    
    framed = f"{USER_TOKEN}{prompt}{NEXUS_TOKEN}"
    ids = tokenizer.encode(framed)
    device = next(model.parameters()).device
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    start_len = idx.size(1)

    # Streaming callback: decode each new token id and pipe it on.
    def _on_token(tok_id: int) -> None:
        if on_text is None:
            return
        text = tokenizer.decode([tok_id])
        # Suppress the role markers themselves from the visible stream
        if text in (USER_TOKEN, NEXUS_TOKEN, END_TOKEN):
            return
        on_text(text)

    out = model.generate(
        idx,
        max_new_tokens=cfg.sample_max_new_tokens,
        temperature=cfg.sample_temperature,
        top_k=cfg.sample_top_k,
        stop_token_id=tokenizer.end_id,
        on_token=_on_token,
    )

    # Extract only the newly generated tokens.
    new_ids = out[0, start_len:].tolist()
    reply = tokenizer.decode(new_ids)

    # Trim role markers if they snuck in.
    for marker in (END_TOKEN, USER_TOKEN, NEXUS_TOKEN):
        reply = reply.replace(marker, "")
    return reply.strip()
