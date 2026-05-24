"""Character-level tokenizer for Nexus AI, with named role tokens.

Why char-level? It works on tiny corpora, has zero vocab-OOV failures, and
keeps the implementation honest — no external BPE library, no merge-rules
file you have to schlep around.

Why role tokens then? So the model can learn turn structure (user → nexus →
end) without having to memorise a particular phrasing. We register them as
single atomic tokens that get matched BEFORE we fall back to char-by-char,
so the string "<user>" tokenises to ONE id, not six.

Conversation wire format (also produced by dataset.py from the corpus):

    <user>How do I read a 4-band resistor?<nexus>Read it from the band ...<end>

At inference time we feed the prompt up through and including "<nexus>" and
sample until the model emits "<end>" (or we hit the token budget).
"""

from __future__ import annotations
from typing import Iterable, List
import re


# Role tokens. The angle-bracket form is just an ergonomic naming choice;
# what matters is that these strings are matched as ATOMIC tokens before
# the char fallback kicks in.
USER_TOKEN = "<user>"
NEXUS_TOKEN = "<nexus>"
END_TOKEN = "<end>"
SPECIAL_TOKENS: tuple[str, ...] = (USER_TOKEN, NEXUS_TOKEN, END_TOKEN)


class CharTokenizer:
    """Greedy tokenizer: tries each special token first, then single chars.

    The vocabulary is built from `fit()` over the training corpus. Special
    tokens are always assigned the lowest ids (0, 1, 2, ...), so they stay
    stable even if the corpus changes.
    """

    def __init__(self, vocab: List[str] | None = None) -> None:
        if vocab is None:
            vocab = list(SPECIAL_TOKENS)
        self._set_vocab(vocab)

    def _set_vocab(self, vocab: List[str]) -> None:
        # de-duplicate while preserving order
        seen, ordered = set(), []
        for t in vocab:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        # ensure all special tokens are present at the front
        for s in reversed(SPECIAL_TOKENS):
            if s not in ordered:
                ordered.insert(0, s)
            else:
                ordered.remove(s)
                ordered.insert(0, s)
        self.vocab: List[str] = ordered
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.itos: dict[int, str] = {i: t for i, t in enumerate(self.vocab)}
        # Precompile a regex that matches any special token, longest-first.
        specials_sorted = sorted(SPECIAL_TOKENS, key=len, reverse=True)
        self._special_re = re.compile("|".join(re.escape(s) for s in specials_sorted))

    # ─── lifecycle ───────────────────────────────────────────────────
    def fit(self, text: str) -> None:
        """Build the vocab from a training corpus.

        We strip out the role-token strings first so we don't double-count
        their letters as chars; then add every unique remaining character.
        """
        stripped = self._special_re.sub("", text)
        chars = sorted(set(stripped))
        self._set_vocab(list(SPECIAL_TOKENS) + chars)

    # ─── encode / decode ─────────────────────────────────────────────
    def encode(self, text: str) -> List[int]:
        """Greedy tokenization. Unknown chars fall back to a space-id so the
        model never sees an exception at inference time — but during fit()
        we cover the whole corpus, so this is a runtime safety net."""
        ids: List[int] = []
        i = 0
        while i < len(text):
            m = self._special_re.match(text, i)
            if m is not None:
                ids.append(self.stoi[m.group(0)])
                i = m.end()
                continue
            ch = text[i]
            ids.append(self.stoi.get(ch, self.stoi.get(" ", 0)))
            i += 1
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)

    # ─── ids for special tokens (convenience) ────────────────────────
    @property
    def user_id(self) -> int: return self.stoi[USER_TOKEN]
    @property
    def nexus_id(self) -> int: return self.stoi[NEXUS_TOKEN]
    @property
    def end_id(self) -> int: return self.stoi[END_TOKEN]

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={len(self)})"
