"""Interactive REPL for Nexus AI.

    python -m nexus.cli

Loads checkpoints/nexus.pt, then drops you into a chat loop. Streams the
reply character-by-character as the model samples.
"""

from __future__ import annotations
import sys
from pathlib import Path

from .config import CHECKPOINT_PATH
from .train import load_checkpoint
from .generate import chat
from .tokenizer_bpe import BPETokenizer


BANNER = r"""
   _  __                   ___    ____
  / |/ /__ _ ___ __ _____ / _ |  /  _/
 /    / -_) \ / // (_-< / __ | _/ /
/_/|_/\__/_\_\\_,_/___//_/ |_|/___/

  Nexus AI · a tiny from-scratch transformer for electronics nerds
  type a question and hit enter.  /help for commands.  /quit to exit.
"""


HELP = """
  /help                 show this help
  /temp <float>         set sampling temperature (default 0.8)
  /topk <int>           set top-k cutoff (default 40)
  /max <int>            set max new tokens per reply
  /quit                 exit
"""


def main() -> None:
    try:
        if not Path(CHECKPOINT_PATH).exists():
            print(f"Error: No checkpoint found at {CHECKPOINT_PATH}.")
            print("Train the model first:  python -m nexus.train")
            sys.exit(1)

        print(BANNER)
        print(f"[nexus] loading checkpoint from {CHECKPOINT_PATH} ...")
        model, cfg, tokenizer = load_checkpoint(CHECKPOINT_PATH)
        tokenizer_type = "BPE" if isinstance(tokenizer, BPETokenizer) else "char-level"
        print(f"[nexus] tokenizer type: {tokenizer_type}")
        print(cfg.summary())
        print()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    while True:
        try:
            user = input("you  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue

        # ─── slash commands ───────────────────────────────────────────
        if user.startswith("/"):
            parts = user.split()
            cmd = parts[0].lower()
            if cmd in ("/quit", "/exit"):
                break
            if cmd == "/help":
                print(HELP)
                continue
            if cmd == "/temp" and len(parts) > 1:
                try:
                    temp = float(parts[1])
                    if temp <= 0:
                        print("Error: temperature must be positive")
                        continue
                    cfg.sample_temperature = temp
                    print(f"  temperature = {cfg.sample_temperature}")
                except ValueError:
                    print("Error: invalid temperature value")
                continue
            if cmd == "/topk" and len(parts) > 1:
                try:
                    topk = int(parts[1])
                    if topk <= 0:
                        print("Error: top_k must be positive")
                        continue
                    cfg.sample_top_k = topk
                    print(f"  top_k = {cfg.sample_top_k}")
                except ValueError:
                    print("Error: invalid top_k value")
                continue
            if cmd == "/max" and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    if max_tokens <= 0:
                        print("Error: max_new_tokens must be positive")
                        continue
                    cfg.sample_max_new_tokens = max_tokens
                    print(f"  max_new_tokens = {cfg.sample_max_new_tokens}")
                except ValueError:
                    print("Error: invalid max_new_tokens value")
                continue
            print(f"unknown command: {cmd}  (try /help)")
            continue

        # ─── chat ─────────────────────────────────────────────────────
        print("nexus> ", end="", flush=True)
        def _stream(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()
        try:
            chat(user, model, tokenizer, cfg, on_text=_stream)
            print()  # newline after streaming reply
        except (ValueError, TypeError) as e:
            print(f"\nError: {e}")
        except Exception as e:
            print(f"\nUnexpected error: {e}")

    print("\n[nexus] goodbye.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[nexus] interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[nexus] unexpected error: {e}")
        sys.exit(1)
