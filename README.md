# Nexus AI

A small transformer language model, built from scratch in PyTorch, trained
on a hand-written corpus of electronics Q&A. **No external API. No
pretrained weights.** Architecture, tokenizer, training loop, and sampler
are all in this repo.

It is, deliberately, tiny — a few million parameters, trains on a CPU in
roughly 5–15 minutes, runs interactively on any laptop. The whole point is
that the model is *yours* end-to-end: you can read every line of the
attention head, retrain on your own corpus, and watch it learn.

It will also be **wrong** sometimes. A 3.5M-parameter model trained on a
few hundred Q&A pairs is closer to a clever parrot than to an oracle. The
charm is the from-scratch-ness, not the accuracy.

## Quickstart

```bash
pip install -r requirements.txt   # just torch
python -m nexus.train             # train (CPU: ~5–15 min, GPU: way less)
python -m nexus.cli               # chat
```

## What's in the box

```
nexus/
  config.py      hyperparameters + paths (NexusConfig)
  tokenizer.py   char-level tokenizer with <user>/<nexus>/<end> role tokens
  model.py       NexusGPT: causal multi-head self-attention transformer
  dataset.py     corpus loader + (x, y) batcher
  train.py       AdamW + cosine LR schedule, eval, checkpointing
  generate.py    autoregressive sampling with top-k + temperature
  cli.py         interactive REPL
data/
  corpus.txt     hand-written electronics Q&A (extend it!)
```

## Architecture

A standard decoder-only transformer (GPT-style), written out by hand:

- learned token + positional embeddings
- N transformer blocks, each: pre-LayerNorm → causal multi-head
  self-attention → residual → pre-LayerNorm → 4× MLP with GELU → residual
- final LayerNorm + tied LM head
- GPT-2 style scaled init on the residual projections

Defaults (`NexusConfig`):

| knob          | value | what it does                          |
|---------------|-------|---------------------------------------|
| `n_layer`     | 6     | transformer blocks                    |
| `n_head`      | 8     | attention heads per block             |
| `n_embd`      | 256   | residual stream width                 |
| `block_size`  | 256   | max context length (chars)            |
| `batch_size`  | 32    | sequences per step                    |
| `max_iters`   | 4000  | training steps                        |
| `learning_rate` | 3e-4 | peak LR (linear warmup, cosine decay) |

Roughly **3.5M parameters**. Bump anything you like.

## Wire format

Conversations are framed with three atomic role tokens:

```
<user>How do I read a 4-band resistor?<nexus>Read it from the band ...<end>
```

`tokenizer.py` matches these strings as single ids before falling back to
char-level. `generate.chat()` frames a user prompt for you and stops at
`<end>`.

## Extending the corpus

Open `data/corpus.txt` and add more Q&A lines in the same format. The
tokenizer rebuilds its vocab on every training run, so adding new
characters is fine. Then re-train.

## Sampling knobs

From inside the CLI:

```
/temp 0.6      lower = sharper, higher = wilder  (default 0.8)
/topk 20       restrict sampling to the top-k logits (default 40)
/max 200       cap on new tokens per reply
```

## Caveats

- **Tiny model + tiny corpus = limited knowledge.** It memorizes more than
  it generalizes. Treat answers as suggestions, not facts.
- Char-level tokenization means generation is slower per "word" than BPE,
  but the implementation stays small.
- No safety filtering, no RLHF, no fine-tuning. It says what the corpus
  taught it to say.

## License

Public domain. Have fun.
