"""Training loop for Nexus AI.

Run with:
    python -m nexus.train

Saves a checkpoint to checkpoints/nexus.pt containing model weights, the
tokenizer vocab, and the full config. cli.py reads it back to chat.
"""

from __future__ import annotations
import math
import time
import logging
from pathlib import Path
from datetime import datetime

import torch
from torch.cuda.amp import GradScaler, autocast

from .config import NexusConfig, CORPUS_PATH, CHECKPOINT_PATH, CHECKPOINT_DIR
from .tokenizer import CharTokenizer
from .tokenizer_bpe import BPETokenizer
from .dataset import load_corpus, CorpusDataset
from .model import NexusGPT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─── learning-rate schedule ─────────────────────────────────────────────
def get_lr(it: int, cfg: NexusConfig) -> float:
    """Linear warmup, then cosine decay to min_lr."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─── evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model, ds: CorpusDataset, cfg: NexusConfig, device: str) -> dict:
    """Estimate train & val loss over a few minibatches. Used for logging only."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = ds.get_batch(split, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ─── checkpoint I/O ─────────────────────────────────────────────────────
def save_checkpoint(model, cfg: NexusConfig, tokenizer: CharTokenizer | BPETokenizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save tokenizer state based on type
    if isinstance(tokenizer, BPETokenizer):
        cfg.tokenizer_vocab = []  # BPE doesn't use vocab list
        cfg.tokenizer_model_path = tokenizer.model_path
    else:
        cfg.tokenizer_vocab = tokenizer.vocab
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, device: str | None = None):
    """Load a trained checkpoint. Returns (model, cfg, tokenizer)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = NexusConfig.from_dict(ckpt["config"])
    if device is not None:
        cfg.device = device

    # Load appropriate tokenizer
    if cfg.tokenizer_type == "bpe":
        tokenizer = BPETokenizer(model_path=cfg.tokenizer_model_path)
    else:
        tokenizer = CharTokenizer(vocab=cfg.tokenizer_vocab)

    model = NexusGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.resolve_device())
    model.eval()
    return model, cfg, tokenizer


# ─── main training entry point ──────────────────────────────────────────
def train(cfg: NexusConfig | None = None) -> None:
    cfg = cfg or NexusConfig()
    torch.manual_seed(cfg.seed)
    
    logger.info(f"Starting training at {datetime.now().isoformat()}")
    logger.info(f"Configuration: {cfg.to_dict()}")

    device = cfg.resolve_device()
    logger.info(f"device = {device}")
    logger.info(f"loading corpus from {CORPUS_PATH}")
    text = load_corpus(CORPUS_PATH)
    logger.info(f"corpus: {len(text):,} characters")

    # Fit tokenizer on the full corpus.
    if cfg.tokenizer_type == "bpe":
        tokenizer = BPETokenizer(vocab_size=cfg.vocab_size if cfg.vocab_size > 0 else 1000)
        tokenizer.fit(text, model_prefix=str(CHECKPOINT_DIR / "nexus_sp"))
        cfg.vocab_size = len(tokenizer)
        cfg.tokenizer_model_path = tokenizer.model_path
        logger.info(f"tokenizer: BPE with {len(tokenizer)} tokens (including {tokenizer.user_id, tokenizer.nexus_id, tokenizer.end_id} as user/nexus/end)")
    else:
        tokenizer = CharTokenizer()
        tokenizer.fit(text)
        cfg.vocab_size = len(tokenizer)
        logger.info(f"tokenizer: char-level with {len(tokenizer)} tokens (including {tokenizer.user_id, tokenizer.nexus_id, tokenizer.end_id} as user/nexus/end)")

    ds = CorpusDataset(text, tokenizer, cfg.block_size)
    logger.info(f"dataset: {ds.n_train:,} train tokens, {ds.n_val:,} val tokens")

    model = NexusGPT(cfg).to(device)
    logger.info(cfg.summary())
    logger.info(f"trainable params: {model.num_params():,}")

    # Decoupled weight decay: only on 2D matrices (Linear, Embedding); not
    # on 1D params like LayerNorm gains and biases.
    decay_params = [p for p in model.parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
    )

    t0 = time.time()
    best_val = float("inf")
    logger.info(f"training for {cfg.max_iters} iterations...")
    if cfg.use_amp and device != "cpu":
        scaler = GradScaler()
        logger.info("using mixed precision training (AMP)")
    else:
        scaler = None
        logger.info("mixed precision training disabled")

    for it in range(cfg.max_iters):
        # set lr for this step
        lr = get_lr(it, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = ds.get_batch("train", cfg.batch_size, device)

        if scaler is not None:
            with autocast():
                _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if it % cfg.log_interval == 0:
            elapsed = time.time() - t0
            msg = f"iter {it:>5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:5.1f}s"
            print(f"  {msg}")
            logger.info(msg)

        if it > 0 and it % cfg.eval_interval == 0:
            losses = estimate_loss(model, ds, cfg, device)
            eval_msg = f"[eval] train {losses['train']:.4f}  val {losses['val']:.4f}"
            print(f"    {eval_msg}")
            logger.info(eval_msg)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(model, cfg, tokenizer, CHECKPOINT_PATH)
                save_msg = f"[save] new best val={best_val:.4f} → {CHECKPOINT_PATH}"
                print(f"    {save_msg}")
                logger.info(save_msg)

    # final save
    save_checkpoint(model, cfg, tokenizer, CHECKPOINT_PATH)
    total_time = time.time() - t0
    final_msg = f"done in {total_time:.1f}s. checkpoint at {CHECKPOINT_PATH}"
    print(f"\n[nexus] {final_msg}")
    logger.info(f"Training completed: {final_msg}")
    logger.info(f"Final best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    train()
