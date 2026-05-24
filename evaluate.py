"""Evaluation utilities for Nexus AI.

Provides metrics beyond training loss: accuracy on test questions,
sample quality assessment, and perplexity calculations.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple

import torch

from .config import NexusConfig, CHECKPOINT_PATH, CORPUS_PATH
from .train import load_checkpoint
from .generate import chat
from .dataset import load_corpus, CorpusDataset


def calculate_perplexity(model, dataset: CorpusDataset, cfg: NexusConfig, device: str) -> float:
    """Calculate perplexity on a dataset split."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(cfg.eval_iters):
            x, y = dataset.get_batch("val", cfg.batch_size, device)
            _, loss = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    model.train()
    return perplexity


def evaluate_test_questions(
    model,
    tokenizer,
    cfg: NexusConfig,
    test_questions: List[str],
    expected_keywords: List[List[str]],
) -> Tuple[float, List[str]]:
    """Evaluate model on test questions by checking for expected keywords.

    Returns:
        accuracy: fraction of answers containing expected keywords
        answers: list of generated answers
    """
    model.eval()
    correct = 0
    answers = []

    with torch.no_grad():
        for question, keywords in zip(test_questions, expected_keywords):
            answer = chat(question, model, tokenizer, cfg)
            answers.append(answer)

            # Check if any expected keyword appears in the answer
            if any(keyword.lower() in answer.lower() for keyword in keywords):
                correct += 1

    accuracy = correct / len(test_questions) if test_questions else 0.0
    model.train()
    return accuracy, answers


def generate_samples(
    model,
    tokenizer,
    cfg: NexusConfig,
    prompts: List[str],
    max_samples: int = 5,
) -> List[str]:
    """Generate sample outputs for quality assessment."""
    model.eval()
    samples = []

    with torch.no_grad():
        for i, prompt in enumerate(prompts[:max_samples]):
            try:
                sample = chat(prompt, model, tokenizer, cfg)
                samples.append(f"Q: {prompt}\nA: {sample}\n")
            except Exception as e:
                samples.append(f"Q: {prompt}\nA: ERROR: {e}\n")

    model.train()
    return samples


def run_evaluation(cfg: NexusConfig | None = None) -> None:
    """Run comprehensive evaluation on a trained checkpoint."""
    cfg = cfg or NexusConfig()

    if not Path(CHECKPOINT_PATH).exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH}")
        return

    print("[nexus] loading checkpoint for evaluation...")
    model, cfg, tokenizer = load_checkpoint(CHECKPOINT_PATH)
    device = cfg.resolve_device()

    # Load corpus for perplexity calculation
    text = load_corpus(CORPUS_PATH)
    dataset = CorpusDataset(text, tokenizer, cfg.block_size)

    print("\n=== Evaluation Results ===")
    print(f"Model: {cfg.summary()}")

    # Calculate perplexity
    ppl = calculate_perplexity(model, dataset, cfg, device)
    print(f"Validation Perplexity: {ppl:.2f}")

    # Test questions with expected keywords
    test_questions = [
        "What is Ohm's law?",
        "How do I read a resistor color code?",
        "What is a capacitor?",
        "What is the difference between AC and DC?",
        "How do I solder?",
    ]

    expected_keywords = [
        ["voltage", "current", "resistance", "v = i * r"],
        ["color", "band", "digit", "multiplier"],
        ["store", "energy", "electric", "field", "dielectric"],
        ["alternating", "direct", "direction", "frequency"],
        ["heat", "flux", "iron", "solder"],
    ]

    accuracy, answers = evaluate_test_questions(model, tokenizer, cfg, device, test_questions, expected_keywords)
    print(f"\nTest Question Accuracy: {accuracy:.1%} ({accuracy * len(test_questions):.0f}/{len(test_questions)} correct)")

    print("\n=== Sample Answers ===")
    for q, a in zip(test_questions, answers):
        print(f"\nQ: {q}")
        print(f"A: {a[:200]}..." if len(a) > 200 else f"A: {a}")

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    run_evaluation()
