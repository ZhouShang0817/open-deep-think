"""
HLE pass@k sampling: for each of N questions, generate k solutions,
compute pass@k for all k in 1..K.

Usage:
  python hle/passk_sample.py --n 50 --k 12 --workers 20 --seed 42
"""

import argparse
import json
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from opendeepthink.llm_client import call_gemini
from hle.utils import extract_answer, normalize, is_correct
from tqdm import tqdm

DATA_FILE = Path("data/hle/hle_gold.json")
OUT_DIR   = Path("data/hle/passk_results")

PROMPT_TEMPLATE = """\
You are an expert at solving extremely difficult exam questions.

## Question
{question}

Carefully reason through this step by step, then provide your final answer.

At the very end of your response, write your final answer on its own line in this exact format:
ANSWER: <your answer here>

For multiple choice questions, just write the letter (e.g. ANSWER: A).
For numerical answers, write the number (e.g. ANSWER: 42 or ANSWER: 3.14).
For other answers, write the answer as concisely as possible.
"""




def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def solve_one_attempt(item: dict, attempt_idx: int) -> dict:
    prompt = PROMPT_TEMPLATE.format(question=item['question'])
    try:
        response = call_gemini(prompt, temperature=1.0)
        predicted = extract_answer(response)
        correct = is_correct(predicted, item['answer'])
    except Exception as e:
        response = f"ERROR: {e}"
        predicted = None
        correct = False

    return {
        "id":           item["id"],
        "attempt":      attempt_idx,
        "answer":       item["answer"],
        "predicted":    predicted,
        "correct":      correct,
        "raw_response": response,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=50)
    ap.add_argument("--k",       type=int, default=12)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    data = json.loads(DATA_FILE.read_text())
    rng  = random.Random(args.seed)
    sample = rng.sample(data, min(args.n, len(data)))
    print(f"Sampled {len(sample)} questions, {args.k} attempts each = {len(sample)*args.k} total calls")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build all (item, attempt_idx) tasks
    tasks = [(item, i) for item in sample for i in range(args.k)]

    # per-question results: id -> list of attempt dicts
    from collections import defaultdict
    per_question: dict[str, list] = defaultdict(list)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_one_attempt, item, idx): (item, idx)
                   for item, idx in tasks}
        with tqdm(total=len(futures), desc="sampling") as bar:
            for fut in as_completed(futures):
                res = fut.result()
                per_question[res["id"]].append(res)
                bar.update(1)

    # Save per-question results
    for qid, attempts in per_question.items():
        out = OUT_DIR / f"{qid}.json"
        out.write_text(json.dumps(attempts, indent=2, ensure_ascii=False))

    # Compute pass@k for k=1..K
    n_total = len(sample)
    K = args.k

    # per-question: how many correct out of K attempts
    q_correct = {qid: sum(1 for a in attempts if a["correct"])
                 for qid, attempts in per_question.items()}

    print(f"\n=== pass@k results (n={n_total} questions, {K} samples each) ===")
    print(f"{'k':>4}  {'pass@k':>8}")
    print("-" * 16)
    for k in range(1, K + 1):
        avg = sum(pass_at_k(K, c, k) for c in q_correct.values()) / n_total
        print(f"{k:>4}  {avg:>7.1%}")

    # Save summary
    summary = {
        "n": n_total,
        "k": K,
        "pass_at_k": {
            str(k): sum(pass_at_k(K, c, k) for c in q_correct.values()) / n_total
            for k in range(1, K + 1)
        },
        "per_question": {
            qid: {"correct": c, "total": K, "pass1_empirical": c / K}
            for qid, c in q_correct.items()
        },
    }
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
