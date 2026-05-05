"""
HLE pass@1 sampling: randomly pick N questions, generate 1 solution each,
extract answer, compare to ground truth.

Usage:
  python hle/pass1_sample.py --n 50 --workers 20 --seed 42
"""

import argparse
import json
import random
import re
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from opendeepthink.llm_client import call_gemini
from hle.utils import extract_answer, normalize, is_correct
from tqdm import tqdm

DATA_FILE  = Path("data/hle/hle_gold.json")
OUT_DIR    = Path("data/hle/pass1_results")

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




def solve_one(item: dict) -> dict:
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
        "id":          item["id"],
        "category":    item["category"],
        "question":    item["question"][:200],
        "answer":      item["answer"],
        "predicted":   predicted,
        "correct":     correct,
        "raw_response": response,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=50)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    data = json.loads(DATA_FILE.read_text())
    rng  = random.Random(args.seed)
    sample = rng.sample(data, min(args.n, len(data)))
    print(f"Sampled {len(sample)} questions")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_one, item): item for item in sample}
        with tqdm(total=len(futures), desc="solving") as bar:
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                # save incrementally
                out_path = OUT_DIR / f"{res['id']}.json"
                out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
                bar.update(1)

    # Summary
    correct = sum(1 for r in results if r["correct"])
    print(f"\n=== Results ===")
    print(f"  Total:   {len(results)}")
    print(f"  Correct: {correct} ({100*correct/len(results):.1f}%)")
    print()

    # By category
    from collections import defaultdict
    cat_stats = defaultdict(lambda: [0, 0])
    for r in results:
        cat_stats[r["category"]][1] += 1
        if r["correct"]:
            cat_stats[r["category"]][0] += 1
    print("  By category:")
    for cat, (ac, tot) in sorted(cat_stats.items()):
        print(f"    {cat:<35} {ac}/{tot} ({100*ac/tot:.0f}%)")

    # Save summary
    summary = {
        "n": len(results),
        "correct": correct,
        "pass1": correct / len(results),
        "by_category": {k: {"ac": v[0], "total": v[1]} for k, v in cat_stats.items()},
        "results": results,
    }
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
