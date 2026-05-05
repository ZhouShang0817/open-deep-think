"""
HLE BT Verify: for each question, run pairwise LLM judge tournament,
fit Bradley-Terry model, select top-ranked solution, compare to GT.

Usage:
  python hle/bt_verify.py --n_rounds 4 --workers 20 --seed 42
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hle.judge import run_tournament
from hle.utils import extract_answer, normalize, is_correct
from opendeepthink.bt import compute_bt_scores, rank_solutions
from tqdm import tqdm

PASSK_DIR = Path("data/hle/passk_results")
OUT_DIR   = Path("data/hle/bt_results")




def process_question(qid: str, attempts: list[dict], n_rounds: int, workers: int, rng: random.Random) -> dict:
    """Run tournament for one question, return result dict."""
    # Sort by attempt index to ensure deterministic order
    attempts = sorted(attempts, key=lambda a: a["attempt"])
    question = attempts[0].get("question", "")
    gt = attempts[0]["answer"]
    solutions = [a["raw_response"] for a in attempts]
    n = len(solutions)

    # Run pairwise tournament
    out_file = OUT_DIR / f"{qid}_comparisons.jsonl"
    comparisons = run_tournament(
        question=question,
        solutions=solutions,
        n_rounds=n_rounds,
        workers=workers,
        rng=rng,
        out_file=out_file,
        label=qid[:8],
    )

    # BT scoring
    bt_scores = compute_bt_scores(n, comparisons)
    ranking = rank_solutions(bt_scores)
    best_idx = ranking[0]

    # Extract answer from best solution
    best_response = solutions[best_idx]
    predicted = extract_answer(best_response)
    correct = is_correct(predicted, gt)

    result = {
        "id":          qid,
        "answer":      gt,
        "bt_winner_idx":   best_idx,
        "bt_predicted":    predicted,
        "bt_correct":      correct,
        "bt_scores":       bt_scores,
        "ranking":         ranking,
        "n_comparisons":   len(comparisons),
        # also record pass@1 (attempt 0) for reference
        "pass1_predicted": extract_answer(solutions[0]),
        "pass1_correct":   is_correct(extract_answer(solutions[0]), gt),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rounds", type=int, default=4)
    ap.add_argument("--workers",  type=int, default=20)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load passk summary to get the list of 50 question IDs
    summary_path = PASSK_DIR / "summary.json"
    summary = json.loads(summary_path.read_text())
    qids = list(summary["per_question"].keys())
    print(f"Processing {len(qids)} questions, {args.n_rounds} rounds each")

    # Load per-question attempts — need full raw_response + question text
    # passk_results/{id}.json has attempts but question text may be missing
    # Fall back to hle_gold.json for question text
    hle_gold = {
        item["id"]: item
        for item in json.loads(Path("data/hle/hle_gold.json").read_text())
    }

    rng = random.Random(args.seed)
    results = []
    bt_correct = 0
    pass1_correct = 0

    # Check already-done questions (for resume)
    done = set()
    for f in OUT_DIR.glob("*.json"):
        if f.name != "summary.json":
            done.add(f.stem)

    outer = tqdm(total=len(qids), desc="questions", unit="q", position=0)

    for i, qid in enumerate(qids):
        if qid in done:
            existing = json.loads((OUT_DIR / f"{qid}.json").read_text())
            results.append(existing)
            bt_correct += int(existing["bt_correct"])
            pass1_correct += int(existing["pass1_correct"])
            outer.update(1)
            tqdm.write(f"[{i+1:2d}/{len(qids)}] {qid[:8]}... (cached) "
                       f"BT={'✓' if existing['bt_correct'] else '✗'}  "
                       f"pass@1={'✓' if existing['pass1_correct'] else '✗'}")
            continue

        # Load attempts
        attempts_path = PASSK_DIR / f"{qid}.json"
        attempts = json.loads(attempts_path.read_text())

        # Inject question text from hle_gold if not present in attempts
        if not attempts[0].get("question") and qid in hle_gold:
            q_text = hle_gold[qid]["question"]
            for a in attempts:
                a["question"] = q_text

        tqdm.write(f"[{i+1:2d}/{len(qids)}] {qid[:8]}... judging ({args.n_rounds} rounds × 6 pairs = {args.n_rounds*6} calls)")
        res = process_question(qid, attempts, args.n_rounds, args.workers, rng)
        results.append(res)

        bt_correct += int(res["bt_correct"])
        pass1_correct += int(res["pass1_correct"])

        # Save per-question result
        out_path = OUT_DIR / f"{qid}.json"
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))

        outer.update(1)
        n_done = len(results)
        tqdm.write(f"           └─ BT={'✓' if res['bt_correct'] else '✗'}  "
                   f"pass@1={'✓' if res['pass1_correct'] else '✗'}  │  "
                   f"累计: BT={bt_correct}/{n_done} ({100*bt_correct/n_done:.0f}%)  "
                   f"pass@1={pass1_correct}/{n_done} ({100*pass1_correct/n_done:.0f}%)")

    outer.close()

    # Final summary
    n = len(results)
    print(f"\n=== Final Results ({n} questions) ===")
    print(f"  pass@1      : {pass1_correct}/{n} ({100*pass1_correct/n:.1f}%)")
    print(f"  BT verify   : {bt_correct}/{n}  ({100*bt_correct/n:.1f}%)")
    print(f"  pass@12 UB  : 72.0%  (reference)")

    # Load majority@12 from earlier computation for comparison
    print(f"  majority@12 : 58.0%  (reference)")

    out_summary = {
        "n": n,
        "n_rounds": args.n_rounds,
        "pass1":    {"correct": pass1_correct, "acc": pass1_correct / n},
        "bt_verify": {"correct": bt_correct,   "acc": bt_correct / n},
        "results": results,
    }
    summary_out = OUT_DIR / "summary.json"
    summary_out.write_text(json.dumps(out_summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {summary_out}")


if __name__ == "__main__":
    main()
