"""
HLE pipeline v2: per-question full pipeline (pass@12 → gen1 → gen2).
Questions are run sequentially; within each question workers=20 run in parallel.
Excludes the 50 questions already sampled in v1 (seed=42, passk_results/).

Usage:
  python hle/pipeline_v2.py --workers 20 --seed 123
  python hle/pipeline_v2.py --workers 20 --seed 123 --n 50   # default
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from opendeepthink.llm_client import call_gemini
from hle.judge import run_tournament
from hle.mutate import mutate_batch
from hle.utils import extract_answer, is_correct
from opendeepthink.bt import compute_bt_scores, rank_solutions
from tqdm import tqdm


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

HLE_GOLD     = Path("data/hle/hle_gold.json")
PASSK_V1_DIR = Path("data/hle/passk_results")
PASSK_V2_DIR = Path("data/hle/passk_results_v2")
EVOLVE_DIR   = Path("data/hle/evolve_v2")

N_SOLUTIONS  = 12
N_ELITE      = 3
N_ROUNDS_G1  = 4   # gen0 → gen1
N_ROUNDS_G2  = 6   # gen1 → gen2

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


# ── Question sampling ─────────────────────────────────────────────────────────

def sample_questions(n: int, seed: int) -> list[dict]:
    """Sample n questions, excluding those already sampled in v1."""
    gold = json.loads(HLE_GOLD.read_text())

    v1_ids: set[str] = set()
    v1_summary = PASSK_V1_DIR / "summary.json"
    if v1_summary.exists():
        v1_ids = set(json.loads(v1_summary.read_text()).get("per_question", {}).keys())

    available = [q for q in gold if q["id"] not in v1_ids]
    rng = random.Random(seed)
    sample = rng.sample(available, min(n, len(available)))
    print(f"  V1 exclusions: {len(v1_ids)}  Available: {len(available)}  Sampled: {len(sample)}", flush=True)
    return sample


# ── Step A: generate solutions ────────────────────────────────────────────────

def generate_solutions(item: dict, workers: int) -> list[str]:
    """Generate N_SOLUTIONS solutions in parallel. Returns list of raw responses (ordered)."""
    question = item["question"]
    prompt   = PROMPT_TEMPLATE.format(question=question)
    results  = [None] * N_SOLUTIONS

    def _call(idx: int) -> tuple[int, str]:
        return idx, call_gemini(prompt, temperature=1.0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_call, i): i for i in range(N_SOLUTIONS)}
        for fut in as_completed(futures):
            try:
                idx, resp = fut.result()
            except Exception as e:
                idx  = futures[fut]
                resp = f"ERROR: {e}"
            results[idx] = resp

    return results


# ── BT step (with resume) ─────────────────────────────────────────────────────

def _save_gen_json(out_dir: Path, gen: int, n_rounds: int,
                   solutions: list[str], ranking: list[int],
                   scores: list[float], qid: str) -> None:
    data = {
        "gen":       gen,
        "qid":       qid,
        "n_rounds":  n_rounds,
        "ranking":   ranking,
        "bt_scores": scores,
        "solutions": [
            {
                "idx":       i,
                "rank":      ranking.index(i),
                "predicted": extract_answer(solutions[i]),
                "response":  solutions[i],
            }
            for i in range(len(solutions))
        ],
    }
    (out_dir / f"gen{gen}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


def run_bt_step(
    qid: str,
    question: str,
    solutions: list[str],
    gen: int,
    n_rounds: int,
    out_dir: Path,
    workers: int,
    rng: random.Random,
) -> tuple[list[dict], list[int]]:
    """
    Run BT tournament for `gen`. Saves gen{gen}.json and gen{gen}_comparisons.jsonl.
    Resume-safe: if comparisons file has enough records, reuse without re-calling LLM.
    Returns (comparisons, ranking).
    """
    n          = len(solutions)
    expected   = n_rounds * (n // 2)
    comp_file  = out_dir / f"gen{gen}_comparisons.jsonl"
    gen_json   = out_dir / f"gen{gen}.json"

    # Resume: enough comparisons already on disk
    if comp_file.exists():
        lines = [l for l in comp_file.read_text().splitlines() if l.strip()]
        if len(lines) >= expected:
            comparisons = [json.loads(l) for l in lines]
            scores      = compute_bt_scores(n, comparisons)
            ranking     = rank_solutions(scores)
            if not gen_json.exists():
                _save_gen_json(out_dir, gen, n_rounds, solutions, ranking, scores, qid)
            return comparisons, ranking
        # Partial file — delete and restart cleanly
        comp_file.unlink()

    # Run fresh tournament (appends to comp_file as results come in)
    comparisons = run_tournament(
        question=question,
        solutions=solutions,
        n_rounds=n_rounds,
        workers=workers,
        rng=rng,
        out_file=comp_file,
        label=f"{qid[:8]} g{gen}",
    )
    scores  = compute_bt_scores(n, comparisons)
    ranking = rank_solutions(scores)
    _save_gen_json(out_dir, gen, n_rounds, solutions, ranking, scores, qid)
    return comparisons, ranking


# ── Per-question pipeline ─────────────────────────────────────────────────────

def run_question(item: dict, workers: int, rng: random.Random) -> dict:
    qid      = item["id"]
    question = item["question"]
    gt       = item["answer"]

    out_dir  = EVOLVE_DIR / qid
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"qid": qid, "gt": gt}

    print(f"\n{'─'*60}", flush=True)
    _log(f"[{qid[:8]}] GT={str(gt)[:45]}")

    # ── Step A: pass@12 ──────────────────────────────────────────────────────
    passk_file = PASSK_V2_DIR / f"{qid}.json"
    if passk_file.exists():
        passk     = sorted(json.loads(passk_file.read_text()), key=lambda a: a["attempt"])
        gen0_sols = [a["raw_response"] for a in passk]
        _log(f"  pass@12 [cached]: {sum(a['correct'] for a in passk)}/{N_SOLUTIONS} correct")
    else:
        _log(f"  pass@12: generating {N_SOLUTIONS} solutions...")
        t0        = time.monotonic()
        gen0_sols = generate_solutions(item, workers)
        passk = [
            {
                "id":           qid,
                "attempt":      i,
                "answer":       gt,
                "predicted":    extract_answer(gen0_sols[i]),
                "correct":      is_correct(extract_answer(gen0_sols[i]), gt),
                "raw_response": gen0_sols[i],
            }
            for i in range(N_SOLUTIONS)
        ]
        passk_file.write_text(json.dumps(passk, indent=2, ensure_ascii=False))
        _log(f"  pass@12 done ({time.monotonic()-t0:.0f}s): {sum(a['correct'] for a in passk)}/{N_SOLUTIONS} correct")

    # ── Step B: gen0 → gen1 ──────────────────────────────────────────────────
    if "gen1_bt_correct" in result:
        _log(f"  gen0→gen1 [cached]: gen1 BT#1={'✓' if result['gen1_bt_correct'] else '✗'}")
    else:
        # BT on gen0
        _log(f"  gen0 BT ({N_ROUNDS_G1} rounds, {N_ROUNDS_G1 * N_SOLUTIONS//2} pairs)...")
        t0 = time.monotonic()
        comps0, ranking0 = run_bt_step(qid, question, gen0_sols, gen=0,
                                        n_rounds=N_ROUNDS_G1, out_dir=out_dir,
                                        workers=workers, rng=rng)
        pred0    = extract_answer(gen0_sols[ranking0[0]])
        correct0 = is_correct(pred0, gt)
        result["gen0_bt_predicted"] = pred0
        result["gen0_bt_correct"]   = correct0
        _log(f"  gen0 BT done ({time.monotonic()-t0:.0f}s): rank#1={pred0!r}  {'✓' if correct0 else '✗'}")

        # Mutate
        _log(f"  mutate gen0: {N_SOLUTIONS - N_ELITE} solutions...")
        t0        = time.monotonic()
        elite     = set(ranking0[:N_ELITE])
        to_mutate = list(ranking0[N_ELITE:])
        mutated   = mutate_batch(question, gen0_sols, comps0, to_mutate, workers)
        gen1_sols = [gen0_sols[i] if i in elite else mutated.get(i, gen0_sols[i])
                     for i in range(N_SOLUTIONS)]
        (out_dir / "gen1_raw.json").write_text(
            json.dumps({"gen": 1, "qid": qid, "solutions": gen1_sols}, indent=2, ensure_ascii=False)
        )
        _log(f"  mutate gen0 done ({time.monotonic()-t0:.0f}s)")

        # BT on gen1
        _log(f"  gen1 BT ({N_ROUNDS_G1} rounds, {N_ROUNDS_G1 * N_SOLUTIONS//2} pairs)...")
        t0 = time.monotonic()
        comps1, ranking1 = run_bt_step(qid, question, gen1_sols, gen=1,
                                        n_rounds=N_ROUNDS_G1, out_dir=out_dir,
                                        workers=workers, rng=rng)
        pred1    = extract_answer(gen1_sols[ranking1[0]])
        correct1 = is_correct(pred1, gt)
        result["gen1_bt_predicted"] = pred1
        result["gen1_bt_correct"]   = correct1
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        _log(f"  gen1 BT done ({time.monotonic()-t0:.0f}s): rank#1={pred1!r}  {'✓' if correct1 else '✗'}")

    # ── Step C: gen1 → gen2 ──────────────────────────────────────────────────
    if "gen2_bt_correct" in result:
        _log(f"  gen1→gen2 [cached]: gen2 BT#1={'✓' if result['gen2_bt_correct'] else '✗'}")
    else:
        gen1_raw  = json.loads((out_dir / "gen1_raw.json").read_text())
        gen1_sols = gen1_raw["solutions"]

        comp1_file = out_dir / "gen1_comparisons.jsonl"
        comps1     = [json.loads(l) for l in comp1_file.read_text().splitlines() if l.strip()] \
                     if comp1_file.exists() else []
        ranking1   = rank_solutions(compute_bt_scores(N_SOLUTIONS, comps1)) if comps1 \
                     else list(range(N_SOLUTIONS))

        # Mutate gen1
        _log(f"  mutate gen1: {N_SOLUTIONS - N_ELITE} solutions...")
        t0        = time.monotonic()
        elite     = set(ranking1[:N_ELITE])
        to_mutate = list(ranking1[N_ELITE:])
        mutated   = mutate_batch(question, gen1_sols, comps1, to_mutate, workers)
        gen2_sols = [gen1_sols[i] if i in elite else mutated.get(i, gen1_sols[i])
                     for i in range(N_SOLUTIONS)]
        (out_dir / "gen2_raw.json").write_text(
            json.dumps({"gen": 2, "qid": qid, "solutions": gen2_sols}, indent=2, ensure_ascii=False)
        )
        _log(f"  mutate gen1 done ({time.monotonic()-t0:.0f}s)")

        # BT on gen2
        _log(f"  gen2 BT ({N_ROUNDS_G2} rounds, {N_ROUNDS_G2 * N_SOLUTIONS//2} pairs)...")
        t0 = time.monotonic()
        _, ranking2 = run_bt_step(qid, question, gen2_sols, gen=2,
                                   n_rounds=N_ROUNDS_G2, out_dir=out_dir,
                                   workers=workers, rng=rng)
        pred2    = extract_answer(gen2_sols[ranking2[0]])
        correct2 = is_correct(pred2, gt)
        result["gen2_bt_predicted"] = pred2
        result["gen2_bt_correct"]   = correct2
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        _log(f"  gen2 BT done ({time.monotonic()-t0:.0f}s): rank#1={pred2!r}  {'✓' if correct2 else '✗'}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=50)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed",    type=int, default=123)
    args = ap.parse_args()

    PASSK_V2_DIR.mkdir(parents=True, exist_ok=True)
    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)

    questions = sample_questions(args.n, args.seed)
    cats = {}
    for q in questions:
        cats[q.get("category", "?")] = cats.get(q.get("category", "?"), 0) + 1
    print(f"Category breakdown: {dict(sorted(cats.items(), key=lambda x: -x[1]))}")

    rng     = random.Random(args.seed)
    results = []

    for i, item in enumerate(questions):
        qid = item["id"]
        print(f"\n[{i+1}/{len(questions)}] {qid[:8]}  {item.get('category','?')}")

        # Full resume: skip if gen2 already done
        out_dir     = EVOLVE_DIR / qid
        result_path = out_dir / "result.json"
        if result_path.exists():
            r = json.loads(result_path.read_text())
            if "gen2_bt_correct" in r:
                g0 = "✓" if r.get("gen0_bt_correct") else "✗"
                g1 = "✓" if r.get("gen1_bt_correct") else "✗"
                g2 = "✓" if r.get("gen2_bt_correct") else "✗"
                print(f"  [cached] gen0={g0} gen1={g1} gen2={g2}")
                results.append(r)
                continue

        try:
            r = run_question(item, args.workers, rng)
            results.append(r)
        except Exception as e:
            import traceback
            tqdm.write(f"  ERROR on {qid[:8]}: {e}")
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    n   = len(results)
    g0c = sum(1 for r in results if r.get("gen0_bt_correct"))
    g1c = sum(1 for r in results if r.get("gen1_bt_correct"))
    g2c = sum(1 for r in results if r.get("gen2_bt_correct"))
    print(f"\n{'='*50}")
    print(f"  Questions   : {n}")
    if n:
        print(f"  gen0 BT#1   : {g0c}/{n} ({100*g0c/n:.1f}%)")
        print(f"  gen1 BT#1   : {g1c}/{n} ({100*g1c/n:.1f}%)")
        print(f"  gen2 BT#1   : {g2c}/{n} ({100*g2c/n:.1f}%)")
    print(f"{'='*50}")

    (EVOLVE_DIR / "summary.json").write_text(
        json.dumps({"seed": args.seed, "n": n, "results": results}, indent=2, ensure_ascii=False)
    )
    print(f"Saved to {EVOLVE_DIR}/summary.json")


if __name__ == "__main__":
    main()
