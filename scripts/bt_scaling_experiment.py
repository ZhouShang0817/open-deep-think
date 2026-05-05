"""
BT Scaling experiment: study (n solutions, m rounds) tradeoff.

For each problem:
  1. Generate 40 solutions (reuse 20 existing gen0, generate 20 more)
  2. Submit all 40 to judge for ground truth
  3. Run full round-robin (39 rounds × 20 pairs = 780 comparisons)
  4. Save everything for offline analysis

Usage:
  python scripts/bt_scaling_experiment.py              # pilot: 1 problem
  python scripts/bt_scaling_experiment.py --all         # all 10 problems
  python scripts/bt_scaling_experiment.py --workers 20
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

from opendeepthink.generate import generate
from opendeepthink.judge    import _extract_code, _judge_pair
from opendeepthink.submit   import submit_all

# ── Config ───────────────────────────────────────────────────────────────────
POP_SIZE  = 40
N_ROUNDS  = 39   # full round-robin for 40 players

OUT_BASE   = Path("data/bt_scaling")
EVOLVE_DIR = Path("data/evolve")

# 10 problems with gen0 AC in [1,4] out of 20, sampled with seed=42
ALL_PIDS = [
    "10542", "11630", "1813", "1824", "1844",
    "4881", "4887", "6540", "8559", "964",
]

PILOT_PID = "10542"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _problems_dir_for(pid: str) -> Path:
    cf = Path("data/cf_problems") / f"{pid}.md"
    if cf.exists():
        return Path("data/cf_problems")
    return Path("data/problems")


def _log(pid: str, msg: str):
    print(f"[{pid}] {msg}", flush=True)


def _round_robin_schedule(n: int) -> list[list[tuple[int, int]]]:
    """
    Generate a full round-robin schedule for n players (n must be even).
    Returns list of rounds, each round is a list of (i, j) pairs.
    Total: n-1 rounds, n/2 pairs each. Every pair appears exactly once.
    """
    assert n % 2 == 0
    players = list(range(n))
    schedule = []
    # Circle method: fix player 0, rotate the rest
    fixed = players[0]
    rotating = players[1:]
    for rnd in range(n - 1):
        round_pairs = []
        current = [fixed] + rotating
        for k in range(n // 2):
            i, j = current[k], current[n - 1 - k]
            round_pairs.append((min(i, j), max(i, j)))
        schedule.append(round_pairs)
        # Rotate: move last to front
        rotating = [rotating[-1]] + rotating[:-1]
    return schedule


# ── Per-problem pipeline ────────────────────────────────────────────────────

def run_problem(pid: str, executor: ThreadPoolExecutor):
    problems_dir = _problems_dir_for(pid)
    out_dir = OUT_BASE / pid
    sol_dir = out_dir / "solutions"
    jud_dir = out_dir / "judge_results"
    cmp_file = out_dir / "comparisons.jsonl"
    sol_dir.mkdir(parents=True, exist_ok=True)
    jud_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    _log(pid, "starting")

    # ── Step 1: Prepare 40 solutions ─────────────────────────────────────
    # Copy existing gen0 solutions (sol00-sol19) from evolve dir
    evolve_sol_dir = EVOLVE_DIR / pid / "solutions"
    existing_count = 0
    for i in range(20):
        src = evolve_sol_dir / f"gen0_sol{i:02d}.md"
        dst = sol_dir / f"sol{i:02d}.md"
        if dst.exists():
            existing_count += 1
            continue
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            existing_count += 1

    # Check how many of sol00-sol39 already exist
    all_exist = all((sol_dir / f"sol{i:02d}.md").exists() for i in range(POP_SIZE))
    if all_exist:
        _log(pid, f"step1: all {POP_SIZE} solutions already on disk")
    else:
        # Generate the remaining (sol20-sol39)
        n_have = sum(1 for i in range(POP_SIZE) if (sol_dir / f"sol{i:02d}.md").exists())
        n_need = POP_SIZE - n_have
        _log(pid, f"step1: have {n_have}, generating {n_need} more ...")

        problem_text = (problems_dir / f"{pid}.md").read_text(encoding="utf-8")
        from opendeepthink.llm_client import call_llm

        SYSTEM_PROMPT = "You are an expert competitive programmer.\nOutput your solution as a single ```cpp ... ``` block, preceded by brief reasoning."

        def _gen_one(idx):
            md = call_llm(
                problem_text,
                provider="gemini",
                system_prompt=SYSTEM_PROMPT,
                temperature=1.0,
            )
            (sol_dir / f"sol{idx:02d}.md").write_text(md, encoding="utf-8")
            return idx

        missing = [i for i in range(POP_SIZE) if not (sol_dir / f"sol{i:02d}.md").exists()]
        futures = {executor.submit(_gen_one, i): i for i in missing}
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % 5 == 0:
                _log(pid, f"  step1: generated {done}/{n_need}")
            f.result()  # raise if error

        _log(pid, f"step1: all {POP_SIZE} solutions ready")

    # ── Step 2: Submit all 40 for judging ────────────────────────────────
    n_judged = sum(1 for i in range(POP_SIZE) if (jud_dir / f"sol{i:02d}.json").exists())
    if n_judged == POP_SIZE:
        _log(pid, f"step2: all {POP_SIZE} judge results on disk")
    else:
        _log(pid, f"step2: {n_judged}/{POP_SIZE} judged, submitting rest ...")
        sol_files = [sol_dir / f"sol{i:02d}.md" for i in range(POP_SIZE)]
        submit_all(pid, sol_files, jud_dir)
        _log(pid, f"step2: judging complete")

    # ── Step 3: Round-robin pairwise comparisons ─────────────────────────
    # Load solutions
    solutions = []
    for i in range(POP_SIZE):
        md = (sol_dir / f"sol{i:02d}.md").read_text(encoding="utf-8")
        solutions.append(md)
    codes = [_extract_code(s) for s in solutions]
    problem_text = (problems_dir / f"{pid}.md").read_text(encoding="utf-8")

    # Check existing comparisons (resume support)
    existing_cmps = set()
    if cmp_file.exists():
        for line in cmp_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing_cmps.add((rec["round"], rec["idx_a"], rec["idx_b"]))

    # Generate round-robin schedule
    schedule = _round_robin_schedule(POP_SIZE)
    total_pairs = sum(len(rnd) for rnd in schedule)
    _log(pid, f"step3: round-robin {N_ROUNDS} rounds × {POP_SIZE//2} pairs = {total_pairs} comparisons")

    # Build work items (skip already done)
    jobs = []
    for rnd_idx, pairs in enumerate(schedule):
        for ia, ib in pairs:
            if (rnd_idx, ia, ib) not in existing_cmps:
                jobs.append((rnd_idx, ia, ib))

    _log(pid, f"step3: {total_pairs - len(jobs)} already done, {len(jobs)} remaining")

    if jobs:
        lock = threading.Lock()

        def _do_cmp(rnd_idx, ia, ib):
            for attempt in range(5):
                try:
                    result = _judge_pair(problem_text, codes[ia], codes[ib])
                    record = {"round": rnd_idx, "idx_a": ia, "idx_b": ib, **result}
                    with lock:
                        with open(cmp_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record) + "\n")
                    return record
                except Exception as e:
                    if attempt < 4:
                        time.sleep(5 * (attempt + 1))
                    else:
                        raise

        futures = {
            executor.submit(_do_cmp, rnd, ia, ib): (rnd, ia, ib)
            for rnd, ia, ib in jobs
        }
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % 20 == 0:
                _log(pid, f"  step3: {done}/{len(jobs)} comparisons")
            f.result()

    _log(pid, f"step3: all {total_pairs} comparisons complete")

    # ── Summary ──────────────────────────────────────────────────────────
    # Load ground truth
    ac_count = 0
    for i in range(POP_SIZE):
        jp = jud_dir / f"sol{i:02d}.json"
        if jp.exists():
            d = json.loads(jp.read_text())
            if d.get("result", {}).get("passed", False):
                ac_count += 1

    elapsed = time.monotonic() - t0
    _log(pid, f"done in {elapsed/60:.1f} min — {ac_count}/{POP_SIZE} AC")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Run all 10 problems (default: pilot with 1)")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    pids = ALL_PIDS if args.all else [PILOT_PID]

    print(f"\n{'='*60}", flush=True)
    print(f"[bt_scaling] {len(pids)} problems: {pids}", flush=True)
    print(f"[bt_scaling] {POP_SIZE} solutions × {N_ROUNDS} rounds = {N_ROUNDS * (POP_SIZE // 2)} comparisons/problem", flush=True)
    print(f"[bt_scaling] workers={args.workers}", flush=True)

    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for pid in pids:
            run_problem(pid, executor)

    elapsed = time.monotonic() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"[bt_scaling] all done in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
