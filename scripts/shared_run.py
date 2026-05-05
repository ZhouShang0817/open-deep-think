"""
Run evolution for multiple problems in parallel, sharing a single thread pool.

All API calls (generate / judge / mutate) from all problems compete for the
same N worker slots, so slow stragglers in one problem don't leave workers idle
— other problems' jobs fill in immediately.

Each problem still respects its own stage dependencies:
  gen0_generate → gen1_tournament → gen1_mutate → ... → final_tournament

Usage:
  python scripts/shared_run.py --pids 2137g 2138d
  python scripts/shared_run.py --pids 2137g 2138d --workers 20
  python scripts/shared_run.py --pids 2139a 2140b 2141c 2142d 2143e --workers 20
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opendeepthink.generate import generate
from opendeepthink.judge    import run_tournament
from opendeepthink.bt       import compute_bt_scores, rank_solutions
from opendeepthink.mutate   import mutate_batch
from opendeepthink.submit   import submit_fire_and_forget

# ── constants ─────────────────────────────────────────────────────────────────
POP_SIZE     = 20
N_ELITE      = 5
N_ELIMINATE  = 5
N_MUTATE     = POP_SIZE - N_ELIMINATE   # 15
EVAL_ROUNDS  = 4
FINAL_ROUNDS = 10
GENERATIONS  = 3

OUT_BASE = Path("data/evolve")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_solutions_if_exist(sol_dir: Path, gen: int, n: int) -> list[str] | None:
    """Return list of solution strings if all n files exist for gen, else None."""
    files = [sol_dir / f"gen{gen}_sol{i:02d}.md" for i in range(n)]
    if all(f.exists() for f in files):
        return [f.read_text(encoding="utf-8") for f in files]
    return None


def _load_comparisons_if_complete(cmp_file: Path, expected: int) -> list[dict] | None:
    """Return parsed comparisons if file has >= expected lines, else None."""
    if not cmp_file.exists():
        return None
    lines = [l for l in cmp_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) >= expected:
        return [json.loads(l) for l in lines]
    return None


def _save_solutions(solutions: list[str], sol_dir: Path, gen: int):
    sol_dir.mkdir(parents=True, exist_ok=True)
    for i, md in enumerate(solutions):
        (sol_dir / f"gen{gen}_sol{i:02d}.md").write_text(md, encoding="utf-8")


def _save_ranking(scores: list[float], ranking: list[int], path: Path, label: str = ""):
    data = {
        "label":   label,
        "ranking": ranking,
        "scores":  {str(i): round(scores[i], 4) for i in range(len(scores))},
        "top5":    ranking[:5],
        "bottom5": ranking[-5:],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _log(pid: str, msg: str):
    print(f"[{pid}] {msg}", flush=True)


# ── per-problem evolution (runs in its own coordinator thread) ─────────────────

def evolve_problem(
    pid: str,
    executor: ThreadPoolExecutor,
    seed: int = 42,
    problems_dir: Path = Path("data/cf_problems"),
):
    rng = random.Random(seed)
    out_dir  = OUT_BASE / pid
    sol_dir  = out_dir / "solutions"
    cmp_dir  = out_dir / "comparisons"
    rank_dir = out_dir / "rankings"
    jud_dir  = out_dir / "judge_results"
    for d in (sol_dir, cmp_dir, rank_dir, jud_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    _log(pid, f"starting  (seed={seed})")

    # ── Gen 0: initial population ──────────────────────────────────────────
    solutions = _load_solutions_if_exist(sol_dir, gen=0, n=POP_SIZE)
    if solutions is not None:
        _log(pid, f"gen0: resuming — all {POP_SIZE} solutions already on disk")
    else:
        _log(pid, f"gen0: generating {POP_SIZE} solutions ...")
        solutions = generate(pid, n=POP_SIZE, problems_dir=problems_dir,
                             executor=executor, sol_dir=sol_dir, gen=0)
        submit_fire_and_forget(pid, sorted(sol_dir.glob("gen0_sol*.md")), jud_dir)
        _log(pid, f"gen0: done + submitted (bg)")

    # ── Generations 1..N ──────────────────────────────────────────────────
    expected_pairs = EVAL_ROUNDS * (POP_SIZE // 2)
    for gen in range(1, GENERATIONS + 1):
        cmp_file = cmp_dir / f"gen{gen}.jsonl"

        # Resume: reload completed tournament
        comparisons = _load_comparisons_if_complete(cmp_file, expected=expected_pairs)
        if comparisons is not None:
            _log(pid, f"gen{gen}: resuming — tournament already complete ({len(comparisons)} pairs)")
        else:
            _log(pid, f"gen{gen}: tournament ({EVAL_ROUNDS} rounds × {POP_SIZE//2} pairs) ...")
            comparisons = run_tournament(
                pid, solutions,
                n_rounds=EVAL_ROUNDS,
                rng=rng,
                out_file=cmp_file,
                label=f"{pid}/gen{gen}",
                problems_dir=problems_dir,
                executor=executor,
            )

        scores  = compute_bt_scores(POP_SIZE, comparisons)
        ranking = rank_solutions(scores)
        _save_ranking(scores, ranking, rank_dir / f"gen{gen}.json", label=f"gen{gen}")

        elite_idx  = ranking[:N_ELITE]
        mutate_idx = ranking[:N_MUTATE]

        # Resume: reload completed gen solutions if already saved
        existing = _load_solutions_if_exist(sol_dir, gen=gen, n=POP_SIZE)
        if existing is not None:
            _log(pid, f"gen{gen}: resuming — all {POP_SIZE} solutions already on disk")
            solutions = existing
            continue

        _log(pid, f"gen{gen}: BT top-5={elite_idx}  mutating {N_MUTATE} ...")

        # Write elite copies to bottom-5 slots immediately (no LLM needed)
        for bottom_i, elite_i in zip(ranking[-N_ELITE:], ranking[:N_ELITE]):
            (sol_dir / f"gen{gen}_sol{bottom_i:02d}.md").write_text(
                solutions[elite_i], encoding="utf-8"
            )

        # Mutate top-15; each result saved to disk as it arrives
        mutations = mutate_batch(
            pid, solutions, comparisons, mutate_idx,
            problems_dir=problems_dir,
            executor=executor,
            sol_dir=sol_dir, gen=gen,
        )

        next_solutions = list(solutions)
        for i in mutate_idx:
            next_solutions[i] = mutations[i]
        for bottom_i, elite_i in zip(ranking[-N_ELITE:], ranking[:N_ELITE]):
            next_solutions[bottom_i] = solutions[elite_i]
        solutions = next_solutions

        _save_solutions(solutions, sol_dir, gen=gen)
        submit_fire_and_forget(pid, sorted(sol_dir.glob(f"gen{gen}_sol*.md")), jud_dir)
        _log(pid, f"gen{gen}: saved + submitted (bg)")

    # ── Final tournament ───────────────────────────────────────────────────
    cmp_file          = cmp_dir / "final.jsonl"
    expected_final    = FINAL_ROUNDS * (POP_SIZE // 2)
    final_comparisons = _load_comparisons_if_complete(cmp_file, expected=expected_final)
    if final_comparisons is not None:
        _log(pid, f"final: resuming — tournament already complete ({len(final_comparisons)} pairs)")
    else:
        _log(pid, f"final: {FINAL_ROUNDS}-round tournament ...")
        final_comparisons = run_tournament(
            pid, solutions,
            n_rounds=FINAL_ROUNDS,
            rng=rng,
            out_file=cmp_file,
            label=f"{pid}/final",
            problems_dir=problems_dir,
            executor=executor,
        )
    final_scores  = compute_bt_scores(POP_SIZE, final_comparisons)
    final_ranking = rank_solutions(final_scores)
    _save_ranking(final_scores, final_ranking, rank_dir / "final.json", label="final")
    submit_fire_and_forget(pid, sorted(sol_dir.glob(f"gen{GENERATIONS}_sol*.md")), jud_dir)

    elapsed = time.monotonic() - t0
    winner  = final_ranking[0]
    _log(pid, f"done in {elapsed/60:.1f} min  winner=sol{winner:02d}  top5={final_ranking[:5]}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids",         nargs="+", required=True)
    ap.add_argument("--workers",      type=int, default=20)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--problems-dir", default="data/cf_problems")
    args = ap.parse_args()

    problems_dir = Path(args.problems_dir)
    pids = args.pids

    print(f"\n{'='*60}", flush=True)
    print(f"[shared_run] {len(pids)} problems: {pids}", flush=True)
    print(f"[shared_run] shared pool workers={args.workers}", flush=True)

    t_start = time.monotonic()

    # One shared executor for all problems
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        threads = []
        for pid in pids:
            t = threading.Thread(
                target=evolve_problem,
                args=(pid, executor),
                kwargs={"seed": args.seed, "problems_dir": problems_dir},
                name=f"coord-{pid}",
                daemon=False,
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    elapsed = time.monotonic() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"[shared_run] all done in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
