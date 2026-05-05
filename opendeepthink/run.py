"""
Evolutionary solver for a single competitive programming problem.

Flow:
  Gen 0  : generate 20 solutions, save to solutions/gen0_sol{i:02d}.md
  Gen 1-3: tournament → BT rank → keep top-5 elite + mutate top-15 → save
  Final  : 10-round tournament → rank → save
  Submit : all solutions from all generations to judge

Usage:
  python evolve/run.py --pid 6537
  python evolve/run.py --pid 6537 --workers 20 --generations 3 --seed 42
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
from pathlib import Path

from opendeepthink.generate import generate
from opendeepthink.judge    import run_tournament
from opendeepthink.bt       import compute_bt_scores, rank_solutions
from opendeepthink.mutate   import mutate_batch
from opendeepthink.submit   import submit_fire_and_forget

# ── constants ─────────────────────────────────────────────────────────────────
POP_SIZE      = 20
N_ELITE       = 5
N_ELIMINATE   = 5   # bottom 5 whose slots get replaced by mutations
EVAL_ROUNDS   = 4
FINAL_ROUNDS  = 10
WORKERS       = 20

OUT_BASE = Path("data/evolve")


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


def run_evolution(pid: str, workers: int = WORKERS, seed: int = 42,
                  pop: int = POP_SIZE, generations: int = 3,
                  problems_dir: Path = Path("data/cf_problems")) -> Path:
    rng = random.Random(seed)
    out_dir   = OUT_BASE / pid
    sol_dir   = out_dir / "solutions"
    cmp_dir   = out_dir / "comparisons"
    rank_dir  = out_dir / "rankings"
    judge_dir = out_dir / "judge_results"
    for d in (sol_dir, cmp_dir, rank_dir, judge_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_elite    = N_ELITE
    n_mutate   = pop - N_ELIMINATE   # top 15

    print(f"\n{'='*60}", flush=True)
    print(f"[evolve] pid={pid}  pop={pop}  generations={generations}  seed={seed}", flush=True)
    print(f"[evolve] out_dir={out_dir}", flush=True)

    expected_pairs = EVAL_ROUNDS * (pop // 2)

    def _load_solutions(gen: int) -> list[str] | None:
        files = [sol_dir / f"gen{gen}_sol{i:02d}.md" for i in range(pop)]
        if all(f.exists() for f in files):
            return [f.read_text(encoding="utf-8") for f in files]
        return None

    def _load_comparisons(cmp_file: Path, expected: int) -> list[dict] | None:
        if not cmp_file.exists():
            return None
        lines = [l for l in cmp_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) >= expected:
            return [json.loads(l) for l in lines]
        return None

    # ── Gen 0: initial population ──────────────────────────────────────────────
    solutions = _load_solutions(0)
    if solutions is not None:
        print(f"\n[gen 0] resuming — all {pop} solutions already on disk", flush=True)
    else:
        print(f"\n[gen 0] generating {pop} initial solutions ...", flush=True)
        solutions = generate(pid, n=pop, workers=workers, problems_dir=problems_dir)
        _save_solutions(solutions, sol_dir, gen=0)
        submit_fire_and_forget(pid, sorted(sol_dir.glob("gen0_sol*.md")), judge_dir)
        print(f"[gen 0] saved + submitted (bg) → {sol_dir}", flush=True)

    # ── Generations 1..N ──────────────────────────────────────────────────────
    for gen in range(1, generations + 1):
        print(f"\n{'─'*60}", flush=True)

        # Resume: skip entire gen if solutions already saved
        existing = _load_solutions(gen)
        if existing is not None:
            print(f"[gen {gen}] resuming — all {pop} solutions already on disk", flush=True)
            solutions = existing
            continue

        cmp_file = cmp_dir / f"gen{gen}.jsonl"
        comparisons = _load_comparisons(cmp_file, expected=expected_pairs)
        if comparisons is not None:
            print(f"[gen {gen}] resuming — tournament already complete ({len(comparisons)} pairs)", flush=True)
        else:
            print(f"[gen {gen}] tournament: {EVAL_ROUNDS} rounds × {pop//2} pairs = {EVAL_ROUNDS*pop//2} comparisons", flush=True)
            comparisons = run_tournament(
                pid, solutions,
                n_rounds=EVAL_ROUNDS,
                workers=workers,
                rng=rng,
                out_file=cmp_file,
                label=f"gen{gen}",
                problems_dir=problems_dir,
            )

        scores  = compute_bt_scores(pop, comparisons)
        ranking = rank_solutions(scores)
        _save_ranking(scores, ranking, rank_dir / f"gen{gen}.json", label=f"gen{gen}")

        elite_idx  = ranking[:n_elite]
        mutate_idx = ranking[:n_mutate]
        print(f"[gen {gen}] BT top-5: sol{elite_idx}  scores={[round(scores[i],2) for i in elite_idx]}", flush=True)
        print(f"[gen {gen}] mutating {len(mutate_idx)} solutions (top {n_mutate}) ...", flush=True)

        mutations = mutate_batch(pid, solutions, comparisons, mutate_idx, workers=workers,
                                 problems_dir=problems_dir)

        next_solutions = list(solutions)
        for i in mutate_idx:
            next_solutions[i] = mutations[i]
        for bottom_i, elite_i in zip(ranking[-n_elite:], ranking[:n_elite]):
            next_solutions[bottom_i] = solutions[elite_i]
        solutions = next_solutions

        _save_solutions(solutions, sol_dir, gen=gen)
        submit_fire_and_forget(pid, sorted(sol_dir.glob(f"gen{gen}_sol*.md")), judge_dir)
        print(f"[gen {gen}] saved + submitted (bg) → {sol_dir}", flush=True)

    # ── Final tournament ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)

    cmp_file = cmp_dir / "final.jsonl"
    final_comparisons = _load_comparisons(cmp_file, expected=FINAL_ROUNDS * (pop // 2))
    if final_comparisons is not None:
        print(f"[final] resuming — tournament already complete ({len(final_comparisons)} pairs)", flush=True)
    else:
        print(f"[final] {FINAL_ROUNDS}-round tournament ({FINAL_ROUNDS*pop//2} comparisons) ...", flush=True)
        final_comparisons = run_tournament(
            pid, solutions,
            n_rounds=FINAL_ROUNDS,
            workers=workers,
            rng=rng,
            out_file=cmp_file,
            label="final",
            problems_dir=problems_dir,
        )
    final_scores  = compute_bt_scores(pop, final_comparisons)
    final_ranking = rank_solutions(final_scores)
    _save_ranking(final_scores, final_ranking, rank_dir / "final.json", label="final")
    submit_fire_and_forget(pid, sorted(sol_dir.glob(f"gen{generations}_sol*.md")), judge_dir)

    winner_idx = final_ranking[0]
    print(f"[final] winner: sol{winner_idx:02d}  BT={final_scores[winner_idx]:.3f}", flush=True)
    print(f"[final] top-5: sol{final_ranking[:5]}", flush=True)

    return out_dir



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid",          required=True)
    ap.add_argument("--workers",      type=int, default=WORKERS)
    ap.add_argument("--pop",          type=int, default=POP_SIZE)
    ap.add_argument("--generations",  type=int, default=3)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--problems-dir", default="data/cf_problems")
    args = ap.parse_args()

    run_evolution(
        pid=args.pid,
        workers=args.workers,
        seed=args.seed,
        pop=args.pop,
        generations=args.generations,
        problems_dir=Path(args.problems_dir),
    )


if __name__ == "__main__":
    main()
