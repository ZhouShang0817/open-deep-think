"""
HLE single-question evolution: gen0 → BT → mutate → BT → ...

Usage:
  python hle/evolve_one.py --all --workers 20                        # gen0→gen1 (4 rounds)
  python hle/evolve_one.py --all --start_gen 1 --n_rounds 6          # gen1→gen2 (6 rounds)
  python hle/evolve_one.py --qid <id> --start_gen 1 --n_rounds 6
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hle.judge import run_tournament
from hle.mutate import mutate_batch
from hle.utils import extract_answer, is_correct
from opendeepthink.bt import compute_bt_scores, rank_solutions
from tqdm import tqdm

PASSK_DIR  = Path("data/hle/passk_results")
BT_DIR     = Path("data/hle/bt_results")
EVOLVE_DIR = Path("data/hle/evolve")
HLE_GOLD   = Path("data/hle/hle_gold.json")

N_SOLUTIONS = 12
N_ELITE     = 3   # top 1/4 kept as-is
N_MUTATE    = 9   # remaining 3/4 mutated


def load_gen0(qid: str, question: str) -> list[str]:
    attempts = sorted(
        json.loads((PASSK_DIR / f"{qid}.json").read_text()),
        key=lambda a: a["attempt"]
    )
    return [a["raw_response"] for a in attempts]


def load_gen0_comparisons(qid: str) -> list[dict]:
    comp_file = BT_DIR / f"{qid}_comparisons.jsonl"
    if comp_file.exists():
        return [json.loads(l) for l in comp_file.read_text().splitlines() if l.strip()]
    return []


def run_gen(
    qid: str,
    question: str,
    solutions: list[str],
    gen: int,
    n_rounds: int,
    workers: int,
    rng: random.Random,
    out_dir: Path,
    existing_comps: list[dict] | None = None,
) -> tuple[list[str], list[dict], list[float], list[int]]:
    n = len(solutions)

    if existing_comps is not None:
        tqdm.write(f"  gen{gen}: reusing {len(existing_comps)} existing comparisons")
        comparisons = existing_comps
    else:
        comp_file = out_dir / f"gen{gen}_comparisons.jsonl"
        tqdm.write(f"  gen{gen}: BT tournament ({n_rounds} rounds × {n//2} pairs = {n_rounds*n//2} calls)")
        comparisons = run_tournament(
            question=question,
            solutions=solutions,
            n_rounds=n_rounds,
            workers=workers,
            rng=rng,
            out_file=comp_file,
            label=f"{qid[:8]} gen{gen}",
        )

    scores  = compute_bt_scores(n, comparisons)
    ranking = rank_solutions(scores)

    gen_data = {
        "gen":       gen,
        "qid":       qid,
        "n_rounds":  n_rounds if existing_comps is None else "reused",
        "ranking":   ranking,
        "bt_scores": scores,
        "solutions": [
            {
                "idx":       i,
                "rank":      ranking.index(i),
                "predicted": extract_answer(solutions[i]),
                "response":  solutions[i],
            }
            for i in range(n)
        ],
    }
    (out_dir / f"gen{gen}.json").write_text(
        json.dumps(gen_data, indent=2, ensure_ascii=False)
    )
    return solutions, comparisons, scores, ranking


def evolve_step(
    qid: str,
    question: str,
    gt: str,
    start_gen: int,
    n_rounds: int,
    workers: int,
    rng: random.Random,
) -> dict:
    """Run one evolution step: gen{start_gen} → BT → mutate → gen{start_gen+1} → BT."""
    out_dir = EVOLVE_DIR / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    src_gen = start_gen
    dst_gen = start_gen + 1

    tqdm.write(f"\n{'─'*60}")
    tqdm.write(f"  QID: {qid[:8]}  gen{src_gen}→gen{dst_gen}  {n_rounds}rounds  GT: {gt[:50]}")

    # ── Load source generation solutions + comparisons ──
    if src_gen == 0:
        src_sols  = load_gen0(qid, question)
        src_comps = load_gen0_comparisons(qid)
    else:
        src_sols  = json.loads((out_dir / f"gen{src_gen}_raw.json").read_text())["solutions"]
        comp_file = out_dir / f"gen{src_gen}_comparisons.jsonl"
        src_comps = [json.loads(l) for l in comp_file.read_text().splitlines() if l.strip()]

    # ── BT ranking on source gen (reuse existing comparisons, no new calls) ──
    sols_src, comps_src, _, ranking_src = run_gen(
        qid, question, src_sols, gen=src_gen,
        n_rounds=n_rounds, workers=workers, rng=rng,
        out_dir=out_dir, existing_comps=src_comps if src_comps else None,
    )

    best_src   = ranking_src[0]
    pred_src   = extract_answer(sols_src[best_src])
    correct_src = is_correct(pred_src, gt)
    tqdm.write(f"  gen{src_gen} BT#1: {pred_src!r}  {'✓' if correct_src else '✗'}")

    # ── Mutate ──
    elite_indices  = set(ranking_src[:N_ELITE])
    mutate_indices = list(ranking_src[N_ELITE:])
    tqdm.write(f"  mutating {len(mutate_indices)} solutions (elite: {sorted(elite_indices)})...")

    mutated = mutate_batch(
        question=question,
        solutions=sols_src,
        comparisons=comps_src,
        indices=mutate_indices,
        workers=workers,
    )

    sols_dst = []
    for i in range(N_SOLUTIONS):
        if i in elite_indices:
            sols_dst.append(sols_src[i])
        else:
            sols_dst.append(mutated.get(i, sols_src[i]))

    raw_file = out_dir / f"gen{dst_gen}_raw.json"
    raw_file.write_text(json.dumps(
        {"gen": dst_gen, "qid": qid, "solutions": sols_dst},
        indent=2, ensure_ascii=False
    ))

    # ── BT on destination generation ──
    sols_dst, comps_dst, _, ranking_dst = run_gen(
        qid, question, sols_dst, gen=dst_gen,
        n_rounds=n_rounds, workers=workers, rng=rng,
        out_dir=out_dir,
    )

    best_dst    = ranking_dst[0]
    pred_dst    = extract_answer(sols_dst[best_dst])
    correct_dst = is_correct(pred_dst, gt)
    tqdm.write(f"  gen{dst_gen} BT#1: {pred_dst!r}  {'✓' if correct_dst else '✗'}")

    # ── Update result.json ──
    result_path = out_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"qid": qid, "gt": gt}
    result[f"gen{src_gen}_bt_predicted"] = pred_src
    result[f"gen{src_gen}_bt_correct"]   = correct_src
    result[f"gen{dst_gen}_bt_predicted"] = pred_dst
    result[f"gen{dst_gen}_bt_correct"]   = correct_dst
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def print_summary(results: list[dict], start_gen: int) -> None:
    n = len(results)
    src, dst = start_gen, start_gen + 1
    k0 = f"gen{src}_bt_correct"
    k1 = f"gen{dst}_bt_correct"
    c0 = sum(1 for r in results if r.get(k0))
    c1 = sum(1 for r in results if r.get(k1))
    rescued   = sum(1 for r in results if not r.get(k0) and r.get(k1))
    regressed = sum(1 for r in results if r.get(k0) and not r.get(k1))
    print(f"\n{'='*50}")
    print(f"  题数              : {n}")
    print(f"  gen{src} BT#1     : {c0}/{n} ({100*c0/n:.1f}%)")
    print(f"  gen{dst} BT#1     : {c1}/{n} ({100*c1/n:.1f}%)")
    print(f"  救回 ({src}→{dst}) : {rescued}")
    print(f"  退步 ({dst}→{src}) : {regressed}")
    print(f"  参考 pass@12 UB   : 72.0%")
    print(f"{'='*50}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid",       type=str, default=None)
    ap.add_argument("--all",       action="store_true")
    ap.add_argument("--workers",   type=int, default=20)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--n_rounds",  type=int, default=4, help="BT rounds per generation")
    ap.add_argument("--start_gen", type=int, default=0, help="Start from this gen (0=gen0→gen1, 1=gen1→gen2)")
    args = ap.parse_args()

    hle_gold = {item["id"]: item for item in json.loads(HLE_GOLD.read_text())}
    passk_summary = json.loads((PASSK_DIR / "summary.json").read_text())
    all_qids = list(passk_summary["per_question"].keys())

    rng = random.Random(args.seed)
    EVOLVE_DIR.mkdir(parents=True, exist_ok=True)

    if args.qid:
        qids = [args.qid]
    elif args.all:
        qids = all_qids
    else:
        ap.error("Specify --qid <id> or --all")

    dst_gen = args.start_gen + 1
    done_key = f"gen{dst_gen}_bt_correct"

    results = []
    outer = tqdm(total=len(qids), desc=f"gen{args.start_gen}→gen{dst_gen}", unit="q", position=0)

    for qid in qids:
        result_path = EVOLVE_DIR / qid / "result.json"
        # Resume: skip if destination gen already computed
        if result_path.exists():
            r = json.loads(result_path.read_text())
            if done_key in r:
                results.append(r)
                tqdm.write(f"[cached] {qid[:8]}  gen{dst_gen}={'✓' if r[done_key] else '✗'}")
                outer.update(1)
                continue

        item = hle_gold.get(qid)
        if not item:
            tqdm.write(f"[skip] {qid} not in hle_gold")
            outer.update(1)
            continue

        # Check source gen exists
        src_raw = EVOLVE_DIR / qid / f"gen{args.start_gen}_raw.json"
        if args.start_gen > 0 and not src_raw.exists():
            tqdm.write(f"[skip] {qid[:8]} gen{args.start_gen}_raw.json not found")
            outer.update(1)
            continue

        r = evolve_step(
            qid=qid,
            question=item["question"],
            gt=item["answer"],
            start_gen=args.start_gen,
            n_rounds=args.n_rounds,
            workers=args.workers,
            rng=rng,
        )
        results.append(r)
        outer.update(1)

    outer.close()
    print_summary(results, args.start_gen)

    summary_path = EVOLVE_DIR / "summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    existing.update({
        "n": len(results),
        f"gen{dst_gen}_correct": sum(1 for r in results if r.get(done_key)),
        "results": results,
    })
    summary_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"Saved to {summary_path}")


if __name__ == "__main__":
    main()
