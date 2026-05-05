"""
Analyze HLE evolution results.
Merges v1 (evolve/, 50q) and v2 (evolve_v2/, 50q) into a single 100-question analysis.

Usage:
  python hle/analyze.py           # combined v1+v2
  python hle/analyze.py --v1      # v1 only (original 50q)
"""
from __future__ import annotations

import argparse
import json
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from collections import defaultdict
from pathlib import Path
from opendeepthink.bt import compute_bt_scores, rank_solutions
from hle.utils import extract_answer, normalize, is_correct

HLE_GOLD_PATH = Path("data/hle/hle_gold.json")
hle_gold      = {item["id"]: item for item in json.loads(HLE_GOLD_PATH.read_text())}

W = 72  # default separator width


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_comp(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pass1_rate(sols: list[str], gt: str) -> tuple[int, int]:
    correct = sum(1 for s in sols if is_correct(extract_answer(s), gt))
    return correct, len(sols)


def _rank_correct(sols: list[str], ranking: list[int], gt: str) -> list[bool]:
    return [is_correct(extract_answer(sols[ranking[i]]), gt) for i in range(len(sols))]


def _majority_vote(preds: list[str | None], gt: str) -> bool:
    """Most-common predicted answer (by normalized form) vs ground truth."""
    from collections import Counter
    valid = [p for p in preds if p is not None]
    if not valid:
        return False
    winner_norm = Counter(normalize(p) for p in valid).most_common(1)[0][0]
    return normalize(gt) == winner_norm


def _load_rows(evolve_dir: Path, passk_dir: Path, gen0_comp_fn) -> list[dict]:
    rows = []
    if not evolve_dir.exists():
        return rows

    for qdir in sorted(evolve_dir.iterdir()):
        if not qdir.is_dir() or not (qdir / "result.json").exists():
            continue
        qid = qdir.name
        if qid not in hle_gold:
            continue
        gt  = normalize(hle_gold[qid]["answer"])
        cat = hle_gold[qid].get("category", "Unknown")

        passk_file = passk_dir / f"{qid}.json"
        if not passk_file.exists():
            continue
        passk     = sorted(json.loads(passk_file.read_text()), key=lambda a: a["attempt"])
        gen0_sols = [a["raw_response"] for a in passk]
        gen0_preds = [a["predicted"] for a in passk]
        n         = len(gen0_sols)

        comp0  = _load_comp(gen0_comp_fn(qid, qdir))
        rank0  = rank_solutions(compute_bt_scores(n, comp0)) if comp0 else list(range(n))
        c0, n0 = _pass1_rate(gen0_sols, gt)

        row = {
            "qid":           qid,
            "category":      cat,
            "gt_short":      hle_gold[qid]["answer"][:30],
            "c0": c0, "n0": n0,
            "rank0_correct": _rank_correct(gen0_sols, rank0, gt),
            "mv0_correct":   _majority_vote(gen0_preds, gt),
        }

        for gen, raw_key, comp_key in [(1, "gen1_raw.json", "gen1_comparisons.jsonl"),
                                        (2, "gen2_raw.json", "gen2_comparisons.jsonl")]:
            g_raw  = qdir / raw_key
            g_comp = qdir / comp_key
            if g_raw.exists() and g_comp.exists():
                sols  = json.loads(g_raw.read_text())["solutions"]
                preds = [extract_answer(s) for s in sols]
                comps = _load_comp(g_comp)
                rank  = rank_solutions(compute_bt_scores(n, comps))
                c, _  = _pass1_rate(sols, gt)
                row[f"c{gen}"] = c
                row[f"n{gen}"] = n
                row[f"rank{gen}_correct"] = _rank_correct(sols, rank, gt)
                row[f"mv{gen}_correct"]   = _majority_vote(preds, gt)

        res = json.loads((qdir / "result.json").read_text())
        for g in (0, 1, 2):
            row[f"gen{g}_bt_correct"] = res.get(f"gen{g}_bt_correct")

        rows.append(row)
    return rows


def load_rows_v1() -> list[dict]:
    bt_dir = Path("data/hle/bt_results")
    return _load_rows(
        Path("data/hle/evolve"),
        Path("data/hle/passk_results"),
        lambda qid, qdir: bt_dir / f"{qid}_comparisons.jsonl",
    )


def load_rows_v2() -> list[dict]:
    return _load_rows(
        Path("data/hle/evolve_v2"),
        Path("data/hle/passk_results_v2"),
        lambda qid, qdir: qdir / "gen0_comparisons.jsonl",
    )


# ── Table 1: Summary ──────────────────────────────────────────────────────────

def print_summary(rows: list[dict]) -> None:
    """Overall pass@k and BT#1 accuracy across generations."""
    n  = len(rows)
    g1 = [r for r in rows if "c1" in r]
    g2 = [r for r in rows if "c2" in r]

    def _avg(rs, key_c, key_n):
        return sum(r[key_c] / r[key_n] for r in rs) / len(rs) if rs else None

    p0 = _avg(rows, "c0", "n0")
    p1 = _avg(g1, "c1", "n1")
    p2 = _avg(g2, "c2", "n2")

    # BT#1 accuracy from result.json
    b0 = sum(1 for r in rows if r.get("gen0_bt_correct")) / n if n else None
    b1 = sum(1 for r in rows if r.get("gen1_bt_correct")) / len(g1) if g1 else None
    b2 = sum(1 for r in rows if r.get("gen2_bt_correct")) / len(g2) if g2 else None

    # Majority vote accuracy
    mv0 = sum(1 for r in rows if r.get("mv0_correct")) / n if n else None
    mv1 = sum(1 for r in g1 if r.get("mv1_correct")) / len(g1) if g1 else None
    mv2 = sum(1 for r in g2 if r.get("mv2_correct")) / len(g2) if g2 else None

    # rescued / regressed (gen0 BT vs gen2 BT)
    g2_both = [r for r in rows if r.get("gen0_bt_correct") is not None
               and r.get("gen2_bt_correct") is not None]
    rescued   = sum(1 for r in g2_both if not r["gen0_bt_correct"] and r["gen2_bt_correct"])
    regressed = sum(1 for r in g2_both if r["gen0_bt_correct"] and not r["gen2_bt_correct"])

    print(f"\n{'─'*W}")
    print(f"表1: 总体结果  (N={n} 题)")
    print(f"{'─'*W}")
    print(f"  {'':20}  {'gen0':>8}  {'gen1':>8}  {'gen2':>8}  {'Δ(g0→g2)':>10}")
    print(f"  {'─'*60}")

    def _fmt(v): return f"{v:.1%}" if v is not None else "—"
    def _delta(a, b):
        if a is None or b is None: return "—"
        d = b - a
        return f"{'↑' if d > 0.001 else '↓' if d < -0.001 else '→'}{abs(d):.1%}"

    print(f"  {'pass@1 (empirical)':20}  {_fmt(p0):>8}  {_fmt(p1):>8}  {_fmt(p2):>8}  {_delta(p0,p2):>10}  (avg over {n}/{len(g1)}/{len(g2)} q)")
    print(f"  {'BT#1 accuracy':20}  {_fmt(b0):>8}  {_fmt(b1):>8}  {_fmt(b2):>8}  {_delta(b0,b2):>10}")
    print(f"  {'majority vote':20}  {_fmt(mv0):>8}  {_fmt(mv1):>8}  {_fmt(mv2):>8}  {_delta(mv0,mv2):>10}")
    if g2_both:
        print(f"  {'─'*60}")
        print(f"  {'rescued  (✗→✓)':20}  {rescued:>8}  {'':>8}  {'':>8}  (gen0→gen2, N={len(g2_both)})")
        print(f"  {'regressed(✓→✗)':20}  {regressed:>8}")


# ── Table 2: Category breakdown ───────────────────────────────────────────────

def print_category(rows: list[dict]) -> None:
    """Per-category: empirical pass@1 (raw) and BT#1 accuracy for each generation."""
    cats: dict[str, dict] = defaultdict(lambda: {
        "gold": 0,
        "r0_sum": 0.0, "r1_sum": 0.0, "r2_sum": 0.0,   # raw pass@1 sum
        "r0t": 0,      "r1t": 0,      "r2t": 0,
        "b0c": 0, "b0t": 0, "b1c": 0, "b1t": 0, "b2c": 0, "b2t": 0,  # BT#1
        "m0c": 0, "m0t": 0, "m1c": 0, "m1t": 0, "m2c": 0, "m2t": 0,  # majority vote
    })

    for q in hle_gold.values():
        cats[q.get("category", "Unknown")]["gold"] += 1

    for r in rows:
        cat = r["category"]
        for g, (ck, nk) in enumerate([("c0","n0"), ("c1","n1"), ("c2","n2")]):
            if ck in r:
                cats[cat][f"r{g}_sum"] += r[ck] / r[nk]
                cats[cat][f"r{g}t"]   += 1
        for g, bkey, mkey in [(0, "gen0_bt_correct", "mv0_correct"),
                               (1, "gen1_bt_correct", "mv1_correct"),
                               (2, "gen2_bt_correct", "mv2_correct")]:
            bv = r.get(bkey)
            mv = r.get(mkey)
            if bv is not None:
                cats[cat][f"b{g}t"] += 1
                if bv: cats[cat][f"b{g}c"] += 1
            if mv is not None:
                cats[cat][f"m{g}t"] += 1
                if mv: cats[cat][f"m{g}c"] += 1

    has_g2 = any(d["b2t"] > 0 for d in cats.values())
    has_g1 = any(d["b1t"] > 0 for d in cats.values())

    def _pct(c, t): return f"{100*c/t:.0f}%" if t else "—"
    def _raw(s, t): return f"{100*s/t:.0f}%" if t else "—"
    def _delta(c1, t1, c0, t0):
        return f"{100*c1/t1 - 100*c0/t0:+.0f}%" if t1 and t0 else "—"

    n_gens = 3 if has_g2 else 2 if has_g1 else 1
    # Each gen block: raw(4) BT(4) MV(4) = 14 chars + 2 sep
    W2 = 2 + 22 + 2 + 7 + n_gens * 16 + 8
    print(f"\n{'─'*W2}")
    print(f"表2: 类别准确率  (N={len(rows)} 题)  raw=empirical pass@1  BT=BT#1  MV=majority vote")
    print(f"{'─'*W2}")

    gen_labels = "".join(f"  {'gen'+str(g):14}" for g in range(n_gens))
    print(f"  {'Category':22}  {'n/N':>7}{gen_labels}  {'Δ BT':>6}")
    col_hdr = "".join("  raw   BT   MV" for _ in range(n_gens))
    suffix  = f"  {'g0→g2' if has_g2 else 'g0→g1':>6}"
    print(f"  {'':22}  {'':>7}{col_hdr}{suffix}")
    print(f"  {'─'*(W2-2)}")

    totals = defaultdict(float)

    for cat in sorted(cats, key=lambda c: -cats[c]["gold"]):
        d = cats[cat]
        if d["b0t"] == 0:
            continue
        for k, v in d.items():
            totals[k] += v

        nn   = f"{d['b0t']}/{d['gold']}"
        dbt  = _delta(d[f"b{n_gens-1}c"], d[f"b{n_gens-1}t"], d["b0c"], d["b0t"])
        line = f"  {cat[:22]:22}  {nn:>7}"
        for g in range(n_gens):
            line += f"  {_raw(d[f'r{g}_sum'],d[f'r{g}t']):>4} {_pct(d[f'b{g}c'],d[f'b{g}t']):>4} {_pct(d[f'm{g}c'],d[f'm{g}t']):>4}"
        line += f"  {dbt:>6}"
        print(line)

    print(f"  {'─'*(W2-2)}")
    d   = totals
    nn  = f"{int(d['b0t'])}/{int(d['gold'])}"
    dbt = _delta(d[f"b{n_gens-1}c"], d[f"b{n_gens-1}t"], d["b0c"], d["b0t"])
    line = f"  {'TOTAL':22}  {nn:>7}"
    for g in range(n_gens):
        line += f"  {_raw(d[f'r{g}_sum'],d[f'r{g}t']):>4} {_pct(d[f'b{g}c'],d[f'b{g}t']):>4} {_pct(d[f'm{g}c'],d[f'm{g}t']):>4}"
    line += f"  {dbt:>6}"
    print(line)


# ── Table 3: BT rank-i accuracy ───────────────────────────────────────────────

def print_rank_accuracy(rows: list[dict]) -> None:
    """Correctness rate at each BT rank position."""
    g1   = [r for r in rows if "rank1_correct" in r]
    g1g2 = [r for r in rows if "rank1_correct" in r and "rank2_correct" in r]
    if not g1:
        return

    n = len(g1[0]["rank0_correct"])
    W3 = 52 if not g1g2 else 64

    print(f"\n{'─'*W3}")
    print(f"表3: BT rank-i 正确率")
    print(f"  {'Rank':<6} {'gen0':>8}  {'gen1 (N=%d)'%len(g1):>12}" +
          (f"  {'gen2 (N=%d)'%len(g1g2):>12}" if g1g2 else ""))
    print(f"  {'─'*(W3-2)}")

    def _pct(rs, key, i): return sum(r[key][i] for r in rs) / len(rs)
    def _arrow(a, b): return "↑" if b > a + 0.001 else ("↓" if b < a - 0.001 else "→")

    for i in range(n):
        g0v = _pct(g1, "rank0_correct", i)
        g1v = _pct(g1, "rank1_correct", i)
        line = f"  #{i+1:<5} {g0v:>8.1%}  {g1v:>8.1%} {_arrow(g0v,g1v)}"
        if g1g2:
            g1b = _pct(g1g2, "rank1_correct", i)
            g2v = _pct(g1g2, "rank2_correct", i)
            line += f"  {g2v:>8.1%} {_arrow(g1b,g2v)}"
        print(line)


# ── Table 4: Changed questions ────────────────────────────────────────────────

def print_changes(rows: list[dict]) -> None:
    """Only questions where BT#1 outcome changed between gen0 and gen2 (or gen1)."""
    changed = []
    for r in rows:
        g0 = r.get("gen0_bt_correct")
        g1 = r.get("gen1_bt_correct")
        g2 = r.get("gen2_bt_correct")
        if g0 is None: continue
        final = g2 if g2 is not None else g1
        if final is None: continue
        if final != g0:
            changed.append(r)

    if not changed:
        print(f"\n{'─'*W}")
        print("表4: 结果变化明细 — 无变化")
        return

    rescued   = [r for r in changed if not r.get("gen0_bt_correct") and (r.get("gen2_bt_correct") or r.get("gen1_bt_correct"))]
    regressed = [r for r in changed if r.get("gen0_bt_correct") and not (r.get("gen2_bt_correct") if r.get("gen2_bt_correct") is not None else r.get("gen1_bt_correct"))]

    print(f"\n{'─'*W}")
    print(f"表4: 结果变化明细  ({len(rescued)} rescued, {len(regressed)} regressed)")
    print(f"  {'QID':10}  {'Cat':22}  {'g0':>4} {'g1':>4} {'g2':>4}  GT")
    print(f"  {'─'*(W-2)}")

    def _s(v): return "✓" if v else ("✗" if v is not None else "—")

    for r in sorted(changed, key=lambda r: (not r.get("gen0_bt_correct"), r["category"])):
        print(f"  {r['qid'][:10]:10}  {r['category'][:22]:22}  "
              f"{_s(r.get('gen0_bt_correct')):>4} "
              f"{_s(r.get('gen1_bt_correct')):>4} "
              f"{_s(r.get('gen2_bt_correct')):>4}  "
              f"{r['gt_short'][:28]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", action="store_true", help="v1 only (original 50q)")
    args = ap.parse_args()

    rows_v1 = load_rows_v1()
    rows_v2 = [] if args.v1 else load_rows_v2()
    rows    = rows_v1 + rows_v2

    if args.v1:
        label = f"v1 only  ({len(rows)} 题)"
    else:
        label = f"combined v1+v2  ({len(rows_v1)}+{len(rows_v2)}={len(rows)} 题)"

    g1 = sum(1 for r in rows if "c1" in r)
    g2 = sum(1 for r in rows if "c2" in r)
    print(f"{'─'*W}")
    print(f"HLE Evolution Analysis  —  {label}  |  gen1={g1}  gen2={g2}")

    print_summary(rows)
    print_category(rows)
    print_rank_accuracy(rows)
    print_changes(rows)


if __name__ == "__main__":
    main()
