"""
Self-refine + Pointwise judge pipeline on Hard problems using Gemini 3.1 Pro.

Pipeline per problem:
  Round 0:    reuse gen0 solutions from data/evolve/<pid>/solutions/gen0_sol*.md
  Rounds 1-6: self-refine each solution independently (rolling window, single-step)
  Pointwise:  N_POINTWISE YES/NO judgments per round-N_REFINE_ROUNDS solution
  Pick:       max-YES tie group → tie group avg AC from online judge ground truth
  Submit:     all 20 round-6 solutions to online judge (fire-and-forget)

Output: data/refine_pointwise_31pro/<pid>/{solutions,pointwise,judge_results,pick.json}

Usage:
  python scripts/refine_pointwise_hard.py --pilot
  python scripts/refine_pointwise_hard.py
  python scripts/refine_pointwise_hard.py --pids 10081 1173
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Switch model BEFORE importing anything that uses llm_client ──
import opendeepthink.llm_client as llm_client
llm_client.DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"

import argparse
import csv
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from opendeepthink.llm_client import call_llm
from opendeepthink.submit import submit_all


POP_SIZE         = 20
N_REFINE_ROUNDS  = 6
N_POINTWISE      = 8

EVOLVE_DIR = Path("data/evolve")
CF_DIR     = Path("data/cf_problems")
PROB_DIR   = Path("data/problems")
OUT_BASE   = Path("data/refine_pointwise_31pro")
DIFF_CSV   = Path("data/problem_difficulty.csv")


REFINE_SYSTEM = """\
You are an expert competitive programmer.
Output your solution as a single ```cpp ... ``` block, preceded by brief reasoning.\
"""

REFINE_PROMPT = """\
## Problem
{problem}

## Current Solution
```cpp
{code}
```

Review the solution. Output it unchanged if you believe it is correct;
otherwise output an improved version.\
"""

POINTWISE_PROMPT = """\
You are a competitive programming expert.

## Problem Statement
{problem}

## Solution
```cpp
{code}
```

Is this solution correct for all valid inputs within the constraints? Briefly \
explain your reasoning, then end your response with a line containing exactly: \
VERDICT: YES or VERDICT: NO\
"""


# ── helpers ──────────────────────────────────────────────────────────────────

def _problems_dir_for(pid: str) -> Path:
    if (CF_DIR / f"{pid}.md").exists():
        return CF_DIR
    return PROB_DIR


def _extract_code(md: str) -> str | None:
    matches = re.findall(r'```cpp\s*\n(.*?)```', md, re.DOTALL)
    return matches[-1].strip() if matches else None


def _load_difficulty() -> dict[str, str]:
    out = {}
    with open(DIFF_CSV) as f:
        for row in csv.DictReader(f):
            out[row["pid"]] = row["difficulty"]
    return out


def _pilot_pids(seed: int = 42) -> list[str]:
    diff = _load_difficulty()
    candidates = []
    for pid, d in sorted(diff.items()):
        if d != "Hard":
            continue
        jud = EVOLVE_DIR / pid / "judge_results"
        if not jud.exists():
            continue
        ac = total = 0
        for i in range(POP_SIZE):
            jp = jud / f"gen0_sol{i:02d}.json"
            if not jp.exists():
                continue
            try:
                data = json.loads(jp.read_text())
            except Exception:
                continue
            total += 1
            if data.get("result", {}).get("passed", False):
                ac += 1
        if total == POP_SIZE and 0 < ac < POP_SIZE:
            candidates.append(pid)
    rng = random.Random(seed)
    pilot = rng.sample(candidates, 3)
    pilot.sort()
    return pilot


def _hard_pids() -> list[str]:
    diff = _load_difficulty()
    return sorted([p for p, d in diff.items() if d == "Hard"])


def _log(pid: str, msg: str):
    print(f"[{pid}] {msg}", flush=True)


# ── Stage 1: reuse gen0 ───────────────────────────────────────────────────────

def load_gen0_as_round0(pid: str, sol_dir: Path) -> list[str]:
    """Copy data/evolve/<pid>/solutions/gen0_sol*.md → round0_sol*.md and return list."""
    src = EVOLVE_DIR / pid / "solutions"
    sols = []
    for i in range(POP_SIZE):
        src_path = src / f"gen0_sol{i:02d}.md"
        if not src_path.exists():
            raise FileNotFoundError(f"Missing gen0 sol: {src_path}")
        md = src_path.read_text(encoding="utf-8")
        sols.append(md)
        dst = sol_dir / f"round0_sol{i:02d}.md"
        if not dst.exists():
            dst.write_text(md, encoding="utf-8")
    return sols


# ── Stage 2: Self-refine ──────────────────────────────────────────────────────

def refine_one(code: str, problem_text: str) -> str:
    prompt = REFINE_PROMPT.format(problem=problem_text, code=code)
    return call_llm(prompt, provider="gemini",
                    system_prompt=REFINE_SYSTEM, temperature=1.0)


def refine_round(pid: str, sols: list[str], round_idx: int, sol_dir: Path,
                 problem_text: str, executor: ThreadPoolExecutor) -> list[str]:
    """One round across 20 solutions. Round-major. Resume-aware."""
    target = [sol_dir / f"round{round_idx}_sol{i:02d}.md" for i in range(POP_SIZE)]
    if all(p.exists() for p in target):
        _log(pid, f"round{round_idx}: resuming — all 20 already on disk")
        return [p.read_text(encoding="utf-8") for p in target]

    out = list(sols)
    fut_map = {}
    for i in range(POP_SIZE):
        if target[i].exists():
            out[i] = target[i].read_text(encoding="utf-8")
            continue
        prev_code = _extract_code(sols[i])
        if not prev_code:
            target[i].write_text(sols[i], encoding="utf-8")
            out[i] = sols[i]
            continue
        f = executor.submit(refine_one, prev_code, problem_text)
        fut_map[f] = i

    total = len(fut_map)
    done = 0
    for f in as_completed(fut_map):
        i = fut_map[f]
        exc = f.exception()
        if exc:
            _log(pid, f"  [!] round{round_idx} sol{i:02d}: {exc} — keeping previous")
            md = sols[i]
        else:
            md = f.result() or sols[i]
        target[i].write_text(md, encoding="utf-8")
        out[i] = md
        done += 1
        if done % 5 == 0:
            _log(pid, f"  round{round_idx}: {done}/{total}")
    return out


# ── Stage 3: Pointwise judge ──────────────────────────────────────────────────

def pointwise_one(code: str, problem_text: str) -> str:
    """Returns 'YES', 'NO', or 'UNKNOWN'."""
    prompt = POINTWISE_PROMPT.format(problem=problem_text, code=code)
    try:
        text = call_llm(prompt, provider="gemini", temperature=1.0)
    except Exception:
        return "UNKNOWN"
    m = re.search(r"VERDICT:\s*(YES|NO)", text or "", re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def pointwise_stage(pid: str, sols: list[str], pw_dir: Path, problem_text: str,
                    executor: ThreadPoolExecutor) -> dict[int, list[str]]:
    """Run N_POINTWISE calls per solution. Resume-aware. Saves incrementally."""
    results: dict[int, list[str]] = {}
    for i in range(POP_SIZE):
        pf = pw_dir / f"sol{i:02d}.json"
        if pf.exists():
            try:
                data = json.loads(pf.read_text())
                results[i] = list(data.get("verdicts", []))
            except Exception:
                results[i] = []
        else:
            results[i] = []

    save_lock = threading.Lock()

    def _save(i: int):
        yes = sum(1 for v in results[i] if v == "YES")
        (pw_dir / f"sol{i:02d}.json").write_text(json.dumps({
            "verdicts": results[i],
            "yes_count": yes,
        }, indent=2))

    fut_map = {}
    for i in range(POP_SIZE):
        needed = N_POINTWISE - len(results[i])
        if needed <= 0:
            continue
        code = _extract_code(sols[i])
        if not code:
            results[i] = ["UNKNOWN"] * N_POINTWISE
            with save_lock:
                _save(i)
            continue
        for _ in range(needed):
            f = executor.submit(pointwise_one, code, problem_text)
            fut_map[f] = i

    if not fut_map:
        return results

    _log(pid, f"  pointwise: {len(fut_map)} calls submitted")
    done = 0
    for f in as_completed(fut_map):
        i = fut_map[f]
        try:
            verdict = f.result()
        except Exception:
            verdict = "UNKNOWN"
        with save_lock:
            results[i].append(verdict)
            _save(i)
        done += 1
        if done % 20 == 0:
            _log(pid, f"  pointwise: {done}/{len(fut_map)}")
    return results


# ── Stage 4: Compute pick + tie acc ──────────────────────────────────────────

def compute_pick(results: dict[int, list[str]],
                 judge_results: dict[int, bool | None]) -> dict:
    yes_counts = {i: sum(1 for v in results[i] if v == "YES") for i in range(POP_SIZE)}
    max_yes = max(yes_counts.values()) if yes_counts else 0
    tie = sorted([i for i, c in yes_counts.items() if c == max_yes])
    judged_in_tie = [i for i in tie if judge_results.get(i) is not None]
    tie_ac = sum(1 for i in tie if judge_results.get(i) is True)
    tie_acc = tie_ac / len(tie) if tie else 0.0
    return {
        "max_yes_count": max_yes,
        "tie_group": tie,
        "tie_size": len(tie),
        "tie_judged": len(judged_in_tie),
        "tie_ac": tie_ac,
        "tie_acc": tie_acc,
        "yes_counts": yes_counts,
        "judge_results": {str(k): v for k, v in judge_results.items()},
    }


# ── Per-problem coordinator ──────────────────────────────────────────────────

def process_problem(pid: str, executor: ThreadPoolExecutor):
    out_dir = OUT_BASE / pid
    sol_dir = out_dir / "solutions"
    pw_dir  = out_dir / "pointwise"
    jud_dir = out_dir / "judge_results"
    for d in (sol_dir, pw_dir, jud_dir):
        d.mkdir(parents=True, exist_ok=True)

    problem_text = (_problems_dir_for(pid) / f"{pid}.md").read_text(encoding="utf-8")

    t0 = time.monotonic()
    _log(pid, "starting")

    sols = load_gen0_as_round0(pid, sol_dir)
    _log(pid, "round0 (gen0): loaded 20 solutions")

    for r in range(1, N_REFINE_ROUNDS + 1):
        sols = refine_round(pid, sols, r, sol_dir, problem_text, executor)
        _log(pid, f"round{r}: complete")

    pw_results = pointwise_stage(pid, sols, pw_dir, problem_text, executor)
    _log(pid, "pointwise: complete")

    final_files = sorted(sol_dir.glob(f"round{N_REFINE_ROUNDS}_sol*.md"))
    _log(pid, f"submitting {len(final_files)} round{N_REFINE_ROUNDS} sols (sync)")
    submit_all(pid, final_files, jud_dir)
    _log(pid, f"online judge: complete")

    judge_results: dict[int, bool | None] = {}
    for i in range(POP_SIZE):
        jp = jud_dir / f"round{N_REFINE_ROUNDS}_sol{i:02d}.json"
        if jp.exists():
            try:
                data = json.loads(jp.read_text())
                judge_results[i] = bool(data.get("result", {}).get("passed", False))
            except Exception:
                judge_results[i] = None
        else:
            judge_results[i] = None

    pick = compute_pick(pw_results, judge_results)
    (out_dir / "pick.json").write_text(json.dumps(pick, indent=2))

    elapsed = time.monotonic() - t0
    _log(pid, f"done in {elapsed/60:.1f} min  "
              f"max_yes={pick['max_yes_count']}/{N_POINTWISE}  "
              f"tie_size={pick['tie_size']}  "
              f"tie_acc(judged_so_far)={pick['tie_acc']:.2%}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--pilot", action="store_true",
                    help="3 pilot pids (seed=42 sample of mixed-AC Hard)")
    ap.add_argument("--pids", nargs="+", help="Specific pids")
    args = ap.parse_args()

    if args.pids:
        pids = args.pids
    elif args.pilot:
        pids = _pilot_pids(seed=42)
    else:
        pids = _hard_pids()

    print(f"\n{'='*60}", flush=True)
    print(f"[refine_pointwise_hard] model = {llm_client.DEFAULT_GEMINI_MODEL}", flush=True)
    print(f"[refine_pointwise_hard] {len(pids)} problem(s): {pids}", flush=True)
    print(f"[refine_pointwise_hard] workers={args.workers}  "
          f"refine_rounds={N_REFINE_ROUNDS}  pointwise={N_POINTWISE}", flush=True)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        threads = []
        for pid in pids:
            t = threading.Thread(target=process_problem, args=(pid, executor),
                                 name=f"coord-{pid}", daemon=False)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    elapsed = time.monotonic() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"[refine_pointwise_hard] all done in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
