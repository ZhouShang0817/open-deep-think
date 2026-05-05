"""
Pointwise judging experiment on 3 Hard problems × 20 gen0 solutions.
Each solution is judged 14 times (YES/NO). Rank by YES count descending.
Ties share the same accuracy = (# AC in tie group) / (tie group size).

Usage:
  python scripts/pointwise_experiment.py
  python scripts/pointwise_experiment.py --workers 20
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
MODEL        = "gemini-3.1-pro-preview"
TEMPERATURE  = 1.0
N_ROUNDS     = 14
POP_SIZE     = 20

EVOLVE_DIR   = Path("data/evolve")
CF_DIR       = Path("data/cf_problems")
PROB_DIR     = Path("data/problems")
OUT_DIR      = Path("data/pointwise_experiment")

PROMPT = """\
You are a competitive programming expert.

## Problem Statement
{problem}

## Solution
```cpp
{code}
```

Is this solution correct for all valid inputs within the constraints? \
Briefly explain your reasoning, then end your response with a line containing exactly: \
VERDICT: YES or VERDICT: NO\
"""

# ── Eligible Hard problems (gen0 has mix of AC/WA) ──────────────────────────
def _find_candidates() -> list[str]:
    import csv
    diff_map = {}
    with open("data/problem_difficulty.csv") as f:
        for row in csv.DictReader(f):
            diff_map[row["pid"]] = row["difficulty"]

    candidates = []
    for pid, diff in sorted(diff_map.items()):
        if diff != "Hard":
            continue
        jud_dir = EVOLVE_DIR / pid / "judge_results"
        if not jud_dir.exists():
            continue
        ac = total = 0
        for i in range(POP_SIZE):
            jp = jud_dir / f"gen0_sol{i:02d}.json"
            if not jp.exists():
                continue
            d = json.loads(jp.read_text())
            total += 1
            if bool(d.get("result", {}).get("passed", False)):
                ac += 1
        if total == POP_SIZE and 0 < ac < POP_SIZE:
            candidates.append(pid)
    return candidates


# ── API call ─────────────────────────────────────────────────────────────────
def _client():
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def _call_pointwise(prompt: str, max_attempts: int = 30) -> str:
    """Returns 'YES', 'NO', or 'UNKNOWN'."""
    client = _client()
    for attempt in range(max_attempts):
        try:
            config = types.GenerateContentConfig(
                temperature=TEMPERATURE,
                max_output_tokens=65536,
            )
            resp = client.models.generate_content(
                model=MODEL, contents=prompt, config=config,
            )
            # Search all parts (including thought) for verdict
            all_text = ""
            if resp and resp.candidates:
                for part in resp.candidates[0].content.parts:
                    if part.text:
                        all_text += part.text + "\n"
            if not all_text and resp and resp.text:
                all_text = resp.text
            m = re.search(r"VERDICT:\s*(YES|NO)", all_text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
            return "UNKNOWN"
        except Exception as e:
            msg = str(e)
            if "429" in msg or "503" in msg or "UNAVAILABLE" in msg:
                wait = min(10 * (2 ** attempt), 120)
                time.sleep(wait)
                continue
            raise
    return "UNKNOWN"


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pick 3 problems
    candidates = _find_candidates()
    rng = random.Random(args.seed)
    pids = rng.sample(candidates, 3)
    pids.sort()
    print(f"Selected problems: {pids}")

    # Load problems and solutions
    tasks = []  # (pid, sol_idx, problem_text, code, ground_truth)
    for pid in pids:
        # Load problem
        prob_path = CF_DIR / f"{pid}.md"
        if not prob_path.exists():
            prob_path = PROB_DIR / f"{pid}.md"
        problem_text = prob_path.read_text(encoding="utf-8")

        # Load solutions and ground truth
        sol_dir = EVOLVE_DIR / pid / "solutions"
        jud_dir = EVOLVE_DIR / pid / "judge_results"
        for i in range(POP_SIZE):
            sol_path = sol_dir / f"gen0_sol{i:02d}.md"
            jud_path = jud_dir / f"gen0_sol{i:02d}.json"
            md = sol_path.read_text(encoding="utf-8")
            # Extract code
            matches = re.findall(r"```cpp\s*\n(.*?)```", md, re.DOTALL)
            code = matches[-1].strip() if matches else ""
            # Ground truth
            jd = json.loads(jud_path.read_text())
            passed = bool(jd.get("result", {}).get("passed", False))
            tasks.append((pid, i, problem_text, code, passed))

    print(f"Total: {len(tasks)} solutions × {N_ROUNDS} rounds = {len(tasks) * N_ROUNDS} calls")

    # Check for existing results (resume support)
    results_file = OUT_DIR / "results.json"
    if results_file.exists():
        results = json.loads(results_file.read_text())
        print(f"Resuming: {sum(len(v) for v in results.values())} rounds already done")
    else:
        results = {}  # key: "pid/sol_idx" -> list of verdicts

    # Build work items
    work = []
    for pid, sol_idx, problem_text, code, passed in tasks:
        key = f"{pid}/{sol_idx}"
        existing = len(results.get(key, []))
        for round_i in range(existing, N_ROUNDS):
            prompt = PROMPT.format(problem=problem_text, code=code)
            work.append((key, prompt))

    print(f"Remaining API calls: {len(work)}")

    if not work:
        print("All rounds complete.")
    else:
        # Execute
        save_interval = 50
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for key, prompt in work:
                f = executor.submit(_call_pointwise, prompt)
                futures[f] = key

            with tqdm(total=len(work), desc="pointwise", unit="call") as bar:
                for f in as_completed(futures):
                    key = futures[f]
                    try:
                        verdict = f.result()
                    except Exception as e:
                        tqdm.write(f"  [!] {key}: {e}")
                        verdict = "UNKNOWN"

                    if key not in results:
                        results[key] = []
                    results[key].append(verdict)
                    bar.update(1)
                    completed += 1

                    if completed % save_interval == 0:
                        results_file.write_text(json.dumps(results, indent=2))

        # Final save
        results_file.write_text(json.dumps(results, indent=2))

    # ── Analysis ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Pointwise Experiment Results  (model={MODEL}, rounds={N_ROUNDS})")
    print(f"{'='*70}")

    # Build ground truth map
    gt = {}
    for pid, sol_idx, _, _, passed in tasks:
        gt[f"{pid}/{sol_idx}"] = passed

    for pid in pids:
        # Collect (sol_idx, yes_count, ground_truth)
        entries = []
        for i in range(POP_SIZE):
            key = f"{pid}/{i}"
            verdicts = results.get(key, [])
            yes_count = sum(1 for v in verdicts if v == "YES")
            entries.append((i, yes_count, gt[key]))

        # Sort by yes_count descending
        entries.sort(key=lambda x: -x[1])

        # Compute rank accuracy with tie handling
        n_ac = sum(1 for _, _, p in entries if p)
        print(f"\n  {pid}  (gen0: {n_ac}/20 AC)")
        print(f"  {'Rank':<6} {'Sol':>4} {'YES':>5}/{N_ROUNDS}  {'True':>5}  {'Rank ACC':>9}")
        print(f"  {'─'*40}")

        # Group by yes_count for tie handling
        rank_pos = 0
        i = 0
        rank_accuracies = []
        while i < len(entries):
            # Find tie group
            j = i
            while j < len(entries) and entries[j][1] == entries[i][1]:
                j += 1
            tie_group = entries[i:j]
            tie_ac = sum(1 for _, _, p in tie_group if p)
            tie_acc = tie_ac / len(tie_group)

            for sol_idx, yes_count, passed in tie_group:
                rank_pos += 1
                label = "AC" if passed else "WA"
                print(f"  #{rank_pos:<5} {sol_idx:>4} {yes_count:>5}/{N_ROUNDS}  {label:>5}  {tie_acc:>8.1%}")
                rank_accuracies.append(tie_acc)

            i = j

        # Summary
        print(f"  {'─'*40}")
        # Top-k accuracy
        for k in [1, 3, 5, 10]:
            top_k_acc = sum(rank_accuracies[:k]) / k
            print(f"  Top-{k} avg accuracy: {top_k_acc:.1%}")

    # ── Aggregate across all 3 problems ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Aggregate rank-i accuracy across {len(pids)} problems")
    print(f"{'='*70}")

    all_rank_accs = [[] for _ in range(POP_SIZE)]
    for pid in pids:
        entries = []
        for i in range(POP_SIZE):
            key = f"{pid}/{i}"
            verdicts = results.get(key, [])
            yes_count = sum(1 for v in verdicts if v == "YES")
            entries.append((i, yes_count, gt[f"{pid}/{i}"]))
        entries.sort(key=lambda x: -x[1])

        # Tie handling
        rank_pos = 0
        idx = 0
        while idx < len(entries):
            j = idx
            while j < len(entries) and entries[j][1] == entries[idx][1]:
                j += 1
            tie_ac = sum(1 for _, _, p in entries[idx:j] if p)
            tie_acc = tie_ac / (j - idx)
            for _ in range(idx, j):
                all_rank_accs[rank_pos].append(tie_acc)
                rank_pos += 1
            idx = j

    print(f"\n  {'Rank':<6} {'Accuracy':>10}")
    print(f"  {'─'*20}")
    for r in range(POP_SIZE):
        if all_rank_accs[r]:
            acc = sum(all_rank_accs[r]) / len(all_rank_accs[r])
            n = len(all_rank_accs[r])
            print(f"  #{r+1:<5} {acc:>8.1%}  (n={n})")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
