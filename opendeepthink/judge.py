"""
Pairwise LLM judge: compare two solutions and return winner + feedback.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from opendeepthink.llm_client import call_llm

DEFAULT_PROBLEMS_DIR = Path("data/cf_problems")

PROMPT_PAIR = """\
You are a competitive programming expert.

## Problem Statement
{problem}

## Solution A
```cpp
{code_a}
```

## Solution B
```cpp
{code_b}
```

Which solution is more likely to receive an Accepted verdict from an online judge — \
meaning it produces correct output within the time and memory limits for all valid inputs?

If both solutions appear incorrect (wrong answer, TLE, or other issues), choose the one \
that requires fewer modifications to become Accepted.

If they are fundamentally identical or equally likely to be Accepted, output TIE.

Respond with a JSON object and nothing else, in exactly this format:
{{
  "feedback_a": "one sentence on Solution A's key strength or critical flaw",
  "feedback_b": "one sentence on Solution B's key strength or critical flaw",
  "winner": "A or B or TIE"
}}
"""


def _extract_code(md: str) -> str:
    matches = re.findall(r'```cpp\s*\n(.*?)```', md, re.DOTALL)
    return matches[-1].strip() if matches else md.strip()


def _parse_json(text: str) -> dict:
    clean = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', text.strip(), flags=re.MULTILINE)

    def try_loads(s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', s)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    result = try_loads(clean)
    if result is not None:
        return result
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        return try_loads(m.group()) or {}
    return {}


def _judge_pair(problem_text: str, code_a: str, code_b: str) -> dict:
    prompt = PROMPT_PAIR.format(
        problem=problem_text.strip(),
        code_a=code_a,
        code_b=code_b,
    )
    response = call_llm(prompt, provider="gemini", temperature=0.0)
    parsed = _parse_json(response)
    winner = parsed.get("winner", "TIE").strip().upper()
    if winner not in ("A", "B", "TIE"):
        winner = "TIE"
    return {
        "winner":     winner,
        "feedback_a": parsed.get("feedback_a", ""),
        "feedback_b": parsed.get("feedback_b", ""),
    }


def run_tournament(
    pid: str,
    solutions: list[str],
    n_rounds: int = 4,
    workers: int = 20,
    rng: random.Random = None,
    out_file: Path = None,
    label: str = "",
    problems_dir: Path = DEFAULT_PROBLEMS_DIR,
    executor=None,
) -> list[dict]:
    """
    Run n_rounds of random pairwise comparisons.
    Each round: randomly pair all solutions (n/2 pairs).
    Comparisons are appended to out_file (jsonl) in real time.
    Returns list of comparison records.
    """
    if rng is None:
        rng = random.Random()

    problem_text = (problems_dir / f"{pid}.md").read_text(encoding="utf-8")
    codes = [_extract_code(s) for s in solutions]
    n = len(solutions)

    jobs = []
    for rnd in range(n_rounds):
        indices = list(range(n))
        rng.shuffle(indices)
        for k in range(0, n - 1, 2):
            ia, ib = indices[k], indices[k + 1]
            jobs.append((rnd, ia, ib))

    results = []
    lock = __import__("threading").Lock()

    def _write(record):
        if out_file:
            with lock:
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

    desc = f"judge {label}" if label else "judge"

    def _run(pool, silent=False):
        futures = {
            pool.submit(_judge_pair, problem_text, codes[ia], codes[ib]): (rnd, ia, ib)
            for rnd, ia, ib in jobs
        }
        total = len(futures)
        done = 0
        with tqdm(total=total, desc=desc, unit="pair", disable=silent) as bar:
            for fut in as_completed(futures):
                rnd, ia, ib = futures[fut]
                exc = fut.exception()
                if exc:
                    record = {"round": rnd, "idx_a": ia, "idx_b": ib,
                              "winner": "TIE", "feedback_a": "", "feedback_b": "",
                              "error": str(exc)}
                else:
                    r = fut.result()
                    record = {"round": rnd, "idx_a": ia, "idx_b": ib, **r}
                _write(record)
                results.append(record)
                done += 1
                if silent and done % 10 == 0:
                    print(f"  [{pid}] {desc} {done}/{total}", flush=True)
                bar.update(1)
        return results

    if executor is not None:
        return _run(executor, silent=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return _run(pool, silent=False)
