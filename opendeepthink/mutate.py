"""
Mutate (refine) solutions using their pairwise feedback from the current generation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from opendeepthink.llm_client import call_llm

DEFAULT_PROBLEMS_DIR = Path("data/cf_problems")

SYSTEM_PROMPT = """\
You are an expert competitive programmer.
Output your solution as a single ```cpp ... ``` block, preceded by brief reasoning.\
"""

PROMPT_WITH_FB = """\
## Problem
{problem}

## Solution
```cpp
{code}
```

## Pairwise Feedback
This solution was compared against other solutions multiple times:

{feedback_sections}
## Task
Write a solution that maximizes the probability of Accepted.
You may refine the existing solution or take a different approach if the current one is fundamentally flawed.

Think briefly, then output your final solution as a single ```cpp ... ``` block.\
"""

PROMPT_NO_FB = """\
## Problem
{problem}

## Solution
```cpp
{code}
```

## Task
Write a solution that maximizes the probability of Accepted.
You may refine the existing solution or take a different approach if the current one is fundamentally flawed.

Think briefly, then output your final solution as a single ```cpp ... ``` block.\
"""


def _extract_code(md: str) -> str:
    matches = re.findall(r'```cpp\s*\n(.*?)```', md, re.DOTALL)
    return matches[-1].strip() if matches else md.strip()


def _build_feedback_sections(wins: list[str], ties: list[str], losses: list[str]) -> str:
    sections = ""
    if wins:
        items = "\n".join(f"- {fb}" for fb in wins)
        sections += f"### Wins (this solution was judged better):\n{items}\n\n"
    if ties:
        items = "\n".join(f"- {fb}" for fb in ties)
        sections += f"### Ties (judged equally likely to be Accepted):\n{items}\n\n"
    if losses:
        items = "\n".join(f"- {fb}" for fb in losses)
        sections += f"### Losses (this solution was judged worse):\n{items}\n\n"
    return sections


def _collect_feedback(sol_idx: int, comparisons: list[dict]) -> tuple[list, list, list]:
    wins, ties, losses = [], [], []
    for c in comparisons:
        ia, ib, winner = c["idx_a"], c["idx_b"], c["winner"]
        fb_a = c.get("feedback_a", "")
        fb_b = c.get("feedback_b", "")

        if ia == sol_idx:
            my_fb = re.sub(r'\bSolution A\b', 'this solution', fb_a)
            my_fb = re.sub(r'\bSolution B\b', 'the other solution', my_fb)
            if winner == "A":   wins.append(my_fb)
            elif winner == "B": losses.append(my_fb)
            else:               ties.append(my_fb)
        elif ib == sol_idx:
            my_fb = re.sub(r'\bSolution B\b', 'this solution', fb_b)
            my_fb = re.sub(r'\bSolution A\b', 'the other solution', my_fb)
            if winner == "B":   wins.append(my_fb)
            elif winner == "A": losses.append(my_fb)
            else:               ties.append(my_fb)

    return wins, ties, losses


def _mutate_one(pid: str, solution_md: str, comparisons: list[dict], sol_idx: int,
                problems_dir: Path = DEFAULT_PROBLEMS_DIR) -> str:
    problem_text = (problems_dir / f"{pid}.md").read_text(encoding="utf-8")
    code = _extract_code(solution_md)
    wins, ties, losses = _collect_feedback(sol_idx, comparisons)

    if wins or ties or losses:
        fb_sections = _build_feedback_sections(wins, ties, losses)
        prompt = PROMPT_WITH_FB.format(
            problem=problem_text.strip(),
            code=code,
            feedback_sections=fb_sections,
        )
    else:
        prompt = PROMPT_NO_FB.format(
            problem=problem_text.strip(),
            code=code,
        )

    return call_llm(prompt, provider="gemini", system_prompt=SYSTEM_PROMPT, temperature=1.0)


def mutate_batch(
    pid: str,
    solutions: list[str],
    comparisons: list[dict],
    indices: list[int],
    workers: int = 20,
    problems_dir: Path = DEFAULT_PROBLEMS_DIR,
    executor=None,
    sol_dir: Path = None, gen: int = None,
) -> dict[int, str]:
    """
    Mutate solutions at given indices. Returns {sol_idx: new_md}.
    Elite solutions (not in indices) keep their originals.

    If executor is provided, jobs are submitted to it (shared pool mode).
    If sol_dir and gen are provided, each mutation is saved to disk immediately.
    """
    silent = executor is not None

    def _save_one(i: int, md: str):
        if sol_dir is not None and gen is not None and md:
            (sol_dir / f"gen{gen}_sol{i:02d}.md").write_text(md, encoding="utf-8")

    def _run(pool):
        results = {}
        futures = {
            pool.submit(_mutate_one, pid, solutions[i], comparisons, i, problems_dir): i
            for i in indices
        }
        total = len(futures)
        done = 0
        with tqdm(total=total, desc=f"mutate {pid}", unit="sol", disable=silent) as bar:
            for fut in as_completed(futures):
                i = futures[fut]
                exc = fut.exception()
                if exc:
                    print(f"  [!] {pid} mutate sol{i}: {exc}", flush=True)
                    results[i] = solutions[i]
                else:
                    results[i] = fut.result()
                    _save_one(i, results[i])
                done += 1
                if silent and done % 5 == 0:
                    print(f"  [{pid}] mutate {done}/{total}", flush=True)
                bar.update(1)
        return results

    if executor is not None:
        return _run(executor)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return _run(pool)
