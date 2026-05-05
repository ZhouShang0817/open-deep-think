"""
Mutate (refine) HLE solutions using pairwise BT feedback.
"""

import re
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opendeepthink.llm_client import call_llm
from tqdm import tqdm

SYSTEM_PROMPT = """\
You are an expert at solving extremely difficult exam questions.
Think carefully step by step, then end your response with your final answer on its own line \
in the format: ANSWER: <your answer>\
"""

PROMPT_WITH_FB = """\
## Question
{question}

## Your Previous Solution
{solution}

## Pairwise Feedback
This solution was compared against other candidate solutions multiple times. \
Here is the feedback from those comparisons:

{feedback_sections}
## Task
Write an improved solution to this question.
- If the feedback reveals an error in your reasoning or final answer, identify and correct it
- If the feedback points to a stronger approach used by another solution, consider adopting it
- If your solution was judged correct but the reasoning can be made cleaner, refine it
- You may take a completely different approach if the current one appears fundamentally flawed

Think carefully step by step, then end with your final answer on its own line:
ANSWER: <your answer>\
"""

PROMPT_NO_FB = """\
## Question
{question}

## Your Previous Solution
{solution}

## Task
Write an improved solution to this question.
- Double-check every reasoning step and calculation for errors
- If the current approach seems flawed, consider a different method

Think carefully step by step, then end with your final answer on its own line:
ANSWER: <your answer>\
"""


def _build_feedback_sections(wins: list[str], ties: list[str], losses: list[str]) -> str:
    sections = ""
    if wins:
        items = "\n".join(f"- {fb}" for fb in wins)
        sections += f"### Wins (this solution was judged better than the other):\n{items}\n\n"
    if ties:
        items = "\n".join(f"- {fb}" for fb in ties)
        sections += f"### Ties (judged roughly equivalent to the other):\n{items}\n\n"
    if losses:
        items = "\n".join(f"- {fb}" for fb in losses)
        sections += f"### Losses (this solution was judged worse than the other):\n{items}\n\n"
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


def _mutate_one(question: str, solution: str, comparisons: list[dict], sol_idx: int) -> str:
    wins, ties, losses = _collect_feedback(sol_idx, comparisons)

    if wins or ties or losses:
        fb_sections = _build_feedback_sections(wins, ties, losses)
        prompt = PROMPT_WITH_FB.format(
            question=question.strip(),
            solution=solution.strip(),
            feedback_sections=fb_sections,
        )
    else:
        prompt = PROMPT_NO_FB.format(
            question=question.strip(),
            solution=solution.strip(),
        )

    return call_llm(prompt, provider="gemini", system_prompt=SYSTEM_PROMPT, temperature=1.0)


def mutate_batch(
    question: str,
    solutions: list[str],
    comparisons: list[dict],
    indices: list[int],
    workers: int = 20,
) -> dict[int, str]:
    """
    Mutate solutions at given indices. Returns {sol_idx: new_response}.
    Elite solutions (not in indices) keep their originals.
    """
    results = {}
    total   = len(indices)
    done    = [0]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_mutate_one, question, solutions[i], comparisons, i): i
            for i in indices
        }
        for fut in as_completed(futures):
            i   = futures[fut]
            exc = fut.exception()
            if exc:
                print(f"  [!] mutate sol{i}: {exc}", flush=True)
                results[i] = solutions[i]
            else:
                results[i] = fut.result()
            done[0] += 1
            if done[0] % 3 == 0 or done[0] == total:
                print(f"  mutate: {done[0]}/{total} done", flush=True)
    return results
