"""
Pairwise LLM judge for HLE: compare two reasoning chains and return winner + feedback.
"""

import json
import re
import random
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opendeepthink.llm_client import call_llm
from tqdm import tqdm

PROMPT_PAIR = """\
You are an expert evaluator for extremely difficult exam questions spanning mathematics, \
science, humanities, and other advanced topics.

## Question
{question}

## Solution A
{solution_a}

## Solution B
{solution_b}

Carefully read both solutions, paying attention to the reasoning steps and the final answer.

Which solution is more likely to be correct?

Consider:
- Is the reasoning logically sound and free of errors?
- Is the final answer consistent with the reasoning?
- Are there any calculation mistakes, flawed assumptions, or gaps in logic?

If both solutions appear incorrect, choose the one that is closer to being right \
(i.e., requires fewer corrections).

If both solutions are essentially equivalent in quality, output TIE.

Respond with a JSON object and nothing else, in exactly this format:
{{
  "feedback_a": "one sentence identifying the key strength or critical flaw in Solution A's reasoning",
  "feedback_b": "one sentence identifying the key strength or critical flaw in Solution B's reasoning",
  "winner": "A or B or TIE"
}}
"""


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


def _judge_pair(question: str, sol_a: str, sol_b: str) -> dict:
    prompt = PROMPT_PAIR.format(
        question=question.strip(),
        solution_a=sol_a.strip(),
        solution_b=sol_b.strip(),
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
    question: str,
    solutions: list[str],
    n_rounds: int = 4,
    workers: int = 20,
    rng: random.Random = None,
    out_file: Path = None,
    label: str = "",
) -> list[dict]:
    """
    Run n_rounds of random pairwise comparisons.
    Each round: randomly pair all solutions (n/2 pairs).
    Comparisons are appended to out_file (jsonl) in real time.
    Returns list of comparison records.
    """
    if rng is None:
        rng = random.Random()

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
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(jobs)
    done  = [0]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_judge_pair, question, solutions[ia], solutions[ib]): (rnd, ia, ib)
            for rnd, ia, ib in jobs
        }
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
            done[0] += 1
            if done[0] % 6 == 0 or done[0] == total:
                lbl = f"judge {label}" if label else "judge"
                print(f"  {lbl}: {done[0]}/{total} pairs done", flush=True)

    return results
