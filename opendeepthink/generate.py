"""
Generate N independent solutions for a problem in parallel.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from opendeepthink.llm_client import call_llm

DEFAULT_PROBLEMS_DIR = Path("data/cf_problems")

SYSTEM_PROMPT = """\
You are an expert competitive programmer.
Output your solution as a single ```cpp ... ``` block, preceded by brief reasoning.\
"""

USER_PROMPT = """\
{problem}
"""


def _generate_one(problem_text: str) -> str:
    return call_llm(
        USER_PROMPT.format(problem=problem_text),
        provider="gemini",
        system_prompt=SYSTEM_PROMPT,
        temperature=1.0,
    )


def generate(pid: str, n: int = 20, workers: int = 20,
             problems_dir: Path = DEFAULT_PROBLEMS_DIR,
             executor=None,
             sol_dir: Path = None, gen: int = None) -> list[str]:
    """Generate n solutions for problem pid. Returns list of raw LLM responses.

    If executor is provided, jobs are submitted to it (shared pool mode).
    If sol_dir and gen are provided, each solution is saved to disk immediately
    upon completion instead of waiting for all n to finish.
    """
    problem_text = (problems_dir / f"{pid}.md").read_text(encoding="utf-8")
    silent = executor is not None

    def _save_one(i: int, md: str):
        if sol_dir is not None and gen is not None and md:
            (sol_dir / f"gen{gen}_sol{i:02d}.md").write_text(md, encoding="utf-8")

    def _run(pool):
        results = [None] * n
        futures = {pool.submit(_generate_one, problem_text): i for i in range(n)}
        done = 0
        with tqdm(total=n, desc=f"generate {pid}", unit="sol", disable=silent) as bar:
            for fut in as_completed(futures):
                i = futures[fut]
                exc = fut.exception()
                md = "" if exc else fut.result()
                results[i] = md
                if exc:
                    print(f"  [!] {pid} generate sol{i} failed: {exc}", flush=True)
                else:
                    _save_one(i, md)
                done += 1
                if silent and done % 5 == 0:
                    print(f"  [{pid}] generate {done}/{n}", flush=True)
                bar.update(1)
        return results

    if executor is not None:
        return _run(executor)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return _run(pool)
