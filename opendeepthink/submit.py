"""
Submit solutions to the judge and poll for results.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL      = "https://yanagiorigami.uk"
POLL_INTERVAL = 15
SUBMIT_DELAY  = 0.3
SUBMIT_RETRIES      = 3
SUBMIT_RETRY_DELAYS = [5, 15, 30]


def _extract_code(md: str) -> str:
    matches = re.findall(r'```cpp\s*\n(.*?)```', md, re.DOTALL)
    return matches[-1].strip() if matches else None


def _submit(pid: str, code: str) -> int:
    r = requests.post(
        f"{BASE_URL}/submit",
        data={"pid": pid, "lang": "cpp"},
        files={"code": ("main.cpp", code.encode(), "text/plain")},
        timeout=30,
    )
    r.raise_for_status()
    return int(r.json()["sid"])


def _fetch(sid: int) -> dict:
    r = requests.get(f"{BASE_URL}/result/{sid}", timeout=20)
    if r.status_code == 404:
        return {"status": "pending"}
    r.raise_for_status()
    return r.json()


def submit_all(
    pid: str,
    solution_files: list[Path],
    out_dir: Path,
    workers: int = 20,
) -> list[dict]:
    """
    Submit all solution .md files to the judge and poll until done.
    Saves each result to out_dir/{stem}.json.
    Returns list of result dicts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect files that still need submission
    entries = []
    skip = 0
    for md_path in solution_files:
        result_path = out_dir / f"{md_path.stem}.json"
        if result_path.exists():
            skip += 1
            continue
        code = _extract_code(md_path.read_text(encoding="utf-8"))
        if not code:
            tqdm.write(f"  [!] {md_path.stem} — no cpp block, skipped")
            continue
        entries.append({"stem": md_path.stem, "md_path": md_path, "code": code})

    print(f"[submit] {len(solution_files)} total, {skip} already done, {len(entries)} to submit", flush=True)

    # Submit in parallel with delay
    sid_map = {}
    with tqdm(total=len(entries), desc="submit", unit="sol") as bar:
        for e in entries:
            for attempt in range(SUBMIT_RETRIES):
                try:
                    sid = _submit(pid, e["code"])
                    sid_map[sid] = e
                    break
                except Exception as ex:
                    if attempt == SUBMIT_RETRIES - 1:
                        tqdm.write(f"  [!] {e['stem']} submit failed: {ex}")
                    else:
                        time.sleep(SUBMIT_RETRY_DELAYS[attempt])
            time.sleep(SUBMIT_DELAY)
            bar.update(1)

    # Poll until all done
    pending = dict(sid_map)
    all_results = []

    with tqdm(total=len(pending), desc="poll", unit="sol") as bar:
        while pending:
            for sid in list(pending):
                try:
                    result = _fetch(sid)
                except Exception as ex:
                    tqdm.write(f"  [!] sid={sid} fetch error: {ex}")
                    continue

                if result.get("status") in ("pending", "queued"):
                    continue

                e = pending.pop(sid)
                record = {"stem": e["stem"], "sid": sid, "result": result}
                all_results.append(record)

                out_path = out_dir / f"{e['stem']}.json"
                out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

                verdict = result.get("result") or result.get("status", "?")
                n_ok = sum(1 for c in result.get("cases", []) if c.get("ok"))
                total = len(result.get("cases", []))
                tqdm.write(f"  [✓] {e['stem']}  sid={sid}  {verdict}  {n_ok}/{total}")
                bar.update(1)

            if pending:
                time.sleep(POLL_INTERVAL)

    return all_results


def submit_fire_and_forget(pid: str, solution_files: list[Path], out_dir: Path) -> None:
    """
    Submit solutions in a background daemon thread — returns immediately.
    Results are saved to out_dir/{stem}.json when they arrive.
    No polling progress is shown; errors are silently logged to out_dir/submit_errors.log.
    """
    def _run():
        out_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for md_path in solution_files:
            result_path = out_dir / f"{md_path.stem}.json"
            if result_path.exists():
                continue
            code = _extract_code(md_path.read_text(encoding="utf-8"))
            if code:
                entries.append({"stem": md_path.stem, "code": code})

        sid_map = {}
        for e in entries:
            for attempt in range(SUBMIT_RETRIES):
                try:
                    sid = _submit(pid, e["code"])
                    sid_map[sid] = e
                    break
                except Exception as ex:
                    if attempt == SUBMIT_RETRIES - 1:
                        _log_error(out_dir, f"{e['stem']} submit failed: {ex}")
                    else:
                        time.sleep(SUBMIT_RETRY_DELAYS[attempt])
            time.sleep(SUBMIT_DELAY)

        pending = dict(sid_map)
        while pending:
            for sid in list(pending):
                try:
                    result = _fetch(sid)
                except Exception as ex:
                    _log_error(out_dir, f"sid={sid} fetch error: {ex}")
                    continue
                if result.get("status") in ("pending", "queued"):
                    continue
                e = pending.pop(sid)
                record = {"stem": e["stem"], "sid": sid, "result": result}
                (out_dir / f"{e['stem']}.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
            if pending:
                time.sleep(POLL_INTERVAL)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _log_error(out_dir: Path, msg: str) -> None:
    with open(out_dir / "submit_errors.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")
