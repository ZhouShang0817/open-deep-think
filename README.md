# OpenDeepThink

## Overview

A population-based test-time compute framework that evolves LLM-generated solutions
via pairwise Bradley–Terry comparison. See the paper for details.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your Gemini API key in .env
```

## Quick Start

Both entry points look up the problem statement at
`<problems-dir>/<pid>.md`. Three sample problems (`10003`, `10007`, `10037`)
are included under `data/sample/`; pass `--problems-dir data/sample` to use
them, or place your own under `data/cf_problems/` (the default).

**Single problem** — runs Gen 0 → tournaments / BT / mutate → final ranking
for one PID:

```bash
python -m opendeepthink.run \
    --pid 10003 \
    --problems-dir data/sample \
    --workers 20 \
    --generations 3 \
    --seed 42
```

**Multiple problems sharing one worker pool** — slow stragglers in one problem
do not idle workers, since other problems' jobs fill in:

```bash
python scripts/shared_run.py \
    --pids 10003 10007 10037 \
    --problems-dir data/sample \
    --workers 20
```

Outputs are written to `data/evolve/<pid>/` (`solutions/`, `judge_results/`,
`rankings/`, `comparisons/`).

## Repository Structure

- `opendeepthink/` — Core algorithm: generation, pairwise judging, BT aggregation, mutation
- `hle/` — HLE benchmark experiments
- `scripts/` — Experiment launch scripts
- `data/` — Sample problems and metadata

## Data

- **CF-73 benchmark**: [anonymous link](https://anonymous.4open.science/r/CF-73-2CB1)
- Full experiment outputs available upon request.

## License

MIT
