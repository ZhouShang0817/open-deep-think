"""
LLM client supporting Gemini and OpenAI APIs.
Automatically logs token usage and cost to a JSONL file.
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RETRY_ATTEMPTS = 40
RETRY_BASE_DELAY = 5   # seconds, doubles each attempt, capped at 90s
RETRY_MAX_DELAY  = 90

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── Pricing (USD per 1M tokens) ──────────────────────────────────────────────
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"

PRICING = {
    "gemini-2.5-pro": {
        "input_per_m": 1.25,
        "output_per_m": 10.00,
        "input_per_m_long": 2.50,
        "output_per_m_long": 15.00,
        "long_context_threshold": 200_000,
    },
    "gemini-3-flash-preview": {
        "input_per_m": 0.15,
        "output_per_m": 0.60,
        "input_per_m_long": 0.30,
        "output_per_m_long": 1.20,
        "long_context_threshold": 200_000,
    },
    "gemini-3.1-pro-preview": {
        # ≤200K context window
        "input_per_m": 2.00,
        "output_per_m": 12.00,
        # >200K context window
        "input_per_m_long": 4.00,
        "output_per_m_long": 18.00,
        "long_context_threshold": 200_000,
    },
    "gpt-5.4": {
        "input_per_m": 2.50,
        "output_per_m": 15.00,
        # >272K context window
        "input_per_m_long": 5.00,
        "output_per_m_long": 15.00,   # output price unchanged at >272K
        "long_context_threshold": 272_000,
    },
}

DEFAULT_LOG_FILE = str(Path(__file__).parent / "logs" / "usage_log.jsonl")


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    p = PRICING.get(model, {})
    if not p:
        return {"input_cost_usd": None, "output_cost_usd": None, "total_cost_usd": None}

    threshold = p.get("long_context_threshold", 0)
    if input_tokens > threshold:
        in_rate = p["input_per_m_long"]
        out_rate = p["output_per_m_long"]
    else:
        in_rate = p["input_per_m"]
        out_rate = p["output_per_m"]

    input_cost = input_tokens / 1_000_000 * in_rate
    output_cost = output_tokens / 1_000_000 * out_rate
    return {
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(input_cost + output_cost, 8),
    }


def _log(record: dict, log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Gemini ────────────────────────────────────────────────────────────────────

def call_gemini(
    prompt: str,
    *,
    model: str = None,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_output_tokens: Optional[int] = None,
    log_file: str = DEFAULT_LOG_FILE,
) -> str:
    from google import genai
    from google.genai import types

    model = model or DEFAULT_GEMINI_MODEL

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=1_800_000),  # 30 min
    )

    config_kwargs: dict = {"temperature": temperature}
    if max_output_tokens:
        config_kwargs["max_output_tokens"] = max_output_tokens
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt

    config = types.GenerateContentConfig(**config_kwargs)

    for attempt in range(RETRY_ATTEMPTS):
        try:
            t0 = time.monotonic()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            latency = round(time.monotonic() - t0, 3)
            break
        except Exception as e:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            base  = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            delay = base * random.uniform(0.8, 1.2)
            if attempt >= 1:  # only print on 2nd+ retry to reduce noise
                print(f"[retry {attempt+1}/{RETRY_ATTEMPTS-1}] {type(e).__name__}: {e} — sleeping {delay:.1f}s", flush=True)
            time.sleep(delay)

    input_tokens = response.usage_metadata.prompt_token_count or 0
    output_tokens = response.usage_metadata.candidates_token_count or 0
    cost = _compute_cost(model, input_tokens, output_tokens)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "gemini",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        **cost,
        "latency_s": latency,
        "prompt_preview": prompt[:200],
    }
    _log(record, log_file)

    return response.text or ""


# ── OpenAI ────────────────────────────────────────────────────────────────────

def call_openai(
    prompt: str,
    *,
    model: str = "gpt-5.4",
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    log_file: str = DEFAULT_LOG_FILE,
) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    t0 = time.monotonic()
    response = client.chat.completions.create(**kwargs)
    latency = round(time.monotonic() - t0, 3)

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = _compute_cost(model, input_tokens, output_tokens)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        **cost,
        "latency_s": latency,
        "prompt_preview": prompt[:200],
    }
    _log(record, log_file)

    return response.choices[0].message.content


# ── Unified entry point ───────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    *,
    provider: str = "gemini",          # "gemini" or "openai"
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    log_file: str = DEFAULT_LOG_FILE,
) -> str:
    """
    Unified LLM call. Provider defaults to gemini.

    Args:
        prompt:        User message.
        provider:      "gemini" or "openai".
        model:         Override default model for that provider.
        system_prompt: Optional system-level instruction.
        temperature:   Sampling temperature.
        max_tokens:    Max output tokens.
        log_file:      Path to the JSONL usage log.

    Returns:
        Model reply as a string.
    """
    if provider == "gemini":
        return call_gemini(
            prompt,
            model=model or DEFAULT_GEMINI_MODEL,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            log_file=log_file,
        )
    elif provider == "openai":
        return call_openai(
            prompt,
            model=model or "gpt-5.4",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            log_file=log_file,
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'gemini' or 'openai'.")


# ── Usage summary helper ──────────────────────────────────────────────────────

def usage_summary(log_file: str = DEFAULT_LOG_FILE) -> dict:
    """Read the JSONL log and print a cost/token summary by model."""
    totals: dict = {}
    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = r["model"]
                t = totals.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0})
                t["calls"] += 1
                t["input_tokens"] += r.get("input_tokens", 0)
                t["output_tokens"] += r.get("output_tokens", 0)
                t["total_cost_usd"] += r.get("total_cost_usd") or 0.0
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return {}

    for model, t in totals.items():
        t["total_cost_usd"] = round(t["total_cost_usd"], 6)
        print(f"{model}: {t['calls']} calls | "
              f"in={t['input_tokens']:,} out={t['output_tokens']:,} tokens | "
              f"${t['total_cost_usd']:.6f}")
    return totals


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick LLM call test")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    parser.add_argument("--summary", action="store_true", help="Print usage summary instead of calling API")
    args = parser.parse_args()

    if args.summary:
        usage_summary()
    else:
        reply = call_llm(args.prompt, provider=args.provider)
        print(reply)
