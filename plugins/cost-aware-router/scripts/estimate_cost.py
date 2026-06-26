#!/usr/bin/env python3
"""Pre-call cost estimation utility.

Estimates the cost of an LLM call before making it, and suggests cheaper
alternatives if available.

Usage:
    python scripts/estimate_cost.py --model claude-opus-4 --input 1000 --output 500
    python scripts/estimate_cost.py --model sonnet --prompt "Write a haiku about clouds"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cost_router import cost_tracker


def estimate_from_prompt(prompt: str, model: str) -> tuple[int, int]:
    """Rough token estimate from prompt text (4 chars ≈ 1 token)."""
    input_tokens = len(prompt) // 4
    output_tokens = min(input_tokens * 2, 4096)
    return input_tokens, output_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate LLM call costs")
    parser.add_argument("--model", required=True, help="Model name or alias")
    parser.add_argument("--input", type=int, help="Input token count")
    parser.add_argument("--output", type=int, help="Output token count")
    parser.add_argument("--prompt", help="Prompt text (alternative to --input)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.prompt:
        input_tokens, output_tokens = estimate_from_prompt(args.prompt, args.model)
    elif args.input is not None and args.output is not None:
        input_tokens = args.input
        output_tokens = args.output
    else:
        parser.error("Either --prompt or both --input and --output are required")

    resolved = cost_tracker.resolve_model_name(args.model)
    cost = cost_tracker.calculate_cost(args.model, input_tokens, output_tokens)
    cheaper = cost_tracker.suggest_cheaper_alternative(args.model)

    result = {
        "model": resolved,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
    }

    if cheaper:
        cheaper_cost = cost_tracker.calculate_cost(cheaper, input_tokens, output_tokens)
        result["cheaper_alternative"] = {
            "model": cheaper,
            "cost_usd": round(cheaper_cost, 6),
            "savings_usd": round(cost - cheaper_cost, 6),
        }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Model: {resolved}")
        print(f"Input: {input_tokens:,} tokens")
        print(f"Output: {output_tokens:,} tokens")
        print(f"Estimated cost: ${cost:.6f}")
        if cheaper:
            print(f"\nCheaper alternative: {cheaper}")
            print(f"  Cost: ${cheaper_cost:.6f}")
            print(f"  Savings: ${cost - cheaper_cost:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
