"""Smoke test: confirm the visual critic now runs (routed to a vision model)
when the main pipeline model is text-only (o3-mini), instead of 400-ing.

Runs 2 multi-component prompts through explore→finalize and prints every
`visual_critic` phase detail plus the critic's analysis text. A pass looks like
"routing visual critic to gpt-4o" followed by a real analysis / VERDICT — NOT
"image_url is only supported by certain models".
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import e2_runner as R  # noqa: E402

MODEL = os.getenv("E2_MODEL", "o3-mini")
PROMPTS = [
    "A 1x2 MMI splitter feeding two C-band pin photodiodes.",
    "A Mach-Zehnder interferometer built from two 2x2 MMIs with a thermo-optic "
    "phase shifter on one arm.",
]


def main():
    orch = R.RealOrchestrator()
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n{'=' * 70}\n[{i}] {prompt}\n{'=' * 70}", flush=True)
        explore_state = None
        for ev in orch.explore(prompt, MODEL):
            if ev.get("type") == "clarification":
                explore_state = ev
        if explore_state is None:
            print("  explore produced no state; skipping", flush=True)
            continue
        for ev in orch.finalize(
            explore_state, prompt, MODEL,
            max_critic_rounds=2, enable_topology_gate=True, enable_ar_gate=False,
        ):
            if ev.get("phase") == "visual_critic":
                print(f"  [phase] {ev.get('detail')}", flush=True)
            if ev.get("type") == "agent_text" and "critic" in (ev.get("content", "").lower()):
                print(f"  [critic]\n{ev['content'][:800]}", flush=True)


if __name__ == "__main__":
    main()
