"""Minimal repro: run TB088 (64-MZI crossbar) through the graphrag arm with
PHIDO_DEBUG_TOKENS on to see which LLM call overflows and whether compaction fires."""
import os
from pathlib import Path

os.environ["PHIDO_DEBUG_TOKENS"] = "1"

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import logging

_fh = logging.FileHandler(Path(__file__).resolve().parent / "results" / "_repro_tb088_debug.log", mode="w")
_fh.setFormatter(logging.Formatter("%(message)s"))
_fh.setLevel(logging.WARNING)
logging.getLogger("mcp_servers.llm_client").addHandler(_fh)
logging.getLogger("mcp_servers.llm_client").setLevel(logging.WARNING)

import e2_runner as R

PROMPT = ("Design a 8×8 optical switching network using a crossbar architecture with "
          "mzi_2x2_pn_diode devices. Arrange 64 of these 2×2 switches in a grid such that "
          "each of the 8 input waveguides intersects with each of the 8 output waveguides "
          "exactly once.")

cfg = R.RunConfig(model="o3-mini", enable_ar_gate=False,
                  enable_topology_gate=True, max_critic_rounds=2)
r = R.run_one(PROMPT, prompt_id="TB088", level=4, arm="graphrag", config=cfg)
print("RESULT stages:", r["stages"])
print("RESULT failed_at:", r.get("cost", {}).get("failed_at"))
print("RESULT error:", str(r.get("error"))[:300])
