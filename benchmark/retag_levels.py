"""Re-tag E2 prompt complexity levels to the PhIDO paper's Table 1 definition (component count).

The repo's `level` fields are heuristic/unreviewed and disagree with the paper: single composite
prompts (an MZI = 1 component) were tagged L3, and no true L4 exists in the gold set
([[e2-prompt-level-mislabeling-2026-07-09]]). Table 1 defines levels purely by COMPONENT COUNT,
counting a named composite block (MZI, ring, WDM, modulator, mesh, phased array) as ONE:
    L1 = 1 | L2 = 2 (single connecting edge) | L3 = 3-15 | L4 = 16-112

Two backends:
  --gold       count = number of gold topology nodes (composite = 1 node) -> DETERMINISTIC, exact
               for b3_gold / e2_prompts (which share ids). No LLM.
  --testbench  count from prompt text via one LLM call/prompt (e2_testbench_prompts has no gold).
               Approximate but paper-aligned; flag as LLM-derived.

Writes SIDECAR files (originals untouched): results/level_retag_{gold,testbench}.{json,md}. The JSON
adds `level_paper` + `n_comp` alongside the original `level` (kept as `level_repo`).
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Single source of truth for the Table 1 boundary lives in b3_eval (used by the scorer too).
from b3_eval import paper_level  # noqa: E402

COUNT_SYS = (
    "You count photonic components in a circuit description, matching the PhIDO paper's Table 1 "
    "convention. Rules: (1) Count each instantiated component/block. (2) A NAMED COMPOSITE block "
    "counts as ONE — an MZI, ring resonator, wavelength (de)multiplexer, modulator, optical phased "
    "array, Clements/Reck mesh, Spanke switch — do NOT count its internal parts (couplers, phase "
    "shifters). (3) Multiplicity counts each instance: 'four cascaded MZIs' = 4; a '1x16 tree of 1x2 "
    "MMIs' = 15 MMIs; '64 outputs each with a VOA and phase shifter' = 64 + 64. (4) Passive elements "
    "(waveguide, bend, directional coupler, grating/edge coupler, standalone MMI) each count as one "
    "when instantiated. Output ONLY a single integer — no words, no reasoning."
)


def _write(stem: str, rows: list[dict], header_cols: list[str], backend: str):
    """rows: list of dicts with id, prompt?, level_repo, n_comp, level_paper. Write json + md."""
    (HERE / f"results/{stem}.json").write_text(json.dumps(rows, indent=1))
    n_mis = sum(r["level_paper"] is not None and r["level_paper"] != r["level_repo"] for r in rows)
    dist_repo = Counter(r["level_repo"] for r in rows)
    dist_paper = Counter(r["level_paper"] for r in rows if r["level_paper"] is not None)
    lines = [f"# E2 level re-tag ({backend}) — paper Table 1 (component count)", "",
             f"L1=1 | L2=2 | L3=3-15 | L4=16-112. Mismatches vs repo tags: **{n_mis}/{len(rows)}**.", "",
             f"- repo dist:  {dict(sorted(dist_repo.items()))}",
             f"- paper dist: {dict(sorted(dist_paper.items()))}", "",
             "| " + " | ".join(header_cols) + " |",
             "|" + "|".join("---" for _ in header_cols) + "|"]
    for r in rows:
        mark = "" if r["level_paper"] is None else ("ok" if r["level_paper"] == r["level_repo"] else "**XX**")
        cells = [r["id"], f"L{r['level_repo']}", str(r["n_comp"]),
                 f"L{r['level_paper']}" if r["level_paper"] else "?", mark]
        if "prompt" in header_cols[-1].lower() or "prompt" in r:
            cells.append((r.get("prompt", "") or "")[:90].replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    (HERE / f"results/{stem}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote results/{stem}.json + .md  (mismatches {n_mis}/{len(rows)})", flush=True)
    print(f"  repo dist  {dict(sorted(dist_repo.items()))}")
    print(f"  paper dist {dict(sorted(dist_paper.items()))}")


def retag_gold():
    from b3_eval import load_gold
    _, gold = load_gold()
    order = sorted(gold, key=lambda k: (gold[k]["level"], k))
    rows = []
    for gid in order:
        g = gold[gid]
        n = len(g["nodes"])
        rows.append({"id": gid, "level_repo": g["level"], "n_comp": n,
                     "level_paper": paper_level(n), "gold_nodes": dict(Counter(g["nodes"].values()))})
    _write("level_retag_gold", rows, ["id", "repo", "n_comp", "paper", "match"], "gold node-count")


def retag_testbench(model: str, limit: int | None):
    from mcp_servers.llm_client import create_client
    client = create_client(model)
    ps = json.load((HERE / "e2_testbench_prompts.json").open())["prompts"]
    if limit:
        ps = ps[:limit]
    rows = []
    for i, p in enumerate(ps):
        r = client.complete([{"role": "user", "content": p["prompt"]}], system=COUNT_SYS)
        m = re.search(r"-?\d+", (r.content or ""))
        n = int(m.group()) if m else None
        rows.append({"id": p["id"], "level_repo": p["level"], "n_comp": n,
                     "level_paper": paper_level(n) if n is not None else None, "prompt": p["prompt"]})
        print(f"[{i+1}/{len(ps)}] {p['id']} repo=L{p['level']} n={n} "
              f"paper=L{rows[-1]['level_paper']}", flush=True)
    _write(f"level_retag_testbench{'_smoke' if limit else ''}", rows,
           ["id", "repo", "n_comp", "paper", "match", "prompt"], f"testbench LLM-count ({model})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--testbench", action="store_true")
    ap.add_argument("--model", default="o1")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.gold:
        retag_gold()
    if a.testbench:
        retag_testbench(a.model, a.limit)
    if not (a.gold or a.testbench):
        ap.error("pass --gold and/or --testbench")
