"""Qwen trace analysis: join 180 traces to per-run outcomes, contrast tool-interaction patterns
for success vs failure, and characterize the failure modes (critic KeyError, DesignIntent miss)."""
import glob
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["E2_MODEL"] = "qwen/qwen3.6-27b"
os.environ["E2_GOLD"] = str(ROOT / "benchmark" / "b3_gold_v2.json")
os.environ["E2_PROMPTS"] = str(ROOT / "benchmark" / "e2_prompts_v2.json")
import _score_correctness as SC  # noqa: E402
from topology_eval import Topology  # noqa: E402


def _edges(el):
    out = []
    for e in el:
        eps = re.findall(r"'([^']+)'", e) if isinstance(e, str) else list(e)
        if len(eps) == 2:
            out.append(frozenset(eps))
    return out


def _edgef1(rec, pid):
    if rec.get("status") != "ok" or not rec.get("nodes"):
        return None
    try:
        return SC._score_topo(Topology(nodes=dict(rec["nodes"]), edges=_edges(rec.get("edges", [])),
                                       external={}), pid)["edgeF1"]
    except Exception:  # noqa: BLE001
        return None


# outcome map: (arm,pid,rep_idx0) -> (status, edgeF1)
outcome = {}
for f in glob.glob(str(ROOT / "benchmark/results/qwen_trace_ablation_w*.json")):
    for arm, byp in json.load(open(f)).items():
        for pid, reps in byp.items():
            for i, rec in enumerate(reps):
                outcome[(arm, pid, i)] = (rec.get("status"), _edgef1(rec, pid))


def feats(trace):
    tools = [tc["name"] for s in trace for tc in s.get("tool_calls", [])]
    return {
        "n_steps": len(trace),
        "n_tool_calls": len(tools),
        "n_structured": sum(1 for s in trace if s.get("kind") == "structured"),
        "reasoning_chars": sum(len(s.get("reasoning") or "") for s in trace),
        "tools": tools,
    }


# join traces
rows = []
for tf in glob.glob(str(ROOT / "benchmark/results/qwen_traces/*.json")):
    d = json.load(open(tf))
    arm, pid, rep = d["arm"], d["prompt_id"], d["rep"]
    st_, ef = outcome.get((arm, pid, rep - 1), (None, None))
    rows.append({"arm": arm, "pid": pid, "status": st_, "edgeF1": ef, **feats(d["trace"])})

print(f"\njoined {len(rows)} traces to outcomes")


# ---- bucket by outcome ----
def bucket(r):
    if r["status"] != "ok":
        return "FAILED"
    if r["edgeF1"] is None:
        return "ok(no-edge)"
    return "HIGH(ef>=.8)" if r["edgeF1"] >= 0.8 else ("MID(.5-.8)" if r["edgeF1"] >= 0.5 else "LOW(<.5)")


by = defaultdict(list)
for r in rows:
    by[bucket(r)].append(r)
print("\n===== trace features by outcome bucket (mean) =====")
print(f"  {'bucket':14} {'n':>4} {'steps':>6} {'toolcalls':>10} {'reasoning_chars':>16}")
for b in ["HIGH(ef>=.8)", "MID(.5-.8)", "LOW(<.5)", "FAILED", "ok(no-edge)"]:
    g = by.get(b, [])
    if not g:
        continue
    print(f"  {b:14} {len(g):>4} {st.mean(r['n_steps'] for r in g):6.1f} "
          f"{st.mean(r['n_tool_calls'] for r in g):10.1f} {st.mean(r['reasoning_chars'] for r in g):16.0f}")

# ---- tool frequency: success vs failure ----
def toolfreq(rs):
    c = Counter(t for r in rs for t in r["tools"])
    tot = sum(c.values()) or 1
    return {k: f"{v} ({100 * v / tot:.0f}%)" for k, v in c.most_common(8)}


print("\n===== tool usage: HIGH-edgeF1 vs FAILED =====")
print("  HIGH:", toolfreq(by.get("HIGH(ef>=.8)", [])))
print("  FAILED:", toolfreq(by.get("FAILED", [])))

# ---- failure-mode characterization ----
print("\n===== FAILED traces (status + arm + last tools) =====")
for r in rows:
    if r["status"] != "ok":
        print(f"  {r['arm']:16} {r['pid']}  {r['status']:42} steps={r['n_steps']} "
              f"lastTools={r['tools'][-4:]}")
