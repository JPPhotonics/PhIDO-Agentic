"""Qwen agent-trace analysis (Poon's 2nd ask): what tool-use / pipeline-stage patterns end in
SUCCESSFUL circuits vs the agent's shortfalls. Joins all 360 traces to the human 3-axis labels
(A2 connectivity = did it build the right circuit) and the automatic edge-F1.

Trace step schema: {kind:'complete', tool_calls:[{name}], reasoning} for tool-loop iterations;
{kind:'structured', schema:<PydanticName>} for pipeline stages (ExtractedConcepts,
RequirementManifest, DesignIntent, ComponentSelectionLLM, ComplianceVerdict, CriticVerdict).
"""
import glob
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = json.load(open(ROOT / "benchmark/results/_qwen_review_input_review3_labels.json"))["annotations"]
REV = json.load(open(ROOT / "benchmark/results/_qwen_review_input.json"))

SEARCH = {"search_pdk", "search_knowledge_graph", "get_concept_neighborhood", "search_pdk_by_function"}
VALIDATE = {"validate_ports", "get_component_properties", "get_component_info",
            "get_pdk_cell_details", "get_pdk_implementations", "get_module_params", "resolve_function"}
STAGES = ["ExtractedConcepts", "RequirementManifest", "DesignIntent", "ComponentSelectionLLM",
          "ComplianceVerdict", "CriticVerdict"]


def feats(trace):
    tools = [tc["name"] for s in trace for tc in s.get("tool_calls", [])]
    schemas = [s.get("schema") for s in trace if s.get("kind") == "structured"]
    nt = len(tools)
    return {
        "tools": tools, "n_tool": nt, "n_distinct": len(set(tools)),
        "n_search": sum(t in SEARCH for t in tools), "n_val": sum(t in VALIDATE for t in tools),
        "val_ratio": (sum(t in VALIDATE for t in tools) / nt) if nt else 0.0,
        "reasoning": sum(len(s.get("reasoning") or "") for s in trace),
        "n_complete": sum(1 for s in trace if s.get("kind") == "complete"),
        "stages": set(schemas),
        "reached_di": "DesignIntent" in schemas,
        "reached_sel": "ComponentSelectionLLM" in schemas,
        "critic_fired": "CriticVerdict" in schemas,
    }


rows = []
for tf in glob.glob(str(ROOT / "benchmark/results/qwen_traces/*.json")):
    d = json.load(open(tf))
    arm, pid, rep = d["arm"], d["prompt_id"], d["rep"]
    lab = LAB.get(f"{arm}::{pid}::rep{rep}", {}).get("axes", {})
    try:
        rec = REV[arm][pid][rep - 1]
    except (KeyError, IndexError):
        rec = {}
    rows.append({"arm": arm, "pid": pid, "rep": rep, "A1": lab.get("A1"), "A2": lab.get("A2"),
                 "A3": lab.get("A3"), "status": rec.get("status"), "edgeF1": rec.get("edgeF1"),
                 **feats(d["trace"])})

joined = [r for r in rows if r["A2"]]
print(f"traces={len(rows)}  joined-to-human={len(joined)}")


# success bucket by human connectivity (A2): did the agent build the right circuit?
def bucket(r):
    return {"correct": "SUCCESS", "partial": "PARTIAL", "wrong": "FAIL"}.get(r["A2"], "?")


by = defaultdict(list)
for r in joined:
    by[bucket(r)].append(r)

print("\n===== trace features by SUCCESS bucket (human A2 = connectivity) =====")
print(f"  {'bucket':9} {'n':>4} {'tools':>6} {'distinct':>8} {'search':>7} {'valid':>6} "
      f"{'val%':>5} {'reason':>7} {'reachDI%':>9} {'critic%':>8}")
for b in ["SUCCESS", "PARTIAL", "FAIL"]:
    g = by.get(b, [])
    if not g:
        continue
    m = lambda k: st.mean(r[k] for r in g)
    print(f"  {b:9} {len(g):>4} {m('n_tool'):6.1f} {m('n_distinct'):8.1f} {m('n_search'):7.1f} "
          f"{m('n_val'):6.1f} {100 * m('val_ratio'):5.0f} {m('reasoning'):7.0f} "
          f"{100 * st.mean(r['reached_di'] for r in g):9.0f} {100 * st.mean(r['critic_fired'] for r in g):8.0f}")


def toolfreq(rs):
    c = Counter(t for r in rs for t in r["tools"])
    tot = sum(c.values()) or 1
    return {k: f"{100 * v / tot:.0f}%" for k, v in c.most_common(9)}


print("\n===== tool mix: SUCCESS vs FAIL (share of tool calls) =====")
print("  SUCCESS:", toolfreq(by.get("SUCCESS", [])))
print("  FAIL   :", toolfreq(by.get("FAIL", [])))

print("\n===== pipeline-stage REACH rate by bucket (%) =====")
print(f"  {'stage':22} {'SUCCESS':>8} {'PARTIAL':>8} {'FAIL':>6}")
for stg in STAGES:
    row = [100 * st.mean(stg in r["stages"] for r in by.get(b, [{}]) if r) if by.get(b) else 0
           for b in ["SUCCESS", "PARTIAL", "FAIL"]]
    print(f"  {stg:22} {row[0]:8.0f} {row[1]:8.0f} {row[2]:6.0f}")

# ---- agent shortfalls: failure taxonomy joined to stage-reach ----
print("\n===== AGENT SHORTFALLS: FAIL runs by status × did-it-reach-DesignIntent =====")
fail = by.get("FAIL", [])
tax = Counter()
for r in fail:
    stt = (r["status"] or "?").split(":")[0]
    tax[(stt, "reachedDI" if r["reached_di"] else "no-DI")] += 1
for (stt, di), n in tax.most_common():
    print(f"  {stt:16} {di:12} {n}")

# ---- the shortfall signature: build-wrong-wiring (reached selection but A2=wrong) ----
miswire = [r for r in fail if r["reached_sel"] and r["status"] == "scored"]
print(f"\n  MISWIRE shortfall (reached ComponentSelection + built a circuit, but human=wrong wiring): "
      f"{len(miswire)}/{len(fail)} fails")
print(f"    their mean val_ratio={100 * st.mean(r['val_ratio'] for r in miswire):.0f}% vs "
      f"SUCCESS {100 * st.mean(r['val_ratio'] for r in by['SUCCESS']):.0f}%")

# ---- arm effect on validation behaviour ----
print("\n===== validation intensity by arm (mean validate-calls, mean val%) =====")
byarm = defaultdict(list)
for r in joined:
    byarm[r["arm"]].append(r)
for a in sorted(byarm):
    g = byarm[a]
    print(f"  {a:16} valcalls={st.mean(r['n_val'] for r in g):4.1f}  val%={100 * st.mean(r['val_ratio'] for r in g):3.0f}  "
          f"success%={100 * st.mean(bucket(r) == 'SUCCESS' for r in g):3.0f}")
