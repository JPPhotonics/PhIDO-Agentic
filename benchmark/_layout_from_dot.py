"""Post-hoc layout + SAX + DRC for the E2 netlist-level runs (gpt-5.4 ablation, Qwen suite).

The E2 correctness drivers (``_score_correctness.py`` and the Qwen/Nemotron drivers built on it)
stopped at the netlist and persisted, per agentic rep, only the class-mapped topology plus the
app's own DOT schematic (``dot_string``). They did NOT persist ``gf_netlist_yaml``. This driver
rebuilds the gdsfactory netlist from that DOT with the SAME code path the pipeline itself uses
after edge routing (``compute_layout`` -> ``find_open_ports`` -> ``_edges_dot_to_dsl`` ->
``_enrich_circuit_dsl`` -> ``export_gf_netlist``), then runs Phase 8 (``run_layout_simulation``:
GDS render -> SAX -> GDS write -> KLayout DRC) and records the five-stage funnel.

What is and is not recovered
  * instance -> PDK module: recovered exactly (the DOT label carries ``(module)``).
  * port-level wiring: recovered exactly (DOT edges ``A:oX -- B:oY``).
  * placements: recomputed with the pipeline's own Graphviz placement from the same DOT and the
    same PDK footprints -> identical to what the pipeline would have produced.
  * component parameters (``settings``): NOT recoverable. The original runs took PDK defaults
    (``get_module_params``) and overlaid LLM ``user_overrides``; the overrides were never stored.
    This driver uses PDK defaults only. The blind review graded modules + wiring, never
    parameters (see build_review_html.py caveat), so the layout here is the layout of the
    artifact that was graded, at library-default parameters.
  * rigid-baseline reps: NOT reconstructable (only class-mapped nodes were stored and the class
    map is many-to-one), so this driver covers the agentic configurations only.

Usage (serial or process-sharded; each worker patches its own BUILD_DIR so parallel workers do
not clobber ``circuit_output.gds`` / ``report.lydrb``):
  SOURCE=gpt54|qwen N_WORKERS=3 WORKER_ID=0 \
  PYTHONPATH=<wt>:<wt>/benchmark <wt>/.venv-layout/bin/python <wt>/benchmark/_layout_from_dot.py   (.venv-layout: kfactory 0.21.7; .venv cannot route)
Resumable: records are appended to results/_layout_<SOURCE>_w<k>.json; done (arm,pid,rep) skip.
Artifacts per rep -> results/_layout_<SOURCE>_artifacts/<arm>/<pid>_rep<k>.{netlist.yml,gds,png,lydrb}
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

SOURCE = os.getenv("SOURCE", "gpt54")
N_WORKERS = int(os.getenv("N_WORKERS", "1"))
WORKER_ID = int(os.getenv("WORKER_ID", "0"))
TIMEOUT = int(os.getenv("REP_TIMEOUT", "900"))  # per-rep wall-clock cap (gf routing can hang)
LIMIT = int(os.getenv("LIMIT", "0") or 0)  # smoke-test: stop after N reps
ONLY = os.getenv("ONLY")  # smoke-test: "arm:pid:rep" filter
RES = ROOT / "benchmark" / "results"
_suffix = f"_w{WORKER_ID}" if N_WORKERS > 1 else ""
OUT = RES / f"_layout_{SOURCE}{_suffix}.json"
ART = RES / f"_layout_{SOURCE}_artifacts"
# Unique per SOURCE *and* worker: two suites running side by side must never share a build dir
# (circuit_output.gds / report.lydrb are fixed filenames -> cross-process races corrupt DRC verdicts).
BUILD = RES / f"_layout_build_{SOURCE}_w{WORKER_ID}"
BUILD.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- inputs
def load_reps() -> list[dict]:
    """Flatten the stored reps to [{arm, pid, rep, status, dot_string}] (agentic arms only)."""
    reps: list[dict] = []
    if SOURCE == "gpt54":
        d = json.load(open(RES / "_ablation_correctness_gpt54_v2.json"))
        for arm, byp in d.items():
            if arm == "baseline":
                continue
            for pid, rs in byp.items():
                for k, r in enumerate(rs, 1):
                    reps.append({"arm": arm, "pid": pid, "rep": k, "status": r.get("status"),
                                 "dot_string": r.get("dot_string")})
    elif SOURCE == "qwen":
        # Same merge rule as _score_qwen_n24.py: concatenate the w* shards per (arm, pid).
        # Rep id = the stored trace_idx (explicit store<->trace join) so it survives purges.
        for f in sorted(glob.glob(str(RES / "qwen_trace_ablation_w*.json"))):
            for arm, byp in json.load(open(f)).items():
                for pid, rs in byp.items():
                    for k, r in enumerate(rs, 1):
                        reps.append({"arm": arm, "pid": pid,
                                     "rep": int(r.get("trace_idx") or k),
                                     "shard": Path(f).stem, "status": r.get("status"),
                                     "dot_string": r.get("dot_string")})
    else:
        raise SystemExit(f"unknown SOURCE={SOURCE}")
    return reps


# --------------------------------------------------------------------------- DOT parsing
_NODE_RE = re.compile(r'^\s*(\w+)\s*\[label="(.*)"\]\s*;\s*$')
_MOD_RE = re.compile(r"\(([A-Za-z0-9_]+)\)")


def parse_dot_nodes(dot: str) -> dict[str, dict]:
    """node_id -> {module, ports:[o1..], label} from the app's record-shaped DOT."""
    nodes: dict[str, dict] = {}
    for line in dot.splitlines():
        m = _NODE_RE.match(line)
        if not m:
            continue
        nid, label = m.group(1), m.group(2)
        mods = _MOD_RE.findall(label)
        nodes[nid] = {"module": mods[-1] if mods else None,
                      "ports": re.findall(r"<(o\d+)>", label), "label": label}
    return nodes


# --------------------------------------------------------------------------- rebuild
class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)

_pdk_names: list[str] | None = None


def _known_modules() -> list[str]:
    global _pdk_names
    if _pdk_names is None:
        from PhotonicsAI.Photon.DemoPDK import list_of_cnames
        _pdk_names = list(list_of_cnames)
    return _pdk_names


def rebuild_netlist(dot: str) -> tuple[str | None, dict]:
    """DOT -> gdsfactory YAML netlist via the pipeline's own post-routing code path."""
    from mcp_servers.pdk_catalog_server import get_component_footprint
    from mcp_servers.pipeline_orchestrator import (
        _authoritative_port_config,
        _edges_dot_to_dsl,
        _enrich_circuit_dsl,
        _get_ground_truth_params,
    )
    from mcp_servers.schematic_builder_server import (
        compute_layout,
        export_gf_netlist,
        find_open_ports,
    )

    parsed = parse_dot_nodes(dot)
    diag: dict = {"n_nodes": len(parsed), "modules": {}, "unknown_modules": []}
    if not parsed:
        diag["rebuild_error"] = "no nodes parsed from DOT"
        return None, diag
    known = set(_known_modules())
    nodes: dict = {}
    for nid, info in parsed.items():
        mod = info["module"]
        diag["modules"][nid] = mod
        if not mod or mod not in known:
            diag["unknown_modules"].append(f"{nid}:{mod}")
            continue
        nodes[nid] = {
            "component": mod,
            "properties": {"ports": _authoritative_port_config(mod) or ""},
            "params": _get_ground_truth_params(mod),
            "label": info["label"],
        }
    if diag["unknown_modules"]:
        diag["rebuild_error"] = f"module(s) not in PDK: {diag['unknown_modules']}"
        return None, diag

    dsl: dict = {"doc": {"name": "reconstructed"}, "nodes": nodes, "edges": {}, "ports": {}}
    _edges_dot_to_dsl(dot, dsl)
    diag["n_edges"] = len(dsl["edges"])

    footprints: dict[str, list[float]] = {}
    for nid, n in nodes.items():
        try:
            fp = json.loads(get_component_footprint(n["component"]))
            if "error" not in fp:
                footprints[nid] = [fp["dx_um"], fp["dy_um"]]
        except Exception:  # noqa: BLE001 — mirrors the orchestrator (footprint is best-effort)
            pass
    try:
        positions = json.loads(compute_layout(dot, json.dumps(footprints))).get("positions", {})
    except Exception:  # noqa: BLE001
        positions = {}
    try:
        circuit_ports = json.loads(find_open_ports(dot)).get("circuit_ports", {})
    except Exception:  # noqa: BLE001
        circuit_ports = {}
    _enrich_circuit_dsl(dsl, None, positions, footprints, circuit_ports)  # selection unused
    out = export_gf_netlist(json.dumps(dsl))
    if out.lstrip().startswith("{"):
        diag["rebuild_error"] = out[:300]
        return None, diag
    return out, diag


def layout_sim(netlist_yaml: str) -> tuple[dict | None, str | None]:
    """Run Phase 8 on a netlist; returns (layout_sim_done result | None, error message | None)."""
    from mcp_servers.pipeline_orchestrator import run_layout_simulation

    result, err = None, None
    for ev in run_layout_simulation(netlist_yaml):
        t = ev.get("type")
        if t == "layout_sim_done":
            result = ev["result"]
        elif t == "error":
            err = ev.get("message")
    return result, err


def _patch_build_dir(name: str | None = None) -> None:
    """Point the layout server's build dir at a private dir (parallel-safe). ``name`` overrides
    the worker default for auxiliary drivers (e.g. the fidelity check) so they never share one."""
    from mcp_servers import layout_sim_server as L
    d = RES / f"_layout_build_{name}" if name else BUILD
    d.mkdir(parents=True, exist_ok=True)
    L.BUILD_DIR = d


def _save_artifacts(rep: dict, netlist: str | None, ls: dict | None) -> dict:
    d = ART / rep["arm"]
    d.mkdir(parents=True, exist_ok=True)
    tag = f"_{rep['shard'].rsplit('_', 1)[-1]}" if rep.get("shard") else ""  # e.g. "_w2"
    stem = d / f"{rep['pid']}{tag}_rep{rep['rep']}"
    paths: dict = {}
    if netlist:
        (stem.with_suffix(".netlist.yml")).write_text(netlist)
        paths["netlist"] = str(stem.with_suffix(".netlist.yml").relative_to(RES))
    ls = ls or {}
    gp = ls.get("gds_file_path")
    if gp and Path(gp).exists():
        shutil.copyfile(gp, stem.with_suffix(".gds"))
        paths["gds"] = str(stem.with_suffix(".gds").relative_to(RES))
    rp = ls.get("drc_report_path")
    if rp and Path(rp).exists():
        shutil.copyfile(rp, stem.with_suffix(".lydrb"))
        paths["drc_report"] = str(stem.with_suffix(".lydrb").relative_to(RES))
    b64 = ls.get("gds_fig_b64")
    if b64:
        import base64
        try:
            stem.with_suffix(".layout.png").write_bytes(base64.b64decode(b64))
            paths["layout_png"] = str(stem.with_suffix(".layout.png").relative_to(RES))
        except Exception:  # noqa: BLE001
            pass
    return paths


def rep_key(r: dict) -> tuple:
    """Identity of a stored rep. Includes the source shard: early Qwen reps carry no ``trace_idx`` and
    fall back to ``rep=k`` within their shard, so (arm, pid, rep) alone collides across shards."""
    return (r["arm"], r["pid"], r["rep"], r.get("shard"))


def run_rep(rep: dict) -> dict:
    """Rebuild + Phase 8 for one stored rep. Never raises."""
    import e2_runner as R

    t0 = time.time()
    rec = {k: rep[k] for k in ("arm", "pid", "rep", "status") if k in rep}
    if rep.get("shard"):
        rec["shard"] = rep["shard"]
    stages = dict.fromkeys(R.STAGES, False)
    netlist, ls, err, diag = None, None, None, {}
    if not rep.get("dot_string"):
        rec["recon"] = "no_dot"  # rep produced no netlist in the original run
    else:
        signal.alarm(TIMEOUT)
        try:
            netlist, diag = rebuild_netlist(rep["dot_string"])
            if netlist is None:
                rec["recon"] = "rebuild_failed"
            else:
                ls, err = layout_sim(netlist)
                rec["recon"] = "ok" if ls else "layout_error"
                stages = R.funnel_from_results({"gf_netlist_yaml": netlist}, ls)
        except _Timeout:
            rec["recon"] = "timeout"
        except Exception as e:  # noqa: BLE001
            rec["recon"] = f"error:{type(e).__name__}"
            diag["exception"] = str(e)[:300]
        finally:
            signal.alarm(0)
    rec["stages"] = stages
    ls = ls or {}
    rec["diag"] = {
        **diag,
        "layout_error": err,
        "routing_ok": ls.get("routing_ok"),
        "missing_models": ls.get("missing_models"),
        "sim_error": (str(ls.get("sim_error"))[:300] if ls.get("sim_error") else None),
        "gds_write_ok": ls.get("gds_write_ok"),
        "drc_clean": ls.get("drc_clean"),
        "drc_violations": ls.get("drc_violations"),
        "drc_error": ls.get("drc_error"),
    }
    try:
        rec["artifacts"] = _save_artifacts(rep, netlist, ls)
    except Exception as e:  # noqa: BLE001
        rec["artifacts"] = {"error": str(e)[:200]}
    rec["latency_s"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    reps = load_reps()
    if ONLY:
        a, p, k = ONLY.split(":")
        reps = [r for r in reps if r["arm"] == a and r["pid"] == p and str(r["rep"]) == k]
    tasks = [r for i, r in enumerate(reps) if i % N_WORKERS == WORKER_ID]
    store: list[dict] = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {rep_key(r) for r in store}
    todo = [r for r in tasks if rep_key(r) not in done]
    print(f"[w{WORKER_ID}/{N_WORKERS}] SOURCE={SOURCE}: {len(reps)} reps total, "
          f"{len(tasks)} in this shard, {len(done)} done, {len(todo)} to run; BUILD={BUILD}",
          flush=True)
    if os.getenv("DRY") == "1":
        return
    _patch_build_dir()
    n = 0
    for rep in todo:
        rec = run_rep(rep)
        store.append(rec)
        OUT.write_text(json.dumps(store, indent=1, default=str))
        st = rec["stages"]
        reached = [s for s in st if st[s]]
        print(f"[w{WORKER_ID} {rec['arm']:18} {rec['pid']} rep{rec['rep']}] "
              f"{rec['recon']:14} reached={reached[-1] if reached else 'none':12} "
              f"drc={rec['diag'].get('drc_clean')} viol={rec['diag'].get('drc_violations')} "
              f"{rec['latency_s']}s", flush=True)
        n += 1
        if LIMIT and n >= LIMIT:
            break
    print(f"DONE w{WORKER_ID}: {n} reps this invocation -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
