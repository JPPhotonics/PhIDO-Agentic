"""Build a self-contained HTML tool for BLIND, product-quality human review of an E2 ablation run.

Rationale
---------
The rigid PhIDO outcome taxonomy (EE/CS/SG/PC/L/S) is a *process-attrition* scheme: it labels the
earliest stage at which a linear pipeline *halts*. It does not translate to the agentic runs, which
(a) almost never halt (netlist-production ~1.0) and (b) execute non-linearly (explore/grounding,
critic loop, topology gate feed back on each other), so "the stage it failed at" is ill-posed.

This tool instead grades the *final design artifact* on three independent axes, applicable equally
to the rigid and agentic arms and computable purely from what is already on disk (no re-run):

  A1  Component fidelity   — are the right building blocks present?      Correct / Minor / Wrong
  A2  Connectivity fidelity— is it wired into the right circuit?         Correct / Partial / Wrong
  A3  Intent satisfaction  — would this plausibly fulfil the prompt?     Yes / Partial / No

PC (parameters) and L (layout) are intentionally absent: this run's driver (``_score_correctness.py``)
captured neither numeric parameters (the DOT labels carry only role + PDK module) nor a GDS layout.

Every run's schematic is re-rendered *uniformly* from its raw predicted topology
(``pred_node_map`` + ``pred_edges``), not from the app's own DOT — so all 648 have a picture and no
visual-style cue leaks which arm/level produced it. Grading is BLIND: the arm and the automatic
metrics are hidden in the UI (they live in the payload only for the aggregation join).

Usage:
    python benchmark/build_review_html.py \
        [--results benchmark/results/_ablation_correctness_gpt54_v2.json] \
        [--gold    benchmark/b3_gold_v2.json] \
        [--prompts benchmark/e2_prompts_v2.json] \
        [--out     benchmark/results/_ablation_review.html] \
        [--no-render]   # skip PNG regeneration (reuse existing schematics dir)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import _thesis_schematic

ROOT = Path(__file__).resolve().parent.parent
SHUFFLE_SEED = 42  # fixed → stable grading order across rebuilds

# Product-quality rubric. Each axis: id, label, options [(value, key, text)], keyed for the keyboard.
RUBRIC = [
    {"id": "A1", "label": "Component fidelity",
     "help": "Are the right building blocks present (types & counts)?",
     "options": [["correct", "1", "Correct"], ["minor", "2", "Minor"], ["wrong", "3", "Wrong"]],
     "tags": ["missing", "spurious", "wrong-type", "wrong-count"]},
    {"id": "A2", "label": "Connectivity fidelity",
     "help": "Is it wired into the right circuit topology?",
     "options": [["correct", "4", "Correct"], ["partial", "5", "Partial"], ["wrong", "6", "Wrong"]],
     "tags": ["missing-edge", "wrong-edge", "port-mismatch", "disconnected", "wrong-class"]},
    {"id": "A3", "label": "Intent satisfaction",
     "help": ("Would this design plausibly fulfil the prompt's functional intent? "
              "Numeric parameters (arm lengths, gaps, drive voltages, radii) are NOT captured in "
              "this run: judge intent on components and topology only. If the prompt specifies "
              "parameter values, tag it — their satisfaction is unverifiable here."),
     "options": [["yes", "7", "Yes"], ["partial", "8", "Partial"], ["no", "9", "No"]],
     "tags": ["param-specified-unverifiable"]},
]


# Canonical (left/input, right/output) optical port layout per abstract node class, taken from
# b3_gold_v2 `_meta.port_conventions`. Lets the schematic draw EVERY port at its correct side,
# including ports the run left unconnected.
CLASS_PORTS = {
    "STRAIGHT": (["o1"], ["o2"]), "BEND": (["o1"], ["o2"]), "RING_1BUS": (["o1"], ["o2"]),
    "HEATER": (["o1"], ["o2"]), "MZI_1X1": (["o1"], ["o2"]), "MOD_1X1": (["o1"], ["o2"]),
    "PD": (["o1"], ["o2"]), "_ELECTRICAL": (["o1"], ["o2"]),
    "GC": ([], ["o1"]),
    "MMI_1X2": (["o1"], ["o2", "o3"]), "MZI_1X2": (["o1"], ["o2", "o3"]),
    "PSR": (["o1"], ["o2", "o3"]),
    "DC_2X2": (["o1", "o2"], ["o3", "o4"]), "MMI_2X2": (["o1", "o2"], ["o3", "o4"]),
    "MZI_2X2": (["o1", "o2"], ["o3", "o4"]), "CROSSING": (["o1", "o2"], ["o3", "o4"]),
    "COUPLER_RING": (["o1", "o2"], ["o3", "o4"]),
    "RING_2BUS": (["o1", "o4"], ["o2", "o3"]),   # in/add on left; through/drop on right
    "WDM": (["o1"], ["o2", "o3", "o4", "o5"]),
}


def _class_ports(cls: str):
    """(left_ports, right_ports) for a node class, or None if unknown (parses a trailing _NxM)."""
    key = (cls or "").upper()
    if key in CLASS_PORTS:
        left, right = CLASS_PORTS[key]
        return list(left), list(right)
    m = re.search(r"_(\d+)x(\d+)$", key)
    if m:
        ni, no = int(m.group(1)), int(m.group(2))
        return ([f"o{i}" for i in range(1, ni + 1)],
                [f"o{i}" for i in range(ni + 1, ni + no + 1)])
    return None


def _modules_from_dot(dot_string: str) -> dict:
    """node_id -> PDK module name, parsed from the pipeline's own dot_string labels
    ('Ck: Role (module)'). Returns {} if no dot_string. Node ids match pred_node_map."""
    out = {}
    for nid, lab in re.findall(r'^\s*([A-Za-z0-9_]+)\s*\[label="(.*?)"\];', dot_string or "", re.MULTILINE):
        m = re.search(r"\(([^)]+)\)", lab.replace("\\n", " "))
        if m:
            out[nid] = m.group(1)
    return out


def topology_to_dot(node_map: dict, edges: list, title: str = "", modules: dict | None = None) -> str:
    """Reconstruct a uniform Graphviz record schematic from a raw predicted topology.

    node_map: {instance_id: class}. edges: [[ "id.port", "id.port" ], ...] (raw, pre-collapse).
    Every device with a known class is drawn with exactly its canonical port set (inputs left,
    outputs right, per the b3_gold port conventions), wired or not, so the port count matches the
    device spec (e.g. a 1x2 always shows o1 | o2,o3). An edge to an out-of-spec port still routes
    (Graphviz anchors it to the node body) but does not add a phantom port. Unknown classes fall
    back to inferring ports from the wiring.
    """
    def split_ep(ep):
        return ep.rsplit(".", 1) if "." in ep else (ep, None)

    seen = {nid: [] for nid in node_map}
    for e in edges:
        for ep in e:
            nid, p = split_ep(ep)
            if nid in seen and p and p not in seen[nid]:
                seen[nid].append(p)

    lines = ['graph G {', '  rankdir=LR;', '  node [shape=record, fontsize=10, fontname="Helvetica"];',
             '  edge [color="#555555"];']
    for nid, cls in node_map.items():
        cp = _class_ports(cls)
        if cp is None:                                  # unknown class: infer from wired ports
            s = seen[nid]
            left, right = (s[:1], s[1:]) if s else ([], [])
        else:
            left, right = cp                 # exact canonical port set (no phantom out-of-spec ports)
        blk = lambda ps: "|".join(f"<{p}> {p}" for p in ps)
        mod = (modules or {}).get(nid)
        lab = (f"{nid}: {cls} ({mod})" if mod else f"{nid}: {cls}").replace('"', "'")
        L, R = blk(left), blk(right)
        if L and R:
            label = f"{{{{{L}}} | {lab} | {{{R}}}}}"
        elif L:
            label = f"{{{{{L}}} | {lab}}}"
        elif R:
            label = f"{{{lab} | {{{R}}}}}"
        else:
            label = lab
        lines.append(f'  "{nid}" [label="{label}"];')
    for e in edges:
        if len(e) != 2:
            continue
        (a, ap), (b, bp) = split_ep(e[0]), split_ep(e[1])
        sa = f'"{a}":{ap}' if ap else f'"{a}"'
        sb = f'"{b}":{bp}' if bp else f'"{b}"'
        lines.append(f"  {sa} -- {sb};")
    lines.append("}")
    return "\n".join(lines)


def render_png(dot_string: str, dst: Path) -> bool:
    if not dot_string or not shutil.which("dot"):
        return False
    try:
        subprocess.run(["dot", "-Tpng", "-o", str(dst)], input=dot_string.encode(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        return dst.exists()
    except (subprocess.SubprocessError, OSError):
        return False


def build(results_path: Path, gold_path: Path, prompts_path: Path, out_path: Path,
          do_render: bool = True) -> None:
    results = json.loads(results_path.read_text())
    gold = {g["id"]: g for g in json.loads(gold_path.read_text())["gold"]}
    prompts = {p["id"]: p for p in json.loads(prompts_path.read_text())["prompts"]}

    schem_dirname = f"{out_path.stem}_schematics"
    schem_dir = out_path.parent / schem_dirname
    schem_dir.mkdir(exist_ok=True)

    runs, rendered, no_topo = [], 0, 0
    for arm, per_prompt in results.items():
        for pid, reps in per_prompt.items():
            for i, rec in enumerate(reps):
                rid = f"{arm}::{pid}::rep{i + 1}"
                node_map = rec.get("pred_node_map") or {}
                edges = rec.get("pred_edges") or []
                modules = _modules_from_dot(rec.get("dot_string"))
                png_name = f"{rid.replace('::', '__')}.png"
                png_abs = schem_dir / png_name
                schem = None
                if node_map:
                    if do_render or not png_abs.exists():
                        # thesis-style (Ch6) directed/flow-oriented layout when possible;
                        # fall back to the original record layout on a malformed topology.
                        dot = _thesis_schematic.safe_to_dot(pid, node_map, edges, modules) \
                            or topology_to_dot(node_map, edges, pid, modules)
                        if render_png(dot, png_abs):
                            rendered += 1
                            schem = f"{schem_dirname}/{png_name}"
                        else:
                            schem = None
                    elif png_abs.exists():
                        schem = f"{schem_dirname}/{png_name}"
                else:
                    no_topo += 1
                g = gold.get(pid, {})
                p = prompts.get(pid, {})
                runs.append({
                    "run_id": rid, "arm": arm, "pid": pid, "rep": i + 1,
                    "level_paper": rec.get("level_paper") or rec.get("level"),
                    "status": rec.get("status"), "error": rec.get("error", ""),
                    "prompt": p.get("prompt") or g.get("prompt") or "",
                    "pred_node_map": node_map, "pred_edges": edges, "pred_modules": modules,
                    "schematic": schem,
                    "gold_nodes": g.get("nodes") or {}, "gold_edges": g.get("edges") or [],
                    "gold_rationale": g.get("rationale", ""),
                    # hidden from the blind UI; carried for the aggregation join / post-hoc reveal
                    "_compF1": rec.get("compF1"), "_edgeF1": rec.get("edgeF1"), "_ged": rec.get("ged"),
                })

    # Shuffle for arm-blinding, then group runs of the same prompt together so the grader
    # can stay on one prompt at a time. list.sort is stable, so the shuffled (arm-blind)
    # order is preserved WITHIN each prompt group; only the arm is hidden (the prompt text
    # is shown regardless), so grouping by prompt does not weaken the blinding.
    random.Random(SHUFFLE_SEED).shuffle(runs)
    runs.sort(key=lambda r: r["pid"])

    payload = {
        "meta": {"results_file": results_path.name, "gold_file": gold_path.name,
                 "n_runs": len(runs), "seed": SHUFFLE_SEED},
        "rubric": RUBRIC,
        "runs": runs,
    }
    out_path.write_text(HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)))
    print(f"Wrote {out_path}\n  runs={len(runs)}  schematics_rendered={rendered}  "
          f"no_topology(pipeline-fail)={no_topo}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>E2 ablation — blind design review</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --panel2:#1d222b; --line:#2a303c; --fg:#e6e9ef;
          --mut:#98a2b3; --accent:#4a9eff; --gold:#f0b429; --good:#3fb950; --warn:#d29922; --bad:#f85149; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:8px 14px; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; gap:12px; align-items:center; flex-wrap:wrap; position:sticky; top:0; z-index:10; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .grow { flex:1; }
  select, input[type=text] { background:var(--panel2); color:var(--fg); border:1px solid var(--line);
           border-radius:6px; padding:5px 8px; font-size:13px; }
  button { background:var(--panel2); color:var(--fg); border:1px solid var(--line); border-radius:6px;
           padding:5px 10px; cursor:pointer; font-size:13px; }
  button:hover { border-color:var(--accent); }
  .prog { font-size:12px; color:var(--mut); } .prog b { color:var(--fg); }
  label.tgl { font-size:12px; color:var(--mut); display:flex; gap:5px; align-items:center; cursor:pointer; }
  .layout { display:flex; height:calc(100vh - 47px); }
  aside { width:210px; border-right:1px solid var(--line); overflow-y:auto; background:var(--panel); }
  .row { padding:6px 10px; border-bottom:1px solid var(--line); cursor:pointer; font-size:12px;
         display:flex; gap:8px; align-items:center; }
  .row:hover { background:var(--panel2); } .row.active { background:#243040; }
  .row .rid { flex:1; } .dot0 { color:var(--mut); } .doto { color:var(--good); }
  main { flex:1; overflow-y:auto; padding:16px 22px; }
  .meta { color:var(--mut); font-size:12px; margin-bottom:6px; }
  .promptbox { background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent);
               border-radius:6px; padding:10px 12px; font-size:15px; margin-bottom:12px; }
  .cols { display:grid; grid-template-columns:1.2fr .8fr; gap:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
  .card h3 { margin:0 0 8px; font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--mut); }
  .card.gold h3 { color:var(--gold); }
  img.dot { width:100%; background:#fff; border-radius:6px; cursor:zoom-in; }
  table.top { width:100%; border-collapse:collapse; font-size:12px; }
  table.top td { padding:2px 6px; border-bottom:1px solid var(--line); vertical-align:top; }
  table.top td:first-child { color:var(--mut); white-space:nowrap; }
  .edges { font-family:ui-monospace,monospace; font-size:11px; white-space:pre-wrap; }
  details { margin-top:10px; } summary { cursor:pointer; color:var(--mut); font-size:12px; }
  .statusbad { color:var(--bad); font-weight:600; }
  .rubric { position:sticky; bottom:0; background:var(--panel); border-top:1px solid var(--line);
            padding:10px 22px; margin:16px -22px -16px; }
  .axis { display:flex; gap:10px; align-items:center; margin:6px 0; flex-wrap:wrap; }
  .axis .name { width:170px; font-size:12px; } .axis .name b { display:block; font-size:13px; }
  .axis .name small { color:var(--mut); }
  .opt { min-width:78px; padding:7px 10px; font-weight:600; }
  .opt.sel { background:var(--accent); color:#04223f; border-color:var(--accent); }
  .opt.sel.good { background:var(--good); color:#04210d; border-color:var(--good); }
  .opt.sel.warn { background:var(--warn); color:#221803; border-color:var(--warn); }
  .opt.sel.bad  { background:var(--bad);  color:#2a0606; border-color:var(--bad); }
  .tags { display:flex; gap:6px; flex-wrap:wrap; margin-left:6px; }
  .tag { font-size:11px; padding:3px 7px; border-radius:10px; }
  .tag.sel { background:#33507a; border-color:var(--accent); color:#dbe9ff; }
  .notes { width:100%; margin-top:8px; background:var(--panel2); color:var(--fg);
           border:1px solid var(--line); border-radius:6px; padding:6px 8px; font-size:12px; }
  .legend { font-size:11px; color:var(--mut); margin-top:6px; }
  .nav { display:flex; gap:8px; margin-left:auto; }
  .reveal { font-size:12px; color:var(--warn); margin-top:4px; }
  dialog { background:#fff; border:none; border-radius:8px; padding:0; max-width:97vw; max-height:97vh; }
  dialog img { max-width:97vw; max-height:93vh; display:block; } dialog::backdrop { background:rgba(0,0,0,.85); }
  .hide { display:none !important; }
  .caveat { background:#3a2c12; color:#ffd98a; border-bottom:1px solid #6b4e1e; padding:6px 14px; font-size:12px; }
  .caveat b { color:#ffe9b0; }
</style>
</head>
<body>
<header>
  <h1>E2 blind design review</h1>
  <span class="prog" id="prog"></span>
  <span class="grow"></span>
  <input type="text" id="fSearch" placeholder="filter prompt / #…" size="14">
  <select id="fState"><option value="">all</option><option value="todo">incomplete</option><option value="done">complete</option></select>
  <label class="tgl"><input type="checkbox" id="blind" checked> blind</label>
  <button id="btnExport">Export</button>
  <button id="btnImport">Import</button>
  <input type="file" id="fileImport" accept="application/json" class="hide">
</header>
<div class="caveat">Parameters (arm lengths, gaps, drive voltages, radii) are <b>not captured</b> in this run: each node shows only its PDK module, not parameter values. For prompts that specify parameter values, grade A3 on components and topology only and add the <b>param-specified-unverifiable</b> tag.</div>
<div class="layout">
  <aside id="list"></aside>
  <main id="main"></main>
</div>
<dialog id="zoom"><img id="zoomImg" src=""></dialog>
<script>
const DATA = __PAYLOAD__;
const LS_KEY = "e2_review3::" + DATA.meta.results_file;
let ann = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
let filtered = [], cur = 0;
const AXES = DATA.rubric.map(a=>a.id);
const OPTCLASS = {correct:"good",yes:"good",minor:"warn",partial:"warn",wrong:"bad",no:"bad"};

function save(){ localStorage.setItem(LS_KEY, JSON.stringify(ann)); }
function rec(r){ return ann[r.run_id] || (ann[r.run_id]={axes:{},tags:{},note:""}); }
function complete(r){ const a=ann[r.run_id]; return a && AXES.every(x=>a.axes && a.axes[x]); }
function esc(s){ return (s??"").toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function isBlind(){ return document.getElementById("blind").checked; }

function applyFilters(){
  const q=fSearch.value.trim().toLowerCase(), st=fState.value;
  filtered = DATA.runs.filter((r,idx)=>{
    r._idx = idx;
    if(st==="todo" && complete(r)) return false;
    if(st==="done" && !complete(r)) return false;
    if(q && !((r.prompt||"").toLowerCase().includes(q) || String(idx+1).includes(q))) return false;
    return true;
  });
  if(cur>=filtered.length) cur=Math.max(0,filtered.length-1);
  renderList(); renderMain();
}

function renderProg(){
  const n=DATA.runs.filter(complete).length;
  prog.innerHTML=`complete <b>${n}</b>/${DATA.meta.n_runs} · showing <b>${filtered.length}</b>`;
}
function renderList(){
  list.innerHTML="";
  const groupIds=[...new Set(filtered.map(r=>r.pid))];
  let lastPid=null;
  filtered.forEach((r,i)=>{
    if(r.pid!==lastPid){
      lastPid=r.pid;
      const gi=groupIds.indexOf(r.pid)+1;
      const grp=filtered.filter(x=>x.pid===r.pid);
      const gdone=grp.filter(complete).length;
      const h=document.createElement("div");
      h.style.cssText="font-size:11px;font-weight:700;color:#555;padding:8px 4px 2px;border-top:1px solid #ccc;margin-top:4px";
      h.textContent=`prompt ${gi}/${groupIds.length} · level ${r.level_paper} · ${gdone}/${grp.length} done`;
      list.appendChild(h);
    }
    const div=document.createElement("div");
    div.className="row"+(i===cur?" active":"");
    const done=complete(r);
    div.innerHTML=`<span class="${done?'doto':'dot0'}">${done?'●':'○'}</span>
      <span class="rid">#${r._idx+1}${isBlind()?'':' · '+esc(r.arm)}</span>`;
    div.onclick=()=>{cur=i; renderList(); renderMain();};
    list.appendChild(div);
  });
  renderProg();
}

function topoTable(map, edges, modules){
  const rows=Object.entries(map).map(([k,v])=>{
    const m=modules&&modules[k];
    return `<tr><td>${esc(k)}</td><td>${esc(v)}${m?` <span class="dot0">(${esc(m)})</span>`:""}</td></tr>`;
  }).join("") || `<tr><td colspan=2 class="dot0">(no components)</td></tr>`;
  const e=edges.length ? edges.map(x=>Array.isArray(x)?x.join("  —  "):esc(x)).join("\n") : "(no edges)";
  return `<table class="top"><tr><td><b>id</b></td><td><b>type</b></td></tr>${rows}</table>
    <details><summary>edges (${edges.length})</summary><div class="edges">${esc(e)}</div></details>`;
}

function renderMain(){
  if(!filtered.length){ main.innerHTML="<p class='dot0'>No runs match.</p>"; renderProg(); return; }
  const r=filtered[cur], a=rec(r), blind=isBlind();
  const bad = r.status && r.status!=="scored";
  const schem = r.schematic
    ? `<img class="dot" src="${esc(r.schematic)}" onclick="openZoom('${esc(r.schematic)}')">`
    : `<p class="statusbad">No circuit produced (${esc(r.status||'?')}).${r.error?' '+esc(r.error):''}</p>`;
  const revealed = !blind ? `<div class="reveal">arm: <b>${esc(r.arm)}</b> · rep ${r.rep} · ${esc(r.status)}
     · compF1 ${r._compF1} · edgeF1 ${r._edgeF1} · GED ${r._ged}</div>` : "";

  main.innerHTML = `
    <div class="meta">run #${r._idx+1} of ${DATA.meta.n_runs} · level ${r.level_paper}
       ${bad?`· <span class="statusbad">${esc(r.status)}</span>`:""}</div>
    <div class="promptbox">${esc(r.prompt)}</div>
    <div class="cols">
      <div class="card"><h3>Predicted design (schematic)</h3>${schem}
        <div style="margin-top:10px">${topoTable(r.pred_node_map, r.pred_edges, r.pred_modules)}</div></div>
      <div class="card gold"><h3>Gold topology</h3>${topoTable(r.gold_nodes, r.gold_edges)}
        ${r.gold_rationale?`<details open style="margin-top:8px"><summary>rationale</summary>
          <div style="font-size:12px">${esc(r.gold_rationale)}</div></details>`:""}</div>
    </div>
    <div class="rubric">
      ${DATA.rubric.map(ax=>`
        <div class="axis">
          <span class="name"><b>${ax.id} ${esc(ax.label)}</b><small>${esc(ax.help)}</small></span>
          ${ax.options.map(o=>{
            const sel=a.axes[ax.id]===o[0];
            return `<button class="opt ${sel?'sel '+(OPTCLASS[o[0]]||''):''}"
              data-ax="${ax.id}" data-val="${o[0]}" title="key ${o[1]}">${o[2]}</button>`;}).join("")}
          ${ax.tags.length?`<span class="tags">${ax.tags.map(t=>{
            const on=(a.tags[ax.id]||[]).includes(t);
            return `<button class="tag ${on?'sel':''}" data-ax="${ax.id}" data-tag="${t}">${t}</button>`;
          }).join("")}</span>`:""}
        </div>`).join("")}
      <div class="axis">
        <textarea class="notes" id="notes" placeholder="notes (optional)…">${esc(a.note||"")}</textarea>
      </div>
      <div class="legend">Keys: <b>1/2/3</b> A1 · <b>4/5/6</b> A2 · <b>7/8/9</b> A3 ·
        <b>n/→</b> next · <b>p/←</b> prev · auto-advances when all three axes set. ${revealed}</div>
    </div>`;

  main.querySelectorAll(".opt").forEach(b=>b.onclick=()=>setAxis(b.dataset.ax,b.dataset.val));
  main.querySelectorAll(".tag").forEach(b=>b.onclick=()=>toggleTag(b.dataset.ax,b.dataset.tag));
  document.getElementById("notes").oninput=e=>{ rec(r).note=e.target.value; save(); };
}

function setAxis(ax,val){
  const r=filtered[cur]; const a=rec(r);
  a.axes[ax] = (a.axes[ax]===val) ? undefined : val;
  if(!a.axes[ax]) delete a.axes[ax];
  save(); const wasComplete=complete(r); renderList(); renderMain();
  if(wasComplete) setTimeout(()=>move(1),140);
}
function toggleTag(ax,tag){
  const a=rec(filtered[cur]); a.tags[ax]=a.tags[ax]||[];
  const i=a.tags[ax].indexOf(tag);
  if(i<0) a.tags[ax].push(tag); else a.tags[ax].splice(i,1);
  save(); renderMain();
}
function move(d){ cur=Math.min(filtered.length-1,Math.max(0,cur+d)); renderList(); renderMain(); main.scrollTop=0; }
function openZoom(src){ zoomImg.src=src; zoom.showModal(); }
zoom.onclick=()=>zoom.close();

const KEYMAP={};
DATA.rubric.forEach(ax=>ax.options.forEach(o=>KEYMAP[o[1]]=[ax.id,o[0]]));
document.addEventListener("keydown",e=>{
  if(["INPUT","TEXTAREA"].includes(e.target.tagName)) return;
  if(e.key in KEYMAP){ e.preventDefault(); setAxis(...KEYMAP[e.key]); }
  else if(e.key==="n"||e.key==="ArrowRight"){ e.preventDefault(); move(1); }
  else if(e.key==="p"||e.key==="ArrowLeft"){ e.preventDefault(); move(-1); }
});

btnExport.onclick=()=>{
  const out={meta:{...DATA.meta, exported:DATA.runs.filter(complete).length}, annotations:ann};
  const blob=new Blob([JSON.stringify(out,null,2)],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download=DATA.meta.results_file.replace(/\.json$/,"")+"_review3_labels.json"; a.click();
};
btnImport.onclick=()=>fileImport.click();
fileImport.onchange=e=>{ const f=e.target.files[0]; if(!f) return; const rd=new FileReader();
  rd.onload=()=>{ try{ const j=JSON.parse(rd.result); ann=j.annotations||j; save(); applyFilters();
    alert("Imported "+Object.keys(ann).length+" annotations."); }catch(err){ alert("Bad JSON: "+err); } };
  rd.readAsText(f); };

document.getElementById("blind").onchange=()=>{ renderList(); renderMain(); };
[fSearch].forEach(el=>el.oninput=applyFilters);
fState.onchange=applyFilters;
applyFilters();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="benchmark/results/_ablation_correctness_gpt54_v2.json")
    ap.add_argument("--gold", default="benchmark/b3_gold_v2.json")
    ap.add_argument("--prompts", default="benchmark/e2_prompts_v2.json")
    ap.add_argument("--out", default="benchmark/results/_ablation_review.html")
    ap.add_argument("--no-render", action="store_true", help="reuse existing schematic PNGs")
    args = ap.parse_args()
    build(ROOT / args.results, ROOT / args.gold, ROOT / args.prompts, ROOT / args.out,
          do_render=not args.no_render)


if __name__ == "__main__":
    main()
