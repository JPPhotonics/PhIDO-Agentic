"""B3 — independent topology-correctness eval (gold loader + class-normalized scorer).

Loads the hand-authored, pipeline-INDEPENDENT gold in ``b3_gold.json`` (authored from prompt
semantics + domain knowledge, NOT from any pipeline 4_SG.txt), and scores a predicted
schematic-generation netlist against it. Both sides are normalized to component CLASSES
(function + port arity) via the gold's ``class_map`` BEFORE comparison, so this measures
TOPOLOGY (did it wire the right kinds of blocks together) and not component SELECTION
(which exact PDK cell — that is A5/B2). Scoring itself is delegated to
``topology_eval.score_topology`` (node-id-agnostic typed-edge multiset F1 + component-class
F1 + approx_ged).

Why this exists: ``topology_eval.load_gold_topologies`` reads the four GETTING_STARTED
``4_SG.txt`` files, which are the pipeline's OWN saved output on those prompts -> scoring the
pipeline against its own output is contamination. This module supplies independent gold.

Usage:
  # validate that the hand-authored gold is internally consistent (no dangling refs):
  python benchmark/b3_eval.py --validate-gold
  # score a directory of predicted netlists named <id>.txt (4_SG format) against gold:
  python benchmark/b3_eval.py --pred-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from topology_eval import Topology, parse_sg_netlist, score_topology, validity

GOLD_PATH = Path(__file__).resolve().parent / "b3_gold.json"


def paper_level(n: int) -> int:
    """Map a component count to the PhIDO paper's Table 1 complexity level.

    A named composite block (MZI, ring, WDM, mesh, ...) counts as ONE component. The repo's
    stored ``level`` fields are heuristic/unreviewed and disagree with this definition
    ([[e2-prompt-level-mislabeling-2026-07-09]]); ``level_paper`` is the paper-aligned axis.
        L1 = 1 | L2 = 2 | L3 = 3-15 | L4 = 16-112 (>112 clamps to 4; 0 -> 1 degenerate)
    """
    if n <= 1:
        return 1
    if n == 2:
        return 2
    if 3 <= n <= 15:
        return 3
    return 4


def load_gold(path: Path | None = None) -> tuple[dict, dict[str, dict]]:
    """Return (meta, {id: gold_entry}). class_map lives in meta['class_map'].

    Each entry is augmented with ``level_paper`` (Table 1 level from the gold node count, which
    is already class/component-level so a composite = 1 node) and ``level_repo`` (the original
    unreviewed ``level``). ``level`` is left untouched for backward compatibility.
    """
    data = json.loads((path or GOLD_PATH).read_text(encoding="utf-8"))
    meta = data["_meta"]
    gold = {g["id"]: g for g in data["gold"]}
    for g in gold.values():
        n = len(g["nodes"])
        g["level_repo"] = g["level"]
        g["n_comp"] = n
        g["level_paper"] = paper_level(n)
    return meta, gold


def _entry_to_topology(entry: dict) -> Topology:
    """A gold entry's nodes are already CLASS-level; edges are [[ 'N1.o2', 'N2.o1'], ...]."""
    return Topology(
        nodes=dict(entry["nodes"]),
        edges=[frozenset(e) for e in entry["edges"]],
        external=dict(entry.get("external", {})),
    )


def normalize_prediction(sg_text: str, class_map: dict[str, str]) -> Topology:
    """Parse a predicted 4_SG netlist and relabel every node's component to its CLASS.

    Unknown component module names pass through unchanged (so a genuinely wrong / made-up
    component is visible as a class miss rather than being silently coerced).
    """
    pred = parse_sg_netlist(sg_text)
    pred.nodes = {nid: class_map.get(comp, comp) for nid, comp in pred.nodes.items()}
    return pred


def _apply_accept(pred: Topology, entry: dict) -> Topology:
    """Best-effort acceptance of functionally-equivalent component classes.

    The scorer is node-id-agnostic (multisets), so we canonicalize at the CLASS level: any
    predicted class that appears in ANY of this entry's accept-sets is relabeled to the
    canonical gold class for that accept-set. Documented limitation: if two distinct gold
    nodes have overlapping accept-sets this is ambiguous; none of the current gold does.
    """
    accept = entry.get("accept")
    if not accept:
        return pred
    alt_to_canon: dict[str, str] = {}
    for nid, alts in accept.items():
        canon = entry["nodes"][nid]
        for a in alts:
            alt_to_canon.setdefault(a, canon)
    pred.nodes = {nid: alt_to_canon.get(c, c) for nid, c in pred.nodes.items()}
    return pred


def score_prediction(sg_text: str, entry: dict, class_map: dict[str, str]) -> dict:
    pred = _apply_accept(normalize_prediction(sg_text, class_map), entry)
    gold = _entry_to_topology(entry)
    out = score_topology(pred, gold)
    out["id"] = entry["id"]
    out["level"] = entry["level"]
    out["topology_certain"] = entry["topology_certain"]
    return out


def validate_gold(path: Path | None = None) -> int:
    """Self-consistency check on the hand-authored gold. Returns count of problems."""
    meta, gold = load_gold(path)
    classes_used: Counter = Counter()
    problems = 0
    print(f"{'id':6} {'lvl':3} {'cert':4} {'nodes':5} {'edges':5}  validity")
    for gid, entry in gold.items():
        t = _entry_to_topology(entry)
        classes_used.update(t.nodes.values())
        v = validity(t)
        ok = v["all_refs_exist"] and not v["floating_instances"]
        # a single-node gold legitimately has 0 edges and 0 external-only -> not "floating"
        flag = "" if ok else f"  <-- {v}"
        if not ok:
            problems += 1
        print(f"{gid:6} {entry['level']:<3} {str(entry['topology_certain'])[0]:4} "
              f"{len(t.nodes):<5} {len(t.edges):<5} {'OK' if ok else 'CHECK'}{flag}")
    # class_map coverage: every class used in gold must be a class_map VALUE
    known = set(meta["class_map"].values())
    unknown = sorted(c for c in classes_used if c not in known)
    print(f"\nclasses used: {dict(classes_used)}")
    if unknown:
        print(f"WARNING: classes not produced by class_map (typos?): {unknown}")
        problems += len(unknown)
    n_certain = sum(g["topology_certain"] for g in gold.values())
    print(f"\n{len(gold)} gold entries | {n_certain} topology_certain | "
          f"{len(gold) - n_certain} need sign-off | {problems} problems")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-gold", action="store_true")
    ap.add_argument("--pred-dir", type=str, default=None,
                    help="dir of <id>.txt predicted 4_SG netlists to score")
    args = ap.parse_args()

    if args.validate_gold or not args.pred_dir:
        raise SystemExit(0 if validate_gold() == 0 else 1)

    meta, gold = load_gold()
    cm = meta["class_map"]
    pred_dir = Path(args.pred_dir)
    rows = []
    for gid, entry in gold.items():
        f = pred_dir / f"{gid}.txt"
        if not f.exists():
            continue
        rows.append(score_prediction(f.read_text(encoding="utf-8"), entry, cm))
    if not rows:
        print(f"no <id>.txt predictions found in {pred_dir}")
        return
    # typed-edge F1 is None for edgeless gold (N/A) — average only over scorable prompts.
    edge_f1s = [r["typed_edge_prf"]["f1"] for r in rows if r["typed_edge_prf"]["f1"] is not None]
    macro_edge_f1 = sum(edge_f1s) / len(edge_f1s) if edge_f1s else float("nan")
    macro_comp_f1 = sum(r["component_prf"]["f1"] for r in rows) / len(rows)
    print(f"scored {len(rows)} predictions | macro typed-edge F1 {macro_edge_f1:.3f} "
          f"(over {len(edge_f1s)} with edges) | macro component-class F1 {macro_comp_f1:.3f}")
    for r in rows:
        ef = r["typed_edge_prf"]["f1"]
        print(f"  {r['id']:6} L{r['level']} edgeF1={'n/a' if ef is None else f'{ef:.2f}'} "
              f"compF1={r['component_prf']['f1']:.2f} ged={r['approx_ged']} "
              f"{'' if r['pred_validity']['all_refs_exist'] else 'INVALID-REFS'}")


if __name__ == "__main__":
    main()
