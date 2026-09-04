"""Recompute A2 faithfulness pooled + per-type, BEFORE vs AFTER filling Poon's 35 missing labels.

Validation gate: the BEFORE run must reproduce the thesis numbers (precision 123/174=0.707;
per-type HAS_PROPERTY 0.781 n=64, PERFORMS_FUNCTION 0.533 n=30, BASED_ON_PRINCIPLE 0.833 n=24,
RELATED_TO 0.69 n=32, CONTAINS_COMPONENT 0.64 n=14, USES_COMPONENT 0.60 n=10). Only if it does
are the AFTER numbers trustworthy.
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANNOT = ["poon", "sharma", "mu", "xue", "liu"]  # roszko excluded (double-label of xue, IAA only)
VALID = {"supported", "not_supported"}


def load_labels(path):
    d = json.load(open(path))
    body = d["labels"] if isinstance(d, dict) and "labels" in d else d
    return {l["id"]: l.get("label") for l in body}


def pool(fill_poon35: bool):
    prec = defaultdict(lambda: [0, 0])   # relation -> [supported, not_supported]
    rec = [0, 0]                          # recall arm: [supported(=gate should've kept), not]
    prec_overall = [0, 0]
    for a in ANNOT:
        labels = load_labels(HERE / f"labels_{a}.json")
        if a == "poon" and fill_poon35:
            m35 = load_labels(HERE / "bundles" / "labels_poon_missing35.json")
            for i, lab in m35.items():
                if labels.get(i) is None:
                    labels[i] = lab
        meta = json.load(open(HERE / "bundles" / f"bundle_meta_{a}.json"))["items"]
        for i, lab in labels.items():
            if lab not in VALID:
                continue
            m = meta.get(i)
            if not m:
                continue
            arm, rel = m.get("arm"), m.get("relation")
            if arm == "precision":
                prec_overall[0 if lab == "supported" else 1] += 1
                prec[rel][0 if lab == "supported" else 1] += 1
            elif arm == "recall":
                rec[0 if lab == "supported" else 1] += 1
    return prec, prec_overall, rec


def show(tag, prec, prec_overall, rec):
    tp, fp = prec_overall
    n = tp + fp
    print(f"\n===== {tag} =====")
    print(f"PRECISION pooled: {tp}/{n} = {tp / n:.3f}")
    print(f"{'relation':24} {'sup':>4} {'not':>4} {'n':>4} {'rate':>6}")
    for rel in sorted(prec, key=lambda r: -(prec[r][0] + prec[r][1])):
        s, ns = prec[rel]
        print(f"{rel:24} {s:>4} {ns:>4} {s + ns:>4} {s / (s + ns):>6.3f}")
    # recall arm: gate-recall on RELATED_TO = fraction of expert-valid rejected edges the gate kept
    rs, rns = rec
    print(f"RECALL arm labeled: supported(valid)={rs} not_supported={rns} total={rs + rns}")


before = pool(fill_poon35=False)
after = pool(fill_poon35=True)
show("BEFORE (Poon 45 labeled, 35 excluded) — must match thesis", *before)
show("AFTER (Poon 80 labeled, +35)", *after)
