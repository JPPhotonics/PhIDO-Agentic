# Triple faithfulness labeling tool

A zero-hosting, zero-install, fully asynchronous tool for collecting expert
**faithfulness** judgments on knowledge-graph triples — used to **validate the LLM-judge**
for the KB benchmark (see `../../KB_BENCHMARK_PLAN.md` §5c).

Each recruit gets one self-contained `.html` file with *their* triples baked in. They
open it in any browser, label at their own pace (offline is fine), click **Download**, and
email back one `labels_<name>.json`. Nothing is hosted; the live knowledge graph is never
exposed or modified.

> **What we measure, and what we don't.** Annotators judge *faithfulness* — "does the shown
> quote **support** this statement?" — **not** scientific correctness. The faithfulness
> framing is deliberate: it's answerable without deep re-reading, it stays within the
> benchmark's stated scope boundary, and it dampens author-generosity bias when people
> label their own papers.

## The three pieces

| File | Runs where | Does |
|---|---|---|
| `make_labeling_bundles.py` | once, by you, against live Neo4j (graphRAG branch) | builds `labeler_<id>.html` + private `bundle_meta_<id>.json` per annotator |
| `labeler_template.html` | the recruit's browser (`file://`) | the labeling UI; open it directly to preview the built-in demo |
| `ingest_labels.py` | offline, after the `labels_*.json` come back | computes per-paper rates, judge↔expert agreement (+κ), clustered CI, overlap & timing |

## Workflow

1. **Prerequisite (yours):** ingest each recruit's paper into the Neo4j KB
   (`process_papers.py`). The export selects triples by the edge `source_document` property,
   so the papers must already be in the graph.

2. **Export.** Write a config mapping each recruit to their paper:

   ```json
   {
     "annotators": [
       {"annotator_id": "alice", "source_document": "Smith2021_MZM"},
       {"annotator_id": "bob",   "source_document": "Lee2020_RingMod"}
     ],
     "cap": 40,
     "ooo_fraction": 0.6,
     "overlap_ids": ["<elementId(r)>", "..."],
     "seed": 0,
     "out_dir": "bundles"
   }
   ```

   Then `python make_labeling_bundles.py config.json`. `ooo_fraction` oversamples
   **out-of-ontology** edges (where the judge is uncalibrated, so they carry the most
   validation value). `overlap_ids` is an optional shared set every annotator labels — the
   only way to get a human↔human consistency number with this "one paper each" design.

3. **Send.** Email each recruit their `labeler_<id>.html` plus the 3-line instructions
   below. Keep the `bundle_meta_<id>.json` files — they hold `confidence` + the
   out-of-ontology flag, deliberately kept *out* of the recruit's file to avoid anchoring.

4. **Collect & analyse.** Drop the returned `labels_<id>.json` next to the meta files and
   run `python ingest_labels.py --dir bundles [--judge judge_labels.json]`. The judge file
   is `{triple_id: "supported"|"not_supported"}`; with it you get the headline
   **expert↔judge agreement**, including on the out-of-ontology subset.

## Message to send recruits (copy-paste)

> Hi — for my thesis I need a quick check of some statements a system extracted from your
> paper. **Open the attached file in any web browser** (just double-click it). For each
> statement, decide **only whether the quote shown supports it** — *not* whether it's the
> full or perfect truth about your work. Use **Skip** if a quote is missing or you're
> unsure. It takes about 10–15 minutes; your progress saves automatically, so you can pause
> and reopen. When the bar reaches the end, click **Download** and reply with that file.
> Thank you!

## Notes

- **Reporting is conditional** (`ingest_labels.py` reflects this): per-paper rates and
  judge-agreement are always shown; the pooled **cluster t-interval** is printed but flagged
  *exploratory* — with ≤5 documents it has too few degrees of freedom to be a headline
  number. The validated judge, not this CI, measures KG-wide precision.
- **`confidence` never reaches the annotator** (anchoring bias); re-join it from
  `bundle_meta` by triple id during analysis.
- Edges with **no `evidence_quotes`** are excluded from labeling (they can't be
  faithfulness-judged) and counted separately as a graph-health signal.
- Labels live only in flat JSON files; the knowledge graph is not written to.
