"""E1 arm for the RIGID pipeline's component-selection mechanism: LLM-over-JSON.

The E1 "lexical" configuration is the agentic pipeline's own graph-free catalog tool
(``search_components``), NOT the rigid PhIDO pipeline's selector. The rigid pipeline's p200
stage selects components with ``llm_api.llm_search``: every library cell's docstring is
serialized to JSON inside the system prompt and a structured-output LLM call returns a
ranked ``match_list``. This driver runs that exact function (source untouched; only the
provider entry point ``llm_api.callgpt_pydantic`` is monkeypatched, as in
``run_qwen_rigid_baseline.py``) over the same E1 query suites and candidate pool, so the
rigid mechanism can be placed on the E1 tables next to lexical / KG / hybrid.

Two context variants (arms):
  * ``docstring``  — full module docstrings, exactly what the shipped p200 passes (list_of_docs);
  * ``namedesc``   — name + description only, information-matched to the lexical arm.

Env knobs:
  PROVIDER      openrouter (default) | openai | google
  MODEL         openrouter: "openai/o3-mini" (the model callgpt_pydantic hard-codes on this branch)
  ARMS          "docstring" (default) or "docstring,namedesc"
  QUERY_SETS    "testbench,curated" (default)
  STRATA        testbench strata to include: "named,functional" (default)
  LIMIT         cap on queries per set (pilot)
  K             repeats per query (default 1; the mechanism is non-deterministic)
  BUDGET_USD    hard stop on accumulated provider-reported cost (default 0.80)
  WORKERS       concurrent calls (default 3)
  WITH_KG=1     also score the KG / hybrid arms on the identical subset (loads embedding model)

Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmark/run_e1_llm_json.py
Resume is status-aware: infra-classed reps (rate limit, connection, server, timeout, empty
response) are purged and re-run; legitimate failures (parse failure, empty match list) are
kept and score zero, per the failure-as-zero rule.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(HERE))
sys.path.append(str(ROOT))

# .env (manual parse; no dotenv dependency)
for _line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    if "=" in _line and not _line.lstrip().startswith("#"):
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from e1_retrieval import lexical_arm                                  # noqa: E402
from metrics import mean_pass_at_k, mrr, retrievable_coverage_at_k, set_prf  # noqa: E402
from report import Reporter                                           # noqa: E402
from run_e1_stratified import NAMED_KEYWORDS, is_named                # noqa: E402  (single source of the rule)

PROVIDER = os.getenv("PROVIDER", "openrouter")
MODEL = os.getenv("MODEL", {"openrouter": "openai/o3-mini", "openai": "o3-mini", "google": "gemini-2.5-pro"}[PROVIDER])
ARMS = [a.strip() for a in os.getenv("ARMS", "docstring").split(",") if a.strip()]
QUERY_SETS = [q.strip() for q in os.getenv("QUERY_SETS", "testbench,curated").split(",") if q.strip()]
STRATA = {s.strip() for s in os.getenv("STRATA", "named,functional").split(",") if s.strip()}
LIMIT = int(os.getenv("LIMIT", "0"))
K = int(os.getenv("K", "1"))
BUDGET_USD = float(os.getenv("BUDGET_USD", "0.80"))
WORKERS = int(os.getenv("WORKERS", "3"))
PRICE_IN, PRICE_OUT = float(os.getenv("PRICE_IN", "1.10")), float(os.getenv("PRICE_OUT", "4.40"))  # $/M, o3-mini list
STORE = HERE / "results" / "e1_llm_json_store.json"
OUT = HERE / "results" / "e1_llm_json.json"
RETRYABLE = ("error:RateLimitError", "error:APIConnectionError", "error:InternalServerError",
             "error:APITimeoutError", "error:EmptyResponseError", "error:BudgetExhausted",
             "error:APIStatusError402", "error:APIStatusError429", "error:APIStatusError5")

# ───────────────────────────── provider patch (usage-tracked) ─────────────────────────────
_tls = threading.local()
_spend_lock = threading.Lock()
SPENT = {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


class BudgetExhausted(RuntimeError):
    pass


def _record(prompt_tokens: int, completion_tokens: int, cost: float | None) -> None:
    usd = cost if cost is not None else (prompt_tokens * PRICE_IN + completion_tokens * PRICE_OUT) / 1e6
    with _spend_lock:
        SPENT["usd"] += usd
        SPENT["prompt_tokens"] += prompt_tokens
        SPENT["completion_tokens"] += completion_tokens
        SPENT["calls"] += 1
    getattr(_tls, "usage", []).append({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "usd": usd})


def _openai_compatible_pydantic(prompt, sys_prompt, pydantic_model):
    from openai import OpenAI
    with _spend_lock:
        if SPENT["usd"] >= BUDGET_USD:
            raise BudgetExhausted(f"spent {SPENT['usd']:.3f} >= budget {BUDGET_USD}")
    if PROVIDER == "openrouter":
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        extra = {"usage": {"include": True}}
    else:
        client = OpenAI()
        extra = None
    kw = dict(model=MODEL, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
              response_format=pydantic_model)
    if extra:
        kw["extra_body"] = extra
    # Optional cap so OpenRouter's per-request credit reservation (sized by max output) fits a
    # small balance. Shipped callgpt_pydantic sets no cap; keep this far above observed usage.
    if int(os.getenv("MAX_COMPLETION_TOKENS", "0")) > 0:
        # openai-sdk 1.43 `parse()` has no max_completion_tokens kwarg; send the standard field via body.
        kw.setdefault("extra_body", {})["max_tokens"] = int(os.environ["MAX_COMPLETION_TOKENS"])
    completion = client.beta.chat.completions.parse(**kw)
    u = completion.usage
    cost = None
    if u is not None:
        extra_u = getattr(u, "model_extra", None) or {}
        cost = extra_u.get("cost")
        _record(u.prompt_tokens or 0, u.completion_tokens or 0, cost)
    if not getattr(completion, "choices", None):
        raise EmptyResponseError("no choices")
    message = completion.choices[0].message
    if message.parsed:
        return message.parsed
    raise ValueError(f"structured parse failed: refusal={getattr(message, 'refusal', None)!r}")


class EmptyResponseError(RuntimeError):
    pass


from PhotonicsAI.Photon import llm_api  # noqa: E402

if PROVIDER in ("openrouter", "openai"):
    llm_api.callgpt_pydantic = _openai_compatible_pydantic
elif PROVIDER == "google":
    _orig_google = llm_api.callgoogle_pydantic

    def _google_tracked(prompt, sys_prompt, pydantic_model):
        r = _orig_google(prompt, sys_prompt, pydantic_model)
        _record(0, 0, 0.0)  # usage printed by llm_api, not returned
        return r
    llm_api.callgpt_pydantic = _google_tracked
else:
    raise SystemExit(f"unknown PROVIDER {PROVIDER}")


# ───────────────────────────── candidates & contexts ─────────────────────────────
def build_candidates() -> list[dict]:
    """Same pool and same Cypher as run_e1_retrieval / run_e1_stratified (name + description)."""
    from neo4j import GraphDatabase
    from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
    cfg = Neo4jConfig()
    cy = ("MATCH (p:PDK_Cell) RETURN p.module_name AS id, coalesce(p.display_name,p.module_name) AS name, "
          "coalesce(p.description,'') AS description ORDER BY id")
    with GraphDatabase.driver(cfg.uri, auth=(cfg.username, cfg.password)) as drv, drv.session() as s:
        return [{"id": r["id"], "name": r["name"], "description": r["description"]} for r in s.run(cy) if r["id"]]


def build_contexts(candidates: list[dict]) -> dict[str, list[str]]:
    from PhotonicsAI.Photon import utils
    docs = {c["module_name"]: c["docstring"] for c in utils.search_directory_for_docstrings()}
    missing = [c["id"] for c in candidates if c["id"] not in docs]
    if missing:
        raise SystemExit(f"candidates without library docstring: {missing}")
    return {
        "docstring": [docs[c["id"]] for c in candidates],                      # what p200 ships
        "namedesc": [f"{c['name']}: {c['description']}" for c in candidates],  # lexical-matched info
    }


def load_queries(cand_ids: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if "testbench" in QUERY_SETS:
        spec = json.loads((HERE / "e1_queries_testbench.json").read_text())["queries"]
        qs = []
        for q in spec:
            gold = {m for m in q["gold"] if m in cand_ids}
            if not gold:
                continue
            stratum = "named" if is_named(q["query"]) else "functional"
            if stratum in STRATA:
                qs.append({"id": q["id"], "query": q["query"], "gold": gold, "stratum": stratum})
        out["testbench"] = qs[:LIMIT] if LIMIT else qs
    if "curated" in QUERY_SETS:
        spec = json.loads((HERE / "e1_queries.json").read_text())["queries"]
        qs = [{"id": q["id"], "query": q["query"], "gold": {m for m in q["gold"] if m in cand_ids}, "stratum": "curated"}
              for q in spec]
        qs = [q for q in qs if q["gold"]]
        out["curated"] = qs[:LIMIT] if LIMIT else qs
    return out


# ───────────────────────────── one call ─────────────────────────────
def run_one(query: str, contexts: list[str], candidates: list[dict]) -> dict:
    _tls.usage = []
    t0 = time.time()
    rec: dict = {"status": "ok", "ranked": [], "scores": [], "comment": ""}
    try:
        r = llm_api.llm_search(query, contexts)          # the rigid mechanism, untouched
        seen, ranked, scores = set(), [], []
        for i, sc in zip(r.match_list, r.match_scores + [""] * max(0, len(r.match_list) - len(r.match_scores))):
            if isinstance(i, int) and 0 <= i < len(candidates) and candidates[i]["id"] not in seen:
                seen.add(candidates[i]["id"])
                ranked.append(candidates[i]["id"])
                scores.append(sc)
        rec.update(ranked=ranked, scores=scores, comment=r.match_comment, raw_match_list=list(r.match_list))
        if not ranked:
            rec["status"] = "fail:EmptyMatchList"          # legitimate failure → scores zero
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        code = getattr(e, "status_code", None)
        # Provider-side refusals are infrastructure, not model behaviour: 402 (OpenRouter
        # credit reservation for in-flight requests), 429 (rate limit), 5xx (server).
        if code in (402, 429) or (isinstance(code, int) and code >= 500):
            name = "APIStatusError%d" % code
        rec["status"] = f"error:{name}"
        rec["error"] = str(e)[:300]
    rec["latency_s"] = round(time.time() - t0, 1)
    rec["usage"] = list(_tls.usage)
    rec["usd"] = round(sum(u["usd"] for u in _tls.usage), 6)
    return rec


# ───────────────────────────── scoring ─────────────────────────────
def _metrics(pairs: list[tuple[list[str], set[str]]], curated: bool) -> dict:
    out = {"n": len(pairs), "pass@1": mean_pass_at_k(pairs, 1), "pass@3": mean_pass_at_k(pairs, 3), "mrr": mrr(pairs)}
    if curated:
        out["coverage@3"] = retrievable_coverage_at_k(pairs, 3)
        sp = [set_prf(r[:1], g)[0] for r, g in pairs]
        out["set_precision@1"] = sum(sp) / len(sp) if sp else float("nan")
    return out


def score_ranked(ranked_by_qid: dict[str, list[str]], queries: list[dict], qset: str) -> dict:
    pairs_all = [(ranked_by_qid.get(q["id"], []), q["gold"]) for q in queries]
    res = {"overall": _metrics(pairs_all, qset == "curated")}
    if qset == "testbench":
        for st in ("named", "functional"):
            sub = [(ranked_by_qid.get(q["id"], []), q["gold"]) for q in queries if q["stratum"] == st]
            res[st] = _metrics(sub, False)
    return res


def _fmt(d: dict, keys=("pass@1", "pass@3", "mrr", "coverage@3", "set_precision@1")) -> list[str]:
    return [f"{d[k]:.3f}" if k in d else "" for k in keys]


# ───────────────────────────── main ─────────────────────────────
def main() -> None:
    candidates = build_candidates()
    cand_ids = {c["id"] for c in candidates}
    contexts = build_contexts(candidates)
    qsets = load_queries(cand_ids)
    n_total = sum(len(v) for v in qsets.values())
    print(f"provider={PROVIDER} model={MODEL} arms={ARMS} K={K} budget=${BUDGET_USD} | candidates={len(candidates)} | "
          + ", ".join(f"{k}={len(v)}" for k, v in qsets.items()), flush=True)

    if os.getenv("DRY") == "1":
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        for arm in ARMS:
            # rebuild the exact system prompt llm_search will send, to size it
            desc_json = json.dumps(dict(enumerate(contexts[arm])), indent=2)
            n = len(enc.encode(desc_json)) + 350
            print(f"[DRY] arm={arm}: system prompt ≈ {n} tokens; {n_total * K} calls ≈ {n_total * K * n / 1e6:.2f}M prompt tokens "
                  f"≈ ${n_total * K * (n * PRICE_IN + 700 * PRICE_OUT) / 1e6:.2f} at list price (700 completion tok/call assumed)")
        return

    store = json.loads(STORE.read_text()) if STORE.exists() else {}
    purged = 0
    for arm in list(store):
        for model in list(store[arm]):
            for qid in list(store[arm][model]):
                keep = [r for r in store[arm][model][qid] if not str(r.get("status", "")).startswith(RETRYABLE)]
                purged += len(store[arm][model][qid]) - len(keep)
                store[arm][model][qid] = keep
    if purged:
        print(f"[resume] purged {purged} infra-classed rep(s) for re-run", flush=True)

    jobs = []
    for arm in ARMS:
        for qset, queries in qsets.items():
            for q in queries:
                have = len(store.setdefault(arm, {}).setdefault(MODEL, {}).setdefault(q["id"], []))
                for rep in range(have, K):
                    jobs.append((arm, qset, q, rep))
    print(f"{len(jobs)} call(s) to make ({n_total * K * len(ARMS)} total, {n_total * K * len(ARMS) - len(jobs)} resumed)", flush=True)

    def _save():
        STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps(store, indent=1))
        tmp.replace(STORE)

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, q["query"], contexts[arm], candidates): (arm, qset, q, rep) for arm, qset, q, rep in jobs}
        for fut in as_completed(futs):
            arm, qset, q, rep = futs[fut]
            rec = fut.result()
            rec.update(rep=rep, qset=qset, stratum=q["stratum"], query=q["query"], gold=sorted(q["gold"]))
            store[arm][MODEL][q["id"]].append(rec)
            done += 1
            if done % 5 == 0 or done == len(jobs):
                _save()
                print(f"  [{done}/{len(jobs)}] spent=${SPENT['usd']:.3f} tokens={SPENT['prompt_tokens']}+{SPENT['completion_tokens']} "
                      f"last={q['id']} {rec['status']} top={rec['ranked'][:1]}", flush=True)
    _save()

    # ---- score: LLM-over-JSON arms (per rep, then mean over reps) + free deterministic lexical on the same subset
    results: dict = {}
    infra_n = 0
    for arm in ARMS:
        results[f"llm_json({arm})"] = {}
        for qset, queries in qsets.items():
            per_rep = []
            for rep in range(K):
                ranked = {}
                for q in queries:
                    reps = [r for r in store[arm][MODEL].get(q["id"], []) if r.get("rep") == rep]
                    if not reps:
                        continue          # infra-classed and not yet re-run → excluded, not zero
                    r = reps[0]
                    if str(r["status"]).startswith(RETRYABLE):
                        infra_n += 1
                        continue
                    ranked[q["id"]] = r["ranked"] if r["status"] == "ok" else []
                sub = [q for q in queries if q["id"] in ranked]
                if sub:
                    per_rep.append(score_ranked(ranked, sub, qset))
            if per_rep:
                mean = {}
                for section in per_rep[0]:
                    mean[section] = {k: (sum(pr[section][k] for pr in per_rep) / len(per_rep) if k != "n" else per_rep[0][section]["n"])
                                     for k in per_rep[0][section]}
                results[f"llm_json({arm})"][qset] = {"mean_over_reps": mean, "per_rep": per_rep, "K": len(per_rep)}

    lex = {}
    for qset, queries in qsets.items():
        ranked = {q["id"]: lexical_arm(q["query"], candidates) for q in queries}
        lex[qset] = score_ranked(ranked, queries, qset)
    results["lexical (same subset)"] = lex

    if os.getenv("WITH_KG") == "1":
        from kg_retrieve import make_kg_retriever
        from mcp_servers.component_retrieval import retrieve_candidates
        from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
        from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
        client = Neo4jClient(config=Neo4jConfig()); client.connect()
        kg_embed = make_kg_retriever(include_enrichment=False, client=client, weighted=True, functional=False)

        def _prod_hybrid(query, cands):
            r = json.loads(retrieve_candidates(query, top_k=len(cands), client=client))
            return [c["module_name"] for c in r.get("results", [])]
        for name, arm_fn in {"kg_embed (same subset)": kg_embed, "hybrid(prod) (same subset)": _prod_hybrid}.items():
            results[name] = {qset: score_ranked({q["id"]: arm_fn(q["query"], candidates) for q in queries}, queries, qset)
                             for qset, queries in qsets.items()}

    payload = {"provider": PROVIDER, "model": MODEL, "K": K, "arms": ARMS, "n_candidates": len(candidates),
               "queries": {k: len(v) for k, v in qsets.items()}, "strata": sorted(STRATA), "limit": LIMIT,
               "spend": SPENT, "infra_excluded_reps": infra_n, "results": results}
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[json] -> {OUT}")

    # ---- Markdown report
    rep = Reporter(OUT.with_suffix(".md"), "E1: the rigid pipeline's LLM-over-JSON selection on the E1 query suites",
                   meta={"provider/model": f"{PROVIDER} / {MODEL}", "arms": ", ".join(ARMS), "K (repeats)": K,
                         "candidates": len(candidates), "queries": ", ".join(f"{k}={len(v)}" for k, v in qsets.items()),
                         "testbench strata": ", ".join(sorted(STRATA)), "limit": LIMIT or "none",
                         "spend": f"${SPENT['usd']:.3f} over {SPENT['calls']} calls ({SPENT['prompt_tokens']}+{SPENT['completion_tokens']} tok)",
                         "infra-classed reps excluded": infra_n})
    rep.line("The `llm_json(*)` rows run `llm_api.llm_search` unchanged (the rigid p200 selector); `docstring` passes the full "
             "module docstrings exactly as the shipped pipeline does, `namedesc` passes only name+description (the information the "
             "lexical arm sees). Non-`llm_json` rows are deterministic arms re-scored on the identical query subset so the comparison is paired.")
    rep.line("The rigid pipeline consumes only `match_list[0]`, so **pass@1** is the operationally relevant metric for it.")
    for qset in qsets:
        sections = ["overall", "named", "functional"] if qset == "testbench" else ["overall"]
        for sec in sections:
            rows = []
            for name, r in results.items():
                if qset not in r:
                    continue
                d = r[qset]["mean_over_reps"] if "mean_over_reps" in r[qset] else r[qset]
                if sec in d and d[sec]["n"]:
                    rows.append([name, d[sec]["n"], *_fmt(d[sec])])
            if rows:
                rep.h(f"{qset} — {sec}")
                rep.table(["arm", "n", "pass@1", "pass@3", "mrr", "coverage@3", "set_prec@1"], rows)
    # per-rep spread
    for name, r in results.items():
        for qset, d in r.items():
            if isinstance(d, dict) and d.get("K", 1) > 1:
                rep.h(f"{name} / {qset}: per-repeat pass@1 (overall)")
                rep.line(", ".join(f"{pr['overall']['pass@1']:.3f}" for pr in d["per_rep"]))
    rep.h("Failures")
    fails = [(arm, qid, r["status"], r.get("error", "")[:120]) for arm in ARMS for qid, reps in store[arm][MODEL].items()
             for r in reps if r["status"] != "ok"]
    rep.table(["arm", "query", "status", "error"], fails or [["—", "—", "none", ""]])
    rep.h("Read")
    rep.line("- Named/functional split uses `run_e1_stratified.is_named` (keyword rule; list in that file).")
    rep.line("- Infra-classed reps (rate limit, connection, server, timeout, empty response, budget stop) are excluded and re-run on resume; "
             "parse failures and empty match lists are legitimate failures and score zero.")
    rep.line("- Gold is arm-independent (same files as the other E1 drivers). Query subsets under LIMIT/STRATA are pilots, not powered estimates.")
    rep.save()


if __name__ == "__main__":
    main()
