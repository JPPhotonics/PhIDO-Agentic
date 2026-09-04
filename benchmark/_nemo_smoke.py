import os
os.environ.setdefault("E2_MODEL","nvidia/nemotron-3-ultra-550b-a55b:free")
from dotenv import load_dotenv; load_dotenv(".env")
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("E2_GOLD",str(ROOT/"benchmark/b3_gold_v2.json"))
os.environ.setdefault("E2_PROMPTS",str(ROOT/"benchmark/e2_prompts_v2.json"))
import sentence_transformers as _st
_c={}; _o=_st.SentenceTransformer
_st.SentenceTransformer=lambda m=None,*a,**k:_c.setdefault(str(m),_o(m,*a,**k))
import _score_correctness as SC
from mcp_servers import llm_client
import json,time
GOLD={e["id"]:e for e in json.load(open(ROOT/"benchmark/b3_gold_v2.json"))["gold"]}
p=GOLD["L3_01"]["prompt"]
print("PROMPT:",p[:100])
t0=time.time()
topo,status,dot=SC._run_agentic_cfg(p,kg=False,gate=False,critic=False,disable_enforcement=True)
print(f"\nSTATUS={status}  {time.time()-t0:.0f}s  calls={len(llm_client.trace_dump())}  cost=${llm_client.USAGE['cost']:.4f}")
print("nodes:",dict(getattr(topo,'nodes',{}) or {}))
print("edges:",len(getattr(topo,'edges',[]) or []))
