"""
enrich.py
Optional enrichment pass: sends the raw findings from one run to a free-tier
LLM (Groq by default) to dedupe/cluster, classify signal quality better than
regex, and produce a short synthesis paragraph.

Design rule (borrowed from pmf-engine's bias registry): enrichment is
ADDITIVE, never destructive. Raw findings in the `findings` table are never
edited or dropped by this step. Everything here writes to a separate
`enrichment` table, so a bad or biased LLM summary can always be re-derived
or ignored without losing the underlying evidence.

Requires GROQ_API_KEY (free: https://console.groq.com). Uses the OpenAI-
compatible chat completions endpoint so you can swap base_url/model to any
other OpenAI-compatible free provider (OpenRouter free models, local Ollama
serving an OpenAI-compatible endpoint, etc.) without changing this file.
"""

import json
import os
import sys
from dataclasses import asdict

import urllib.request
import urllib.error

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")

ENRICH_SCHEMA = """
CREATE TABLE IF NOT EXISTS enrichment (
    run_id TEXT PRIMARY KEY,
    topic TEXT,
    synthesis TEXT,          -- short paragraph: landscape, gaps, confidence
    cluster_count INTEGER,   -- deduped competitor/finding count
    top_gap TEXT,            -- single clearest unmet need, if any, else 'none found'
    model_used TEXT,
    created_at TEXT
);
"""

SYSTEM_PROMPT = """You analyze raw web-search findings about a possible product idea \
(a Blender add-on). You will get a JSON list of findings, each with pass_type, \
platform, domain, title, snippet, and regex-derived signal flags.

Your job:
1. Merge findings that are clearly the same underlying product/listing/repost \
   (e.g. two search hits pointing at the same gumroad page, or a dev's own \
   plugin variants) into clusters. Report the deduped cluster_count.
2. Write a synthesis paragraph (120-180 words): what's already on the market, \
   how crowded it is, what evidence (if any) of unmet pain exists, and what the \
   adversarial-pass findings say (do NOT omit or soften adversarial findings -- \
   report them even if they undercut the opportunity).
3. Name the single clearest gap you can support with the evidence given, or \
   say "none found" if you can't -- do not invent a gap to sound useful.

Output ONLY minified JSON: {"cluster_count": int, "synthesis": str, "top_gap": str}
No markdown, no preamble, no code fences.
"""


def call_llm(findings: list) -> dict:
    if not GROQ_API_KEY:
        return {"cluster_count": None, "synthesis": "[enrichment skipped: GROQ_API_KEY not set]", "top_gap": "n/a"}

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(findings)[:24000]},  # keep request small/cheap
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    req = urllib.request.Request(
        LLM_BASE_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            # Groq's Cloudflare front-end 403s bare urllib requests with no
            # User-Agent (error code 1010) -- this header is required, not cosmetic.
            "User-Agent": "research-agent/1.0 (+https://github.com/)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        print(f"[llm] enrichment call failed: HTTP {e.code} -- {body}", file=sys.stderr)
        return {"cluster_count": None, "synthesis": f"[enrichment failed: HTTP {e.code}]", "top_gap": "n/a"}
    except urllib.error.URLError as e:
        print(f"[llm] enrichment call failed: {e.reason}", file=sys.stderr)
        return {"cluster_count": None, "synthesis": f"[enrichment failed: {e.reason}]", "top_gap": "n/a"}

    raw = data["choices"][0]["message"]["content"].strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"cluster_count": None, "synthesis": raw[:1000], "top_gap": "unparsed"}


def enrich_run(client, topic: str, run_id: str, findings: list, created_at: str) -> dict:
    client.execute(ENRICH_SCHEMA)
    payload = [
        {"pass_type": f.pass_type, "platform": f.platform, "domain": f.domain,
         "title": f.title, "snippet": f.snippet, "has_paid_signal": f.has_paid_signal,
         "has_adversarial_signal": f.has_adversarial_signal}
        for f in findings
    ]
    result = call_llm(payload)
    client.execute(
        """INSERT OR REPLACE INTO enrichment
           (run_id, topic, synthesis, cluster_count, top_gap, model_used, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        [run_id, topic, result.get("synthesis", ""), result.get("cluster_count"),
         result.get("top_gap", ""), LLM_MODEL if GROQ_API_KEY else "none", created_at],
    )
    return result
