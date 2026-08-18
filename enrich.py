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
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")

ENRICH_SCHEMA = """
CREATE TABLE IF NOT EXISTS enrichment (
    run_id TEXT PRIMARY KEY,
    topic TEXT,
    synthesis TEXT,          -- short paragraph: landscape, gaps, confidence
    cluster_count INTEGER,   -- deduped competitor/finding count
    top_gap TEXT,            -- single clearest unmet need, if any, else 'none found'
    adversarial_n INTEGER,   -- how many adversarial-flagged findings existed in the input
    adversarial_check TEXT,  -- 'ok' | 'MISSING: synthesis has adversarial evidence but never mentions it' | 'n/a: none in input'
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
   IMPORTANT -- precise terminology, checked automatically after your response:
   only use the words "adversarial" or "adversarial finding(s)" if at least one \
   finding in the input actually has pass_type "adversarial" or \
   has_adversarial_signal=true. If there are none, do NOT use that word at all --
   describe complaints/limitations/pain points you see in other findings using \
   plain language instead (e.g. "known limitations", "user-reported issues", \
   "reported performance problems"), never as "adversarial findings", since that \
   specific term implies a maintainer/community explicitly pushed back on the \
   idea, which is a stronger and different claim than "users mentioned a bug".
3. Name the single clearest gap you can support with the evidence given, or \
   say "none found" if you can't -- do not invent a gap to sound useful.

Output ONLY minified JSON: {"cluster_count": int, "synthesis": str, "top_gap": str}
No markdown, no preamble, no code fences.
"""


def _call_groq(messages: list, max_tokens: int = 1200, reasoning_effort: str = "low") -> str | None:
    """Shared low-level Groq call. Returns raw response content string, or
    None on any failure (caller decides the fallback -- this function doesn't
    know whether "no answer" should mean "skip" or "use a heuristic instead")."""
    if not GROQ_API_KEY:
        return None
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.2,
        # gpt-oss models are reasoning models -- Groq bills hidden reasoning
        # tokens against max_tokens too, so a low cap can be entirely eaten
        # by reasoning and leave nothing for the actual answer.
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
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
        print(f"  [llm] call failed: HTTP {e.code} -- {body}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"  [llm] call failed: {e.reason}", file=sys.stderr)
        return None
    content = data["choices"][0]["message"]["content"].strip()
    if not content:
        finish_reason = data["choices"][0].get("finish_reason", "unknown")
        print(f"  [llm] empty response, finish_reason={finish_reason}", file=sys.stderr)
        return None
    return content


KEYWORD_SYSTEM_PROMPT = """You turn a long, descriptive product-research topic into a short \
search-engine query. The topic may be a full sentence or paragraph describing a target \
user, use case, or product idea. Extract the 3-8 word phrase someone would actually type \
into a search engine to find existing products/discussions about this -- concrete nouns \
and the specific technology/domain terms, not the framing language around them.

Example:
Input: "Blender technical artists, VFX studios, and volumetric-capture practitioners who \
need dynamic/temporal Gaussian Splatting data to become a first-class, artist-editable \
citizen inside Blender's native Geometry Nodes procedural system, not just a static viewer."
Output: gaussian splatting geometry nodes blender

Output ONLY the short phrase. No quotes, no explanation, no punctuation at the end.
"""


def extract_search_keywords(topic: str) -> str | None:
    """LLM-based shortening for long/descriptive topics, so narrow query
    templates (adversarial, official-source exact/label search) have
    something a search engine or GitHub's search API can actually match
    against, instead of a full paragraph. Returns None on any failure --
    caller should fall back to the heuristic, non-LLM shortener."""
    content = _call_groq(
        [{"role": "system", "content": KEYWORD_SYSTEM_PROMPT}, {"role": "user", "content": topic}],
        max_tokens=60, reasoning_effort="low",
    )
    if not content:
        return None
    keywords = content.strip().strip('"').strip()
    return keywords or None


def call_llm(findings: list) -> dict:
    if not GROQ_API_KEY:
        return {"cluster_count": None, "synthesis": "[enrichment skipped: GROQ_API_KEY not set]", "top_gap": "n/a"}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(findings)[:24000]},  # keep request small/cheap
    ]
    raw = _call_groq(messages, max_tokens=1200, reasoning_effort="low")
    if raw is None:
        return {"cluster_count": None, "synthesis": "[enrichment failed: see stderr log for details]", "top_gap": "n/a"}
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not raw:
        # _call_groq already screens out truly-empty content -- this only
        # fires if the content was literally just markdown fences, e.g. "```json```".
        return {"cluster_count": None, "synthesis": "[enrichment failed: empty after stripping markdown fences]", "top_gap": "n/a"}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"cluster_count": None, "synthesis": raw[:1000], "top_gap": "unparsed"}


def enrich_run(client, topic: str, run_id: str, findings: list, created_at: str) -> dict:
    client.execute(ENRICH_SCHEMA)
    # best-effort migration for enrichment tables created before these columns existed
    for col, ddl in [("adversarial_n", "ALTER TABLE enrichment ADD COLUMN adversarial_n INTEGER"),
                      ("adversarial_check", "ALTER TABLE enrichment ADD COLUMN adversarial_check TEXT")]:
        try:
            client.execute(ddl)
        except Exception:
            pass  # column already exists

    # Order matters: the payload is truncated to 24000 chars before it goes to
    # the LLM (call_llm), and findings were previously sent in whatever order
    # they came out of the DB -- meaning adversarial/complaint evidence could
    # silently fall off the end of a large run and never reach the model at
    # all. Put it first so it's the last thing truncation would ever drop.
    adversarial_findings = [f for f in findings if f.has_adversarial_signal]
    complaint_findings = [f for f in findings if f.has_complaint_signal and not f.has_adversarial_signal]
    other_findings = [f for f in findings if not f.has_adversarial_signal and not f.has_complaint_signal]
    ordered = adversarial_findings + complaint_findings + other_findings

    payload = [
        {"pass_type": f.pass_type, "platform": f.platform, "domain": f.domain,
         "title": f.title, "snippet": f.snippet, "has_paid_signal": f.has_paid_signal,
         "has_adversarial_signal": f.has_adversarial_signal}
        for f in ordered
    ]
    result = call_llm(payload)

    # Post-hoc sanity check, both directions:
    # - n_adv > 0 but synthesis never engages with it -> evidence silently dropped.
    # - n_adv == 0 but synthesis uses the word "adversarial" anyway -> the model
    #   fabricated/overclaimed adversarial evidence that isn't actually in the
    #   input (seen in practice: it relabeled ordinary complaint-signal findings
    #   as "adversarial findings", which overstates them as maintainer/community
    #   pushback rather than "a user mentioned a bug").
    # This can't verify the synthesis is RIGHT either way, only that the
    # adversarial_n number and the synthesis text agree with each other.
    n_adv = len(adversarial_findings)
    synthesis_text = (result.get("synthesis") or "").lower()
    mentions_adversarial_word = "adversarial" in synthesis_text
    if n_adv == 0:
        if mentions_adversarial_word:
            adv_check = "OVERCLAIM: synthesis says \"adversarial\" but 0 adversarial findings existed in input"
            print(f"  [warn] enrichment overclaim for run {run_id}: {adv_check}", file=sys.stderr)
        else:
            adv_check = "n/a: none in input"
    elif mentions_adversarial_word or any(kw in synthesis_text for kw in
            ("pushback", "wontfix", "won't fix", "already exist", "not a problem",
             "not worth", "unnecessary", "overkill", "pointless", "gimmick")):
        adv_check = "ok"
    else:
        adv_check = f"MISSING: {n_adv} adversarial finding(s) in input but synthesis never addresses them"
        print(f"  [warn] enrichment adversarial check failed for run {run_id}: {adv_check}", file=sys.stderr)
        # Logging a warning isn't enough on its own -- it sits in a CI log
        # nobody re-reads, while the digest/enrichment table is what actually
        # gets consulted later. Deterministically append what the model
        # dropped, so the evidence survives in the record even when the LLM
        # doesn't comply with the "don't omit adversarial findings" instruction.
        titles = [f.title or f.url for f in adversarial_findings[:5]]
        appendix = "Adversarial evidence found but not reflected above: " + "; ".join(titles)
        if len(adversarial_findings) > 5:
            appendix += f" (+{len(adversarial_findings) - 5} more)"
        result["synthesis"] = (result.get("synthesis") or "").rstrip() + "\n\n" + appendix

    client.execute(
        """INSERT OR REPLACE INTO enrichment
           (run_id, topic, synthesis, cluster_count, top_gap, adversarial_n, adversarial_check, model_used, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [run_id, topic, result.get("synthesis", ""), result.get("cluster_count"),
         result.get("top_gap", ""), n_adv, adv_check,
         LLM_MODEL if GROQ_API_KEY else "none", created_at],
    )
    result["adversarial_n"] = n_adv
    result["adversarial_check"] = adv_check
    return result
