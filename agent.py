"""
research_agent.py
Free-forever multi-pass web research agent for product R&D (e.g. Blender add-ons).

Loosely mirrors the pmf-engine method (confirming / independence / transactional /
adversarial / segment-fit passes) but runs unattended, structures every hit, and
writes to Turso so results accumulate across many runs instead of living only in
one chat window.

Sourcing follows a legitimacy rule: official API first, always (GitHub Search,
Blender's devtalk Discourse JSON API, HN Algolia -- see sources.py). DuckDuckGo
web search fills the general-web gap and is explicitly tagged tier=2 "scrape" --
it has no clean ToS-compliant path, so treat it as the weakest-legitimacy source
in the mix, not the default one.

Cost: $0. DB = Turso free tier. No LLM call is made in the main path on purpose
(see enrich.py for the optional, clearly-separated LLM enrichment step) --
that's what keeps "24/7 forever" actually free.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Iterable

from ddgs import DDGS
import libsql_client

import sources

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TURSO_URL = os.environ.get("TURSO_DATABASE_URL")       # e.g. libsql://your-db-org.turso.io
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

# Repos genuinely hosted on GitHub -- NOT blender/blender (see sources.py note)
GITHUB_REPOS = ["KhronosGroup/glTF-Blender-IO", "PixarAnimationStudios/OpenUSD", "alembic/alembic"]

PLATFORM_HINTS = {
    "reddit.com": "forum",
    "blenderartists.org": "forum",
    "devtalk.blender.org": "forum",
    "github.com": "issue_tracker",
    "gitlab.com": "issue_tracker",
    "gumroad.com": "marketplace",
    "blendermarket.com": "marketplace",
    "itch.io": "marketplace",
    "artstation.com": "portfolio",
    "youtube.com": "video",
}

PAID_SIGNAL_RE = re.compile(r"\$\d+|\€\d+|£\d+|\bprice\b|\bbuy\b|\bpurchase\b", re.I)
COMPLAINT_RE = re.compile(
    r"\b(wish|annoying|frustrat|workaround|hacky|no way to|can'?t figure|pain point|"
    r"tedious|manual(ly)?|slow to|time.?consuming)\b", re.I
)
ADVERSARIAL_RE = re.compile(
    r"\b(already (solved|exists|does this)|not (a|really a) problem|use .* instead|"
    r"non.?issue|works fine|don'?t need)\b", re.I
)

# The 5 pass types, matched to pmf-engine's method
PASS_TEMPLATES = {
    "confirming": [
        "{topic} blender addon",
        "{topic} blender feature request",
        "{topic} blenderartists",
    ],
    "transactional": [
        "{topic} blender addon buy OR price site:gumroad.com",
        "{topic} blender addon site:blendermarket.com",
    ],
    "independence_platform": [
        "{topic} blender github issue",
        "{topic} blender reddit",
    ],
    "adversarial": [
        "{topic} blender addon \"already exists\" OR \"not a problem\" OR \"use instead\"",
    ],
    "segment_fit": [
        "{topic} blender addon architects OR \"game dev\" OR \"archviz\"",
    ],
}


@dataclass
class Finding:
    run_id: str
    pass_type: str
    query: str
    url: str
    title: str
    snippet: str
    platform: str
    domain: str
    has_paid_signal: bool
    has_complaint_signal: bool
    has_adversarial_signal: bool
    fetched_at: str
    content_hash: str
    tier: int = 2                # 1 = official on-topic API, 2 = general web/scrape
    engagement_signal: int = 0   # reactions/points/likes -- 0 if source doesn't provide one


def classify_domain(url: str) -> tuple[str, str]:
    m = re.search(r"https?://(?:www\.)?([^/]+)/?", url or "")
    domain = m.group(1) if m else "unknown"
    platform = next((v for k, v in PLATFORM_HINTS.items() if k in domain), "other")
    return domain, platform


def to_finding(run_id: str, pass_type: str, query: str, r: dict) -> Finding:
    """DuckDuckGo result -> Finding. Always tier=2: this is the scrape path,
    explicitly the weakest-legitimacy source (see module docstring)."""
    title = r.get("title", "") or ""
    body = r.get("body", "") or ""
    url = r.get("href", "") or r.get("url", "") or ""
    domain, platform = classify_domain(url)
    blob = f"{title} {body}"
    return Finding(
        run_id=run_id,
        pass_type=pass_type,
        query=query,
        url=url,
        title=title.strip(),
        snippet=body.strip()[:500],
        platform=platform,
        domain=domain,
        has_paid_signal=bool(PAID_SIGNAL_RE.search(blob)),
        has_complaint_signal=bool(COMPLAINT_RE.search(blob)),
        has_adversarial_signal=bool(ADVERSARIAL_RE.search(blob)),
        fetched_at=dt.datetime.utcnow().isoformat(),
        content_hash=hashlib.sha256(blob.encode()).hexdigest()[:16],
        tier=2,
    )


def to_finding_from_source(run_id: str, pass_type: str, query: str, r: dict) -> Finding:
    """sources.py result (official API) -> Finding. Carries real tier + engagement."""
    blob = f"{r.get('title','')} {r.get('snippet','')}"
    return Finding(
        run_id=run_id,
        pass_type=pass_type,
        query=query,
        url=r.get("url", ""),
        title=(r.get("title") or "").strip(),
        snippet=(r.get("snippet") or "").strip()[:500],
        platform=r.get("platform", "other"),
        domain=r.get("domain", "unknown"),
        has_paid_signal=bool(PAID_SIGNAL_RE.search(blob)),
        has_complaint_signal=bool(COMPLAINT_RE.search(blob)),
        has_adversarial_signal=bool(ADVERSARIAL_RE.search(blob)),
        fetched_at=dt.datetime.utcnow().isoformat(),
        content_hash=hashlib.sha256(blob.encode()).hexdigest()[:16],
        tier=r.get("tier", 1),
        engagement_signal=r.get("engagement_signal", 0),
    )


def run_pass(topic: str, pass_type: str, run_id: str, max_results: int = 8) -> list[Finding]:
    """Tier 2: general-web via DuckDuckGo scrape."""
    findings = []
    with DDGS() as ddgs:
        for tmpl in PASS_TEMPLATES[pass_type]:
            query = tmpl.format(topic=topic)
            try:
                for r in ddgs.text(query, max_results=max_results):
                    findings.append(to_finding(run_id, pass_type, query, r))
            except Exception as e:
                print(f"  [warn] query failed: {query!r} -> {e}", file=sys.stderr)
            time.sleep(1.5)  # be polite, avoid rate-limit bans -- keeps this free long-term
    return findings


def run_official_sources(topic: str, run_id: str, max_results: int = 8) -> list[Finding]:
    """Tier 1 (+ tier 2 HN). Official, keyless APIs -- the legitimate-first path."""
    findings = []
    for r in sources.search_github_issues(topic, GITHUB_REPOS, max_results):
        findings.append(to_finding_from_source(run_id, "official_github", topic, r))
    time.sleep(1)
    for r in sources.search_devtalk(topic, max_results):
        findings.append(to_finding_from_source(run_id, "official_devtalk", topic, r))
    time.sleep(1)
    for r in sources.search_hn(topic, max_results):
        findings.append(to_finding_from_source(run_id, "official_hn", topic, r))
    return findings


# ---------------------------------------------------------------------------
# Turso persistence
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, topic TEXT, pass_type TEXT, query TEXT,
    url TEXT, title TEXT, snippet TEXT, platform TEXT, domain TEXT,
    has_paid_signal INTEGER, has_complaint_signal INTEGER, has_adversarial_signal INTEGER,
    fetched_at TEXT, content_hash TEXT, tier INTEGER DEFAULT 2, engagement_signal INTEGER DEFAULT 0,
    UNIQUE(topic, content_hash)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, topic TEXT, started_at TEXT, finished_at TEXT,
    n_findings INTEGER, n_distinct_domains INTEGER, n_new INTEGER DEFAULT 0
);
"""


def get_client():
    if not (TURSO_URL and TURSO_TOKEN):
        return None
    return libsql_client.create_client_sync(url=TURSO_URL, auth_token=TURSO_TOKEN)


def ensure_schema(client):
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            client.execute(stmt)
    # best-effort migration for DBs created before tier/engagement_signal existed
    for col, ddl in [("tier", "ALTER TABLE findings ADD COLUMN tier INTEGER DEFAULT 2"),
                      ("engagement_signal", "ALTER TABLE findings ADD COLUMN engagement_signal INTEGER DEFAULT 0")]:
        try:
            client.execute(ddl)
        except Exception:
            pass  # column already exists


def get_known_hashes(client, topic: str) -> set[str]:
    """All content_hashes already stored for this topic, from any prior run.
    Used to compute what's genuinely new before this run's insert happens."""
    rs = client.execute("SELECT content_hash FROM findings WHERE topic = ?", [topic])
    return {row[0] for row in rs.rows}


def persist(client, topic: str, run_id: str, findings: list[Finding], n_new: int):
    ensure_schema(client)
    for f in findings:
        d = asdict(f)
        try:
            client.execute(
                """INSERT OR IGNORE INTO findings
                   (run_id, topic, pass_type, query, url, title, snippet, platform, domain,
                    has_paid_signal, has_complaint_signal, has_adversarial_signal, fetched_at,
                    content_hash, tier, engagement_signal)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [d["run_id"], topic, d["pass_type"], d["query"], d["url"], d["title"], d["snippet"],
                 d["platform"], d["domain"], int(d["has_paid_signal"]), int(d["has_complaint_signal"]),
                 int(d["has_adversarial_signal"]), d["fetched_at"], d["content_hash"], d["tier"],
                 d["engagement_signal"]],
            )
        except Exception as e:
            print(f"  [warn] insert failed: {e}", file=sys.stderr)
    domains = len({f.domain for f in findings})
    client.execute(
        """INSERT OR REPLACE INTO runs (run_id, topic, started_at, finished_at, n_findings, n_distinct_domains, n_new)
           VALUES (?,?,?,?,?,?,?)""",
        [run_id, topic, findings[0].fetched_at if findings else "", dt.datetime.utcnow().isoformat(),
         len(findings), domains, n_new],
    )


# ---------------------------------------------------------------------------
# Dense digest for Claude
# ---------------------------------------------------------------------------

def build_digest(topic: str, run_id: str, findings: list[Finding],
                  known_hashes: set[str] | None = None, is_first_run: bool = True) -> str:
    """known_hashes = hashes that existed BEFORE this run (see get_known_hashes,
    called prior to persist). Anything in `findings` not in known_hashes is new
    this cycle -- that's what gets shown in full; repeats get a one-line count
    only, so a run 20 cycles in doesn't re-dump the same 6 gumroad listings."""
    known_hashes = known_hashes or set()
    new_findings = findings if is_first_run else [f for f in findings if f.content_hash not in known_hashes]
    carryover_count = len(findings) - len(new_findings)

    by_pass: dict[str, list[Finding]] = {}
    for f in new_findings:
        by_pass.setdefault(f.pass_type, []).append(f)

    domains = sorted({f.domain for f in findings})
    paid = [f for f in new_findings if f.has_paid_signal]
    complaints = [f for f in new_findings if f.has_complaint_signal]
    adversarial = [f for f in new_findings if f.has_adversarial_signal]
    tier1 = [f for f in new_findings if f.tier == 1]
    tier2 = [f for f in new_findings if f.tier == 2]

    lines = [
        f"# Research Digest — {topic}",
        f"run_id: {run_id} | total this run: {len(findings)} | **new since last run: {len(new_findings)}** | "
        f"already seen (suppressed): {carryover_count} | distinct domains (all-time context): {len(domains)}",
        f"tier 1 (official API) new: {len(tier1)} | tier 2 (general web) new: {len(tier2)}",
        f"paid/marketplace: {len(paid)} | complaints: {len(complaints)} | adversarial: {len(adversarial)}",
        "",
    ]
    if is_first_run:
        lines.insert(1, "_(first run for this topic — nothing to diff against, showing everything)_")
    elif not new_findings:
        lines.append("_No new findings since last run — market signal unchanged this cycle._")
        lines.append("")

    for pass_type, items in by_pass.items():
        lines.append(f"## {pass_type} ({len(items)} new)")
        # tier 1 first -- a single tier-1 finding outweighs several tier-2 mentions
        for f in sorted(items, key=lambda x: (x.tier, -x.engagement_signal))[:15]:
            flags = "".join([
                "$" if f.has_paid_signal else "",
                "!" if f.has_complaint_signal else "",
                "~" if f.has_adversarial_signal else "",
            ])
            eng = f" (engagement:{f.engagement_signal})" if f.engagement_signal else ""
            lines.append(f"- [T{f.tier}][{f.platform}/{f.domain}] {f.title} {('['+flags+']') if flags else ''}{eng} — {f.url}")
        lines.append("")

    lines.append("## Gate signal (pmf-engine style, mechanical read — verify before trusting)")
    lines.append(f"- source diversity: {len(domains)} domains ({'OK 3+' if len(domains) >= 3 else 'THIN'})")
    lines.append(f"- transactional evidence (new): {len(paid)} hits")
    lines.append(f"- adversarial pass (new): {'completed, ' + str(len(adversarial)) + ' hits' if by_pass.get('adversarial') or by_pass.get('official_github') else 'check raw log'}")
    lines.append(f"- tier1/tier2 disagreement: review manually if tier-1 and tier-2 findings point opposite directions — that's signal, not noise to average away")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", help="Product idea, e.g. 'procedural building generator'")
    ap.add_argument("--max-results", type=int, default=8)
    ap.add_argument("--no-db", action="store_true", help="skip Turso, just print digest (no diff possible)")
    ap.add_argument("--enrich", action="store_true", help="run optional free-LLM enrichment (needs GROQ_API_KEY, needs --no-db off)")
    ap.add_argument("--skip-scrape", action="store_true", help="official sources only, skip the tier-2 DuckDuckGo pass entirely")
    args = ap.parse_args()

    run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    all_findings: list[Finding] = []

    print("[pass] official sources (tier 1/2, keyless APIs) ...", file=sys.stderr)
    all_findings.extend(run_official_sources(args.topic, run_id, args.max_results))

    if not args.skip_scrape:
        for pass_type in PASS_TEMPLATES:
            print(f"[pass] {pass_type} (tier 2, ddg scrape) ...", file=sys.stderr)
            all_findings.extend(run_pass(args.topic, pass_type, run_id, args.max_results))

    enrichment = None
    known_hashes: set[str] = set()
    is_first_run = True

    if not args.no_db:
        client = get_client()
        if client:
            ensure_schema(client)
            known_hashes = get_known_hashes(client, args.topic)   # BEFORE insert -- this is the diff baseline
            is_first_run = len(known_hashes) == 0
            new_findings = [f for f in all_findings if f.content_hash not in known_hashes]

            persist(client, args.topic, run_id, all_findings, n_new=len(new_findings))
            print(f"[db] wrote {len(all_findings)} findings ({len(new_findings)} new) to Turso", file=sys.stderr)

            if args.enrich and new_findings:
                from enrich import enrich_run
                enrichment = enrich_run(client, args.topic, run_id, new_findings,
                                         dt.datetime.utcnow().isoformat())
                print("[llm] enrichment written to Turso", file=sys.stderr)
        else:
            print("[db] TURSO_DATABASE_URL / TURSO_AUTH_TOKEN not set — skipping persistence (no diff without a DB)", file=sys.stderr)

    digest = build_digest(args.topic, run_id, all_findings, known_hashes, is_first_run)
    if enrichment:
        digest = (
            f"## LLM synthesis of NEW findings (deduped clusters: {enrichment.get('cluster_count')})\n"
            f"{enrichment.get('synthesis')}\n\n"
            f"**Top gap:** {enrichment.get('top_gap')}\n\n"
            + digest
        )
    print(digest)


if __name__ == "__main__":
    main()
