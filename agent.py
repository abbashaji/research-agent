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

# ddgs (the installed lib, v9.15+) is actually a metasearch aggregator, not
# just a DuckDuckGo scraper -- it can fall back across several real engines.
# We observed a run where DuckDuckGo alone returned near-random/trending junk
# instead of query-matched results (no exception raised, so retry-on-error
# never caught it) -- listing multiple engines here means a bad day for any
# one of them doesn't take out the whole pass. wikipedia/grokipedia are
# deliberately excluded (encyclopedic, not useful for finding forum/
# marketplace/issue-tracker content). Override via env var if you want to
# pin to one engine for debugging (e.g. DDGS_BACKEND=duckduckgo).
DDGS_BACKEND = os.environ.get("DDGS_BACKEND", "duckduckgo,brave,mojeek,startpage,yahoo")

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
# Broadened beyond the original 3 phrases -- covers issue-tracker rejection
# language ("wontfix", "not planned") and "just do X manually instead"
# redirection, which are real shapes adversarial pushback takes. Deliberately
# does NOT include bare single words like "overkill"/"pointless"/"gimmick"/
# "unnecessary" anymore -- those were tried and produced false positives on
# completely unrelated content (a Discord server literally named "Overkill",
# a Patreon artist, a Raid Shadow Legends comment) whenever DDG's scrape
# returned loosely-matched or off-topic junk. Multi-word phrases below are
# much less likely to appear by coincidence in unrelated text.
ADVERSARIAL_RE = re.compile(
    r"\b(already (solved|exists|does this)|not (a|really a) problem|use .* instead|"
    r"non.?issue|works fine|don'?t need|won'?t ?fix|wont ?fix|not planned|"
    r"such a gimmick|complete(ly)? overkill|totally unnecessary|"
    r"just (do|use|export) .* manually|"
    r"not worth (it|building)|reinvent(ing)? the wheel)\b", re.I
)

# The 5 pass types, matched to pmf-engine's method.
#
# adversarial and segment_fit deliberately get MORE templates than the others
# now, not fewer -- they used to have just 1 query each, which structurally
# starved them relative to confirming/transactional (3 and 2 templates), so a
# run could look like strong market signal purely because the passes most
# likely to surface reasons NOT to build something had the smallest sample
# size going in, not because the pushback wasn't there.
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
        "{topic} blender addon \"wontfix\" OR \"won't fix\" OR \"not planned\"",
        "{topic} blender addon \"pointless\" OR \"overkill\" OR \"gimmick\"",
        "{topic} blender \"just do it manually\" OR \"not worth building\"",
    ],
    "segment_fit": [
        "{topic} blender addon architects OR \"game dev\" OR \"archviz\"",
        # Was "... OR hobbyist OR student" -- "student" alone is exactly the
        # kind of bare generic word that broke adversarial earlier (see
        # ADVERSARIAL_RE comment above): it pulled back student-portal LOGIN
        # PAGES (a university SSO page, ClassLink, a Microsoft login screen)
        # with zero relation to Blender. Compound phrases anchor the engine
        # to the actual intended sense instead of matching the bare word
        # anywhere on the page.
        "{topic} blender addon \"indie developer\" OR \"student project\" OR hobbyist",
    ],
}

# Query fallback rewrites, tried in order when a template comes back with
# zero results. DDG's scrape surface is fragile and degrades unpredictably --
# a query with quoted phrases and site: filters is the most likely to whiff,
# so broaden progressively rather than just logging a warning and losing that
# pass_type's coverage for the whole cycle.
def _broadened_variants(query: str) -> list[str]:
    variants = []
    # 1. Drop site: filters -- these are the single biggest source of
    #    "No results found" since DDG's index of any one site is incomplete.
    no_site = re.sub(r"\s*site:\S+", "", query).strip()
    if no_site != query:
        variants.append(no_site)
    # 2. Drop quoted exact phrases, keep the OR'd bare words -- exact-phrase
    #    matching is brittle against paraphrased real-world text.
    no_quotes = re.sub(r'"([^"]*)"', r"\1", no_site or query)
    if no_quotes not in (query, no_site):
        variants.append(no_quotes)
    return variants


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


_RELEVANCE_TERMS = ("blender", "addon", "add-on", "geometry node", "plugin")


def _is_topic_relevant(blob: str) -> bool:
    """DDG scrape (tier 2) has no query-side relevance filter the way GitHub's
    search API or a site: filter does -- when a query gets broadened (site:
    dropped, phrases unquoted) to recover from a 0-result miss, it can widen
    enough to pull back completely unrelated pages that happen to contain a
    matched word (e.g. a Discord server literally named "Overkill", nothing
    to do with Blender). Require at least one real domain term to actually
    appear before trusting a paid/complaint/adversarial signal match from
    this path -- official-API findings (to_finding_from_source) don't need
    this, they're already scoped by the query itself (repo filter, devtalk/hn
    search)."""
    b = blob.lower()
    return any(term in b for term in _RELEVANCE_TERMS)


def to_finding(run_id: str, pass_type: str, query: str, r: dict) -> Finding:
    """DuckDuckGo result -> Finding. Always tier=2: this is the scrape path,
    explicitly the weakest-legitimacy source (see module docstring)."""
    title = r.get("title", "") or ""
    body = r.get("body", "") or ""
    url = r.get("href", "") or r.get("url", "") or ""
    domain, platform = classify_domain(url)
    blob = f"{title} {body}"
    relevant = _is_topic_relevant(blob)
    return Finding(
        run_id=run_id,
        pass_type=pass_type,
        query=query,
        url=url,
        title=title.strip(),
        snippet=body.strip()[:500],
        platform=platform,
        domain=domain,
        has_paid_signal=relevant and bool(PAID_SIGNAL_RE.search(blob)),
        has_complaint_signal=relevant and bool(COMPLAINT_RE.search(blob)),
        has_adversarial_signal=relevant and bool(ADVERSARIAL_RE.search(blob)),
        fetched_at=dt.datetime.now(dt.UTC).isoformat(),
        content_hash=hashlib.sha256(blob.encode()).hexdigest()[:16],
        tier=2,
    )


def to_finding_from_source(run_id: str, pass_type: str, query: str, r: dict) -> Finding:
    """sources.py result (official API) -> Finding. Carries real tier + engagement."""
    blob = f"{r.get('title','')} {r.get('snippet','')}"
    # A closed+wontfix GitHub issue IS adversarial signal by construction --
    # a maintainer already made that call -- regardless of whether the ADVERSARIAL_RE
    # regex happens to match the issue body text. Don't make this depend on wording.
    forced_adversarial = r.get("source") == "github_issues_wontfix"
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
        has_adversarial_signal=forced_adversarial or bool(ADVERSARIAL_RE.search(blob)),
        fetched_at=dt.datetime.now(dt.UTC).isoformat(),
        content_hash=hashlib.sha256(blob.encode()).hexdigest()[:16],
        tier=r.get("tier", 1),
        engagement_signal=r.get("engagement_signal", 0),
    )


def _ddg_search(ddgs: DDGS, query: str, max_results: int, retries: int = 2) -> list[dict]:
    """Multi-engine metasearch via ddgs, with a short retry-with-backoff for
    transient failures (rate limits, momentary markup changes) on top.
    Distinguishes "0 results" (not an error -- log at info level, try a
    broadened variant) from an actual exception (retry, then give up and warn).

    Uses DDGS_BACKEND (several engines) rather than DuckDuckGo alone, since
    we've observed DuckDuckGo return near-random/trending junk with a 200
    response (no exception) rather than actually failing -- retry logic can't
    catch that, but a different engine in the list is less likely to be
    degraded on the same day for the same reason."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            results = list(ddgs.text(query, max_results=max_results, backend=DDGS_BACKEND))
            return results
        except Exception as e:
            last_exc = e
            if attempt < retries:
                backoff = 2 * (attempt + 1)
                print(f"  [retry] {query!r} failed ({e}), retrying in {backoff}s "
                      f"({attempt + 1}/{retries})...", file=sys.stderr)
                time.sleep(backoff)
    print(f"  [warn] query failed after {retries + 1} attempts: {query!r} -> {last_exc}", file=sys.stderr)
    return []


def run_pass(topic: str, pass_type: str, run_id: str, max_results: int = 8) -> list[Finding]:
    """Tier 2: general-web via DuckDuckGo scrape.

    adversarial gets a max_results boost -- it's the pass most likely to
    surface reasons NOT to build something, and with only 1-4 templates vs.
    confirming's higher query-diversity, it needs more results per query to
    end up with a comparable sample, not a structurally thinner one."""
    findings = []
    pass_max_results = max_results * 2 if pass_type == "adversarial" else max_results
    with DDGS() as ddgs:
        for tmpl in PASS_TEMPLATES[pass_type]:
            query = tmpl.format(topic=topic)
            results = _ddg_search(ddgs, query, pass_max_results)
            # Broadening trades exact-phrase precision for recall by dropping
            # quotes/site: filters -- fine for confirming/transactional/etc,
            # where the templates are already loose keyword combinations. But
            # adversarial's templates ARE the precision (quoted rejection
            # phrases like "wontfix", "already exists") -- loosening those
            # into a bag-of-words query invites off-topic noise from whichever
            # engine handles long OR-heavy unquoted queries worst, rather than
            # recovering a real miss. A real 0 here is legitimate signal (no
            # adversarial chatter found), not a fetch failure to work around;
            # the tier-1 GitHub wontfix source already covers this pass's
            # precision-matching job without that tradeoff.
            if not results and pass_type != "adversarial":
                for variant in _broadened_variants(query):
                    print(f"  [broaden] no results for {query!r}, trying {variant!r}", file=sys.stderr)
                    results = _ddg_search(ddgs, variant, pass_max_results, retries=1)
                    if results:
                        query = variant  # record which query actually produced these
                        break
            if not results:
                print(f"  [info] no results found (incl. broadened variants): {tmpl.format(topic=topic)!r}",
                      file=sys.stderr)
            for r in results:
                findings.append(to_finding(run_id, pass_type, query, r))
            time.sleep(1.5)  # be polite, avoid rate-limit bans -- keeps this free long-term
    return findings


_STOPWORDS = frozenset("""
a an the of for and or to in on with that this need needs needing wants wanting
who which practitioners studios like doing bespoke such as into first class not
just a the be become inside native system practitioners boutique
""".split())


def heuristic_shorten_topic(topic: str, max_words: int = 8) -> str:
    """Zero-cost, no-LLM fallback for shortening a long descriptive topic into
    something closer to a search query: drop parenthetical asides (usually
    examples/context, not core terms), strip stopwords, keep the first
    max_words significant tokens in original order. Not as good as an LLM
    read of what actually matters, but free and always available."""
    no_parens = re.sub(r"\([^)]*\)", " ", topic)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", no_parens)
    kept = [w for w in words if w.lower() not in _STOPWORDS]
    return " ".join(kept[:max_words]) if kept else topic[:60]


def derive_search_topic(topic: str, use_llm: bool, length_threshold: int = 60) -> str:
    """Long, descriptive topics (a full sentence describing a target user/use
    case) break the narrower query templates -- adversarial phrase matching
    and GitHub's label:wontfix search need concrete search terms, not a
    paragraph. Short topics pass through unchanged (existing behavior,
    zero-cost). Long topics get shortened either by a cheap LLM call (if
    --enrich is on, since that already requires GROQ_API_KEY -- keeping the
    "no LLM in the main path unless you opted in" rule from the top of this
    file) or a free heuristic fallback otherwise, and the LLM path also falls
    back to the heuristic on any failure rather than searching the raw
    paragraph."""
    if len(topic) <= length_threshold:
        return topic
    if use_llm:
        try:
            from enrich import extract_search_keywords
            keywords = extract_search_keywords(topic)
            if keywords:
                print(f"  [keywords] LLM-shortened topic for search: {keywords!r}", file=sys.stderr)
                return keywords
        except Exception as e:
            print(f"  [warn] keyword extraction failed, using heuristic fallback: {e}", file=sys.stderr)
    shortened = heuristic_shorten_topic(topic)
    print(f"  [keywords] heuristic-shortened topic for search: {shortened!r}", file=sys.stderr)
    return shortened


def run_official_sources(topic: str, run_id: str, max_results: int = 8) -> list[Finding]:
    """Tier 1 (+ tier 2 HN). Official, keyless APIs -- the legitimate-first path."""
    findings = []
    for r in sources.search_github_issues(topic, GITHUB_REPOS, max_results):
        findings.append(to_finding_from_source(run_id, "official_github", topic, r))
    time.sleep(1)
    # Tier-1 adversarial: closed+wontfix issues are a maintainer explicitly
    # declining to build/merge something -- stronger signal than any DDG
    # scrape hit, and pass_type="adversarial" so it counts toward the same
    # gate check as the tier-2 adversarial pass instead of being invisible to it.
    for r in sources.search_github_issues_adversarial(topic, GITHUB_REPOS, max_results):
        findings.append(to_finding_from_source(run_id, "adversarial", topic, r))
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
    n_findings INTEGER, n_distinct_domains INTEGER, n_new INTEGER DEFAULT 0,
    search_keywords TEXT  -- the actual short query text used, if topic was auto-shortened
);
"""


def get_client():
    if not (TURSO_URL and TURSO_TOKEN):
        return None
    # libsql-client's WebSocket transport (libsql:// or wss://) fails a
    # handshake against some Turso replicas (seen on GitHub Actions runners --
    # aiohttp.WSServerHandshakeError: 400). The HTTP transport (https://) talks
    # to the same database over plain HTTPS instead and sidesteps this
    # entirely, so normalize the scheme regardless of what was pasted from the
    # Turso dashboard (which shows libsql:// by default).
    url = TURSO_URL.replace("libsql://", "https://").replace("wss://", "https://")
    return libsql_client.create_client_sync(url=url, auth_token=TURSO_TOKEN)


def ensure_schema(client):
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            client.execute(stmt)
    # best-effort migration for DBs created before tier/engagement_signal existed
    for col, ddl in [("tier", "ALTER TABLE findings ADD COLUMN tier INTEGER DEFAULT 2"),
                      ("engagement_signal", "ALTER TABLE findings ADD COLUMN engagement_signal INTEGER DEFAULT 0"),
                      ("search_keywords", "ALTER TABLE runs ADD COLUMN search_keywords TEXT")]:
        try:
            client.execute(ddl)
        except Exception:
            pass  # column already exists


def get_known_hashes(client, topic: str) -> set[str]:
    """All content_hashes already stored for this topic, from any prior run.
    Used to compute what's genuinely new before this run's insert happens."""
    rs = client.execute("SELECT content_hash FROM findings WHERE topic = ?", [topic])
    return {row[0] for row in rs.rows}


def persist(client, topic: str, run_id: str, findings: list[Finding], n_new: int, search_keywords: str = ""):
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
    # findings has UNIQUE(topic, content_hash), so INSERT OR IGNORE silently
    # drops same-run duplicates (overlapping query templates often surface the
    # same URL). len(findings) is the pre-dedup count gathered in memory, not
    # what actually landed in the table -- query the real number back out
    # rather than trust the in-memory count, so n_findings/n_new in `runs`
    # reflect what's actually queryable afterward, not an inflated estimate.
    rs = client.execute("SELECT COUNT(*) FROM findings WHERE run_id = ?", [run_id])
    actual_rows = rs.rows[0][0]
    if actual_rows != len(findings):
        print(f"  [info] {len(findings)} findings gathered, {actual_rows} actually new "
              f"(within-run duplicates across overlapping query templates were dropped)", file=sys.stderr)
    # n_new was computed by the caller against known_hashes from BEFORE this
    # run started, so it can also overcount for the same reason -- cap it at
    # what's actually in the table now for this run_id.
    n_new_actual = min(n_new, actual_rows)
    client.execute(
        """INSERT OR REPLACE INTO runs (run_id, topic, started_at, finished_at, n_findings, n_distinct_domains, n_new, search_keywords)
           VALUES (?,?,?,?,?,?,?,?)""",
        [run_id, topic, findings[0].fetched_at if findings else "", dt.datetime.now(dt.UTC).isoformat(),
         actual_rows, domains, n_new_actual, search_keywords if search_keywords != topic else ""],
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
    adversarial_ran = "adversarial" in {f.pass_type for f in findings}  # true if the pass_type appears anywhere this run, new or not
    if adversarial_ran:
        adv_line = f"completed, {len(adversarial)} new hit(s)" if adversarial else "completed, 0 new hits (no adversarial evidence found this cycle -- that's a real result, not a failure)"
    else:
        adv_line = "did not run or produced nothing at all -- check raw log for [warn]/[info] lines"
    lines.append(f"- adversarial pass (new): {adv_line}")
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

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    all_findings: list[Finding] = []

    # Query templates use search_topic (short); the `findings`/`runs` tables
    # keep args.topic (the full original description) as the identity key, so
    # topic history/dedup grouping is unaffected -- only what actually gets
    # typed into GitHub/DDG search changes.
    search_topic = derive_search_topic(args.topic, use_llm=args.enrich)

    print("[pass] official sources (tier 1/2, keyless APIs) ...", file=sys.stderr)
    all_findings.extend(run_official_sources(search_topic, run_id, args.max_results))

    if not args.skip_scrape:
        for pass_type in PASS_TEMPLATES:
            print(f"[pass] {pass_type} (tier 2, ddg scrape) ...", file=sys.stderr)
            all_findings.extend(run_pass(search_topic, pass_type, run_id, args.max_results))

    enrichment = None
    known_hashes: set[str] = set()
    is_first_run = True

    if not args.no_db:
        client = get_client()
        if client:
            try:
                ensure_schema(client)
                known_hashes = get_known_hashes(client, args.topic)   # BEFORE insert -- this is the diff baseline
                is_first_run = len(known_hashes) == 0
                new_findings = [f for f in all_findings if f.content_hash not in known_hashes]

                persist(client, args.topic, run_id, all_findings, n_new=len(new_findings), search_keywords=search_topic)
                print(f"[db] wrote {len(all_findings)} findings ({len(new_findings)} new) to Turso", file=sys.stderr)

                if args.enrich and new_findings:
                    from enrich import enrich_run
                    enrichment = enrich_run(client, args.topic, run_id, new_findings,
                                             dt.datetime.now(dt.UTC).isoformat())
                    print("[llm] enrichment written to Turso", file=sys.stderr)
            finally:
                # libsql_client's sync wrapper runs a background thread with its
                # own event loop -- if this isn't closed, the thread keeps the
                # whole process alive and the script (and the CI step) never
                # exits, even after all work is done.
                client.close()
        else:
            print("[db] TURSO_DATABASE_URL / TURSO_AUTH_TOKEN not set — skipping persistence (no diff without a DB)", file=sys.stderr)

    digest = build_digest(args.topic, run_id, all_findings, known_hashes, is_first_run)
    if enrichment:
        adv_check = enrichment.get("adversarial_check", "")
        adv_line = f"**Adversarial check:** {adv_check}\n\n" if adv_check and adv_check != "n/a: none in input" else ""
        digest = (
            f"## LLM synthesis of NEW findings (deduped clusters: {enrichment.get('cluster_count')})\n"
            f"{enrichment.get('synthesis')}\n\n"
            f"**Top gap:** {enrichment.get('top_gap')}\n\n"
            f"{adv_line}"
            + digest
        )
    print(digest)


if __name__ == "__main__":
    main()
