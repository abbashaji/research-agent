"""
sources.py
Official, keyless, ToS-legitimate APIs -- the "Tier 1/2" sources.

Per the legitimacy rule: official API or RSS first, always. Scraping search
engine result pages has no clean ToS-compliant path, so it stays only as an
explicit, separately-tagged tier-2 fallback in agent.py (DuckDuckGo), not the
primary source.

Each function degrades on failure/quota exhaustion by returning [] and
logging -- it never raises past the caller and never falls back to scraping
to "compensate." A thin cycle is fine; a cycle that silently escalated to
cover a gap is not.
"""

import json
import sys
import time
import urllib.parse
import urllib.request


def _get_json(url: str, headers: dict | None = None, timeout: int = 15):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "rnd-research-agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def search_github_issues(query: str, repos: list[str], max_results: int = 8) -> list[dict]:
    """Tier 1. GitHub Search API, unauthenticated: 10 req/min. Reaction totals
    are a quantified pain-point signal -- log them, don't discard them.

    NOTE: Blender's own core bug tracker is NOT on GitHub -- it moved to
    projects.blender.org (self-hosted Gitea) years ago; blender/blender on
    GitHub is a read-only mirror with issues disabled. Point this at repos
    that are genuinely GitHub-hosted, e.g. KhronosGroup/glTF-Blender-IO,
    PixarAnimationStudios/OpenUSD, alembic/alembic. For Blender core itself,
    use the projects.blender.org Gitea API instead (not yet implemented here
    -- same request shape as devtalk below, see TODO at bottom of file)."""
    out = []
    repo_filter = " ".join(f"repo:{r}" for r in repos)
    q = urllib.parse.quote(f"{query} {repo_filter}")
    try:
        data = _get_json(f"https://api.github.com/search/issues?q={q}&per_page={max_results}",
                          headers={"Accept": "application/vnd.github+json", "User-Agent": "rnd-research-agent/1.0"})
        for item in data.get("items", []):
            out.append({
                "source": "github_issues", "tier": 1,
                "title": item.get("title", ""),
                "snippet": (item.get("body") or "")[:500],
                "url": item.get("html_url", ""),
                "domain": "github.com",
                "platform": "issue_tracker",
                "engagement_signal": item.get("reactions", {}).get("total_count", 0),
            })
    except Exception as e:
        print(f"  [warn] github_issues quota/error, skipping: {e}", file=sys.stderr)
    return out


def search_devtalk(query: str, max_results: int = 8) -> list[dict]:
    """Tier 1. devtalk.blender.org Discourse JSON API, no auth required."""
    out = []
    q = urllib.parse.quote(query)
    try:
        data = _get_json(f"https://devtalk.blender.org/search.json?q={q}")
        for topic in data.get("topics", [])[:max_results]:
            out.append({
                "source": "devtalk", "tier": 1,
                "title": topic.get("title", ""),
                "snippet": "",  # Discourse search doesn't return body; title is the signal
                "url": f"https://devtalk.blender.org/t/{topic.get('slug','')}/{topic.get('id','')}",
                "domain": "devtalk.blender.org",
                "platform": "forum",
                "engagement_signal": topic.get("reply_count", 0) + topic.get("like_count", 0),
            })
    except Exception as e:
        print(f"  [warn] devtalk quota/error, skipping: {e}", file=sys.stderr)
    return out


def search_hn(query: str, max_results: int = 8) -> list[dict]:
    """Tier 2. HN Algolia API, no key, ~10k req/hour. Broad net, lower
    relevance density for a Blender-specific niche, but zero marginal cost."""
    out = []
    q = urllib.parse.quote(query)
    try:
        data = _get_json(f"https://hn.algolia.com/api/v1/search?query={q}&tags=story")
        for hit in data.get("hits", [])[:max_results]:
            out.append({
                "source": "hackernews", "tier": 2,
                "title": hit.get("title", "") or "",
                "snippet": (hit.get("story_text") or "")[:500],
                "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                "domain": "news.ycombinator.com",
                "platform": "forum",
                "engagement_signal": hit.get("points", 0),
            })
    except Exception as e:
        print(f"  [warn] hn_algolia quota/error, skipping: {e}", file=sys.stderr)
    return out


# Not implemented yet, noted so the gap is explicit rather than silent:
# - projects.blender.org Gitea API (Tier 1) -- correct source for Blender
#   CORE issues (see NOTE on search_github_issues above). Gitea exposes a
#   free REST API at /api/v1/repos/{owner}/{repo}/issues; no auth needed for
#   public read. This is the highest-relevance source for exporter/geometry-
#   nodes pain points and isn't built yet -- worth adding before Exa/Reddit.
# - Blender Stack Exchange API (Tier 1) -- same shape as devtalk, add if needed.
# - Reddit official API (Tier 2) -- needs manual OAuth app approval (weeks),
#   and non-commercial classification must be re-checked once/if the addon
#   earns revenue. Apply early if you want this source.
# - Exa (Tier 2, 20k req/mo free) -- needs an API key; best general-web
#   replacement for the DuckDuckGo scrape in agent.py once you have one.
