# Free-forever web research agent (R&D / PMF phase)

## What this actually is
A DuckDuckGo-based, multi-pass research script that runs on a schedule via
GitHub Actions (free), structures every result (platform, domain, paid-signal,
complaint-signal, adversarial-signal), stores it in Turso (free tier), and
prints a dense markdown digest sized to paste into a Claude chat.

No LLM call happens inside the agent — that's what keeps "24/7 forever" cost
$0. Claude's intelligence enters when *you* bring the digest (or a DB export)
into a conversation and ask for a read on direction.

## Setup (10 minutes)

1. **Turso** (free): `turso db create rnd-agent` → `turso db show rnd-agent --url`
   and `turso db tokens create rnd-agent` for the auth token.
2. Create a new GitHub repo, add `agent.py` at the root.
3. Move `.github-workflows-research.yml` to `.github/workflows/research.yml`.
4. In repo Settings → Secrets → Actions, add `TURSO_DATABASE_URL` and
   `TURSO_AUTH_TOKEN`.
5. Push. It now runs every 4 hours forever, or trigger manually from the
   Actions tab with a custom topic.

## Local test (no GitHub, no DB)
```bash
pip install ddgs libsql-client
python agent.py "procedural building generator" --no-db
```

## Pulling accumulated research back out for Claude
```sql
-- via `turso db shell rnd-agent`
SELECT pass_type, platform, title, url, has_paid_signal, has_complaint_signal
FROM findings WHERE topic = 'procedural building generator'
ORDER BY fetched_at DESC LIMIT 200;
```
Export that as CSV/markdown and paste it into a Claude conversation (or
`view` it if using Claude Code against a checked-out repo) — that single
paste is what lets Claude "see multiple hours of search results in one dense
output," per the original ask.

## Sourcing: official APIs first (tier 1), scraping last (tier 2)
`agent.py` now runs **two source layers** every cycle:
1. `sources.py` — official, keyless APIs: GitHub Search (glTF-Blender-IO, OpenUSD,
   Alembic — real GitHub repos, *not* blender/blender core, which moved its
   tracker to projects.blender.org years ago), Blender's devtalk Discourse
   JSON API, HN Algolia. These are tagged `tier=1` (or `tier=2` for HN's
   broader/lower-relevance net) and carry a real `engagement_signal`
   (reactions/replies/points) — a much stronger signal than a regex match.
2. The original DuckDuckGo pass — kept for general-web coverage, but now
   explicitly tagged `tier=2` and documented as the weakest-legitimacy
   source in the mix (no clean ToS-compliant path for scraping search
   results). Run with `--skip-scrape` to drop it entirely and go
   official-sources-only.

Not yet added, in priority order if you want to extend further: the
**projects.blender.org Gitea API** (correct source for Blender *core* issues
— higher relevance than the GitHub repos currently used), Blender Stack
Exchange API, Exa (needs a key, best scrape replacement), Reddit's official
API (needs OAuth app approval with real lead time). All noted inline at the
bottom of `sources.py`.

## Rolling diff: only new findings surface each cycle
Every run now queries Turso for all `content_hash`es already stored for that
topic *before* inserting anything, and treats that as the "already seen"
baseline. The printed digest:
- shows full detail only for genuinely new findings,
- collapses repeats into a single "already seen (suppressed): N" count,
- says explicitly "No new findings since last run" when nothing changed.

This makes repeat cycles readable — cycle 40 doesn't re-dump the same 6
gumroad listings you've already read 39 times. `--no-db` mode can't diff
(nothing to compare against), so it always shows everything.

## Optional: LLM enrichment (better dedupe + a synthesis paragraph)
Get a free key at console.groq.com, add it as a repo secret `GROQ_API_KEY`,
and run with `--enrich`. This calls Llama-3.1-8b (free tier) once per run to:
- merge near-duplicate listings into a real cluster count,
- write a 120-180 word synthesis paragraph,
- name the single clearest gap in the evidence (or say "none found").

It writes to a separate `enrichment` table — raw `findings` rows are never
edited, so a bad or overconfident LLM summary never destroys the underlying
evidence. Treat the synthesis as a *starting read*, not a verdict, and check
it against the raw adversarial-pass rows yourself before trusting it — the
same discipline `pmf-engine` applies to its own synthesis step.

Any other OpenAI-compatible free endpoint works too — just override
`LLM_BASE_URL` and `LLM_MODEL` env vars (e.g. OpenRouter's free-tier models,
or a self-hosted Ollama server exposing an OpenAI-compatible API).

## Honest limits
- DuckDuckGo scraping can rate-limit or change markup; expect occasional
  empty passes — the script logs warnings rather than crashing.
- Structuring is heuristic (regex), not semantic — it flags *candidate*
  signals, it doesn't judge them. Treat `$`/`!`/`~` flags as "look here,"
  not as verdicts.
- This is evidence-gathering only, same evidentiary ceiling as pmf-engine's
  `validated_public_signal` — public search evidence, not real interviews.
