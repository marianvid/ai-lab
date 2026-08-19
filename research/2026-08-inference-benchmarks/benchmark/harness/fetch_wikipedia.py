#!/usr/bin/env python3
"""Build a redistributable article corpus from Wikipedia.

Why this exists. The measurements in this study were run on a corpus of real
news articles, and that text is not ours to publish. Wikipedia is: the article
text is Creative Commons Attribution-ShareAlike 4.0, so it can be redistributed
with attribution. This script collects a set of the same *shape* — full articles
in the same languages, in the same length band — so that anyone can run the
harness on their own machine.

The length band is the point. Sentence-level evaluation sets exist and are
better for measuring quality, but they cannot measure what this study measures:
what happens to an engine when the prompts are long enough to fill a context
window. The band here is the one the original corpus occupied, 2 165 to 5 240
characters.

Collection works in two passes, because most random articles are stubs and
fetching their text would be wasted traffic:

1. ask for a batch of random articles and read only their *wikitext size*,
   which is cheap, and keep the plausible ones;
2. fetch the plain text of those, and keep the ones that really land in the
   band.

Every kept article records its page id, its revision id and its URL, so each
one is attributable and the exact text can be recovered later even if the
article changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Wikipedia asks automated clients to identify themselves and to say how to be
# contacted. https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = ("ai-lab-benchmarks/1.0 (research corpus builder; "
              "https://github.com/marianvid/ai-lab-benchmarks)")

# The band the original news corpus occupied, in characters of plain text.
MIN_CHARS = 2165
MAX_CHARS = 5240

# Wikitext carries markup, templates and references that the plain text does
# not, so a page is always larger in wikitext than in text. These bounds are
# deliberately generous: they only decide what is worth fetching.
MIN_WIKITEXT = 3000
MAX_WIKITEXT = 60000

BATCH = 50              # random pages per request; the API allows up to 500
PAUSE_S = 0.4           # between requests, to stay a polite client


def api(lang: str, params: dict) -> dict:
    """One call to a language edition's API. Raises on anything unexpected."""
    query = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(f"{lang}: {payload['error'].get('info', payload['error'])}")
    return payload


def candidates(lang: str) -> list[dict]:
    """A batch of random articles, filtered to a plausible size by wikitext."""
    payload = api(lang, {
        "action": "query", "generator": "random",
        "grnnamespace": 0, "grnlimit": BATCH,
        "prop": "info", "inprop": "url",
    })
    pages = payload.get("query", {}).get("pages", [])
    return [page for page in pages
            if MIN_WIKITEXT <= page.get("length", 0) <= MAX_WIKITEXT]


def with_text(lang: str, pages: list[dict]) -> list[dict]:
    """Fetch the plain text of a set of pages, in one request."""
    if not pages:
        return []
    payload = api(lang, {
        "action": "query", "pageids": "|".join(str(p["pageid"]) for p in pages),
        "prop": "extracts|revisions", "explaintext": 1, "exsectionformat": "plain",
        "rvprop": "ids",
    })
    return payload.get("query", {}).get("pages", [])


def usable(page: dict) -> str | None:
    """The article's text, or None if this page should not be in the corpus."""
    text = (page.get("extract") or "").strip()
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    # A disambiguation page is a list of links with almost no prose, and a page
    # that is mostly one-line entries is a table in disguise. Both would make
    # the corpus easier than real articles rather than harder.
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    long_lines = [line for line in lines if len(line) > 120]
    if len(long_lines) < 3:
        return None
    return text


def collect(lang: str, wanted: int, budget_requests: int) -> list[dict]:
    """Collect `wanted` articles from one language edition."""
    kept: list[dict] = []
    seen: set[int] = set()
    requests_made = 0
    empty_rounds = 0

    while len(kept) < wanted and requests_made < budget_requests:
        try:
            batch = candidates(lang)
            requests_made += 1
            time.sleep(PAUSE_S)
            pages = with_text(lang, batch)
            requests_made += 1
            time.sleep(PAUSE_S)
        except (urllib.error.URLError, RuntimeError, TimeoutError) as error:
            print(f"  {lang}: {error} — retrying", file=sys.stderr)
            time.sleep(3)
            requests_made += 1
            continue

        before = len(kept)
        for page in pages:
            if len(kept) >= wanted or page["pageid"] in seen:
                continue
            text = usable(page)
            if text is None:
                continue
            seen.add(page["pageid"])
            revisions = page.get("revisions") or [{}]
            kept.append({
                "lang": lang,
                "pageid": page["pageid"],
                "revid": revisions[0].get("revid"),
                "title": page["title"],
                "url": f"https://{lang}.wikipedia.org/?curid={page['pageid']}",
                "chars": len(text),
                "text": text,
            })

        empty_rounds = 0 if len(kept) > before else empty_rounds + 1
        if empty_rounds >= 25:
            print(f"  {lang}: giving up at {len(kept)}/{wanted} — "
                  f"this edition has few articles in the band", file=sys.stderr)
            break
        print(f"  {lang}: {len(kept)}/{wanted}", end="\r", file=sys.stderr)

    print(f"  {lang}: {len(kept)}/{wanted} in {requests_made} requests",
          file=sys.stderr)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", default="ru,uk,lt,pl,ro,en,de,fr,es",
                        help="comma-separated Wikipedia language codes")
    parser.add_argument("--per-language", type=int, default=100)
    parser.add_argument("--budget", type=int, default=400,
                        help="maximum API requests per language")
    parser.add_argument("--out", default="wikipedia_articles.jsonl")
    arguments = parser.parse_args()

    languages = [code.strip() for code in arguments.languages.split(",") if code.strip()]
    everything: list[dict] = []
    for lang in languages:
        everything.extend(collect(lang, arguments.per_language, arguments.budget))

    with open(arguments.out, "w", encoding="utf-8") as handle:
        for row in everything:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in everything:
        counts[row["lang"]] = counts.get(row["lang"], 0) + 1
    print(f"\n{len(everything)} articles -> {arguments.out}")
    for lang in languages:
        print(f"  {lang}: {counts.get(lang, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
