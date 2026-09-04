"""Field test 8 - would a concept-page growth loop terminate?

The proposal: when a concept has no page, write one from the paragraphs that
mention it; the analysis pass then links that page, which surfaces more
concepts, and the loop repeats until nothing new comes out.

The claim that it self-limits is that writing needs DATA - a concept only
mentioned in passing has no definition to write from, so the loop stops at the
edge of what the corpus actually explains.

That is the right principle, and it is not self-enforcing. A model handed one
passing sentence does not refuse; it writes a thin plausible paragraph. So
whether the loop terminates depends on two numbers, and both are measurable on
a real vault rather than arguable:

  SUPPLY   how much mention text exists per unresolved concept
  BRANCH   how many new concepts a written page contributes

Termination needs supply to run out faster than branching replenishes it. This
script measures both and projects the generations.

    PYTHONPATH=. python3 -m sentinel.tests.field_test_growth \\
        --vault ~/obsidian/Obsidian-1 --exclude agent_workspace
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

from ..index import Index
from ..text import shape_ok, strip_links

# A sentence that says what something IS, rather than one that merely uses it.
# Deliberately crude: the point is a distribution, not a classifier.
DEFINING = re.compile(
    r"\b(is|are|is called|refers to|is defined|denotes|means|consists of|"
    r"describes|is the|are the)\b", re.I)


def mention_text(idx, key: str, display: str) -> list[str]:
    rows = idx.db.execute(
        """SELECT c.content FROM chunks c
           WHERE c.is_summary = 0
             AND c.path IN (SELECT source FROM links WHERE target_key = ?)
             AND c.content LIKE ?""", (key, f"%{display}%"))
    return [strip_links(r["content"]) for r in rows]


def defining_sentences(chunks: list[str], display: str) -> list[str]:
    out = []
    for c in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", c):
            if display.lower() in sent.lower() and DEFINING.search(sent):
                out.append(" ".join(sent.split()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--show", type=int, default=0,
                    help="Print the actual sentences the defining-sentence "
                         "test matched, for this many concepts. The test is a "
                         "crude keyword pattern - `X is exceptionally bright` "
                         "matches it and is not a definition - so the count it "
                         "produces is a hypothesis until these are read.")
    ap.add_argument("--min-chars", type=int, default=400,
                    help="Mention text a concept needs before a page could be "
                         "written from it rather than invented.")
    args = ap.parse_args()

    v = Path(args.vault).expanduser()
    db = v / ".sentinel" / "index.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    idx = Index(v, db, exclude=args.exclude)
    idx.sync()

    # ---- BRANCH: what does an analysed page contribute? ------------------
    per_page = []
    for r in idx.db.execute(
            "SELECT path FROM files WHERE summary_provisional = 0"):
        n = idx.db.execute(
            "SELECT COUNT(*) c FROM links WHERE source = ?",
            (r["path"],)).fetchone()["c"]
        words = sum(len(x["content"].split()) for x in idx.db.execute(
            "SELECT content FROM chunks WHERE path = ? AND is_summary = 0",
            (r["path"],)))
        if words > 100:
            per_page.append((r["path"], n, words))

    print("=== BRANCH: concepts contributed per analysed page ===")
    if per_page:
        counts = [n for _, n, _ in per_page]
        print(f"  {len(per_page)} analysed pages, {sum(counts)} links")
        print(f"  per page: median {statistics.median(counts):.0f}, "
              f"mean {statistics.mean(counts):.1f}, "
              f"range {min(counts)}-{max(counts)}")
        dens = [n / (w / 1000) for _, n, w in per_page]
        print(f"  per 1000 words: median {statistics.median(dens):.1f}")
    else:
        print("  no analysed pages yet - run the analysis pass first")
        return

    # ---- SUPPLY: what could actually be written? -------------------------
    rows = list(idx.db.execute(
        """SELECT target_key, MIN(display) d, COUNT(*) n FROM links
           WHERE resolved = 0 GROUP BY target_key ORDER BY n DESC"""))
    rows = [r for r in rows if shape_ok(r["d"])]

    supply = []
    for r in rows:
        chunks = mention_text(idx, r["target_key"], r["d"])
        chars = sum(len(c) for c in chunks)
        defs = defining_sentences(chunks, r["d"])
        supply.append((r["d"], r["n"], chars, len(defs)))

    print(f"\n=== SUPPLY: {len(supply)} unresolved, concept-shaped names ===")
    by_refs = {}
    for _, n, chars, defs in supply:
        b = "1" if n == 1 else "2" if n == 2 else "3+"
        by_refs.setdefault(b, []).append((chars, defs))
    for b in ("3+", "2", "1"):
        items = by_refs.get(b, [])
        if not items:
            continue
        enough = sum(1 for c, _ in items if c >= args.min_chars)
        defined = sum(1 for _, d in items if d)
        print(f"  {b} references: {len(items)} concepts, "
              f"{enough} with >= {args.min_chars} chars of mention text, "
              f"{defined} with a defining sentence")

    writable = [s for s in supply if s[2] >= args.min_chars and s[3]]
    thin = [s for s in supply if s[2] < args.min_chars or not s[3]]
    print(f"\n  writable from the text:     {len(writable)}")
    print(f"  would have to be invented:  {len(thin)}")
    print("\n  the ten best-supplied:")
    for d, n, chars, defs in sorted(supply, key=lambda x: -x[2])[:10]:
        mark = "ok  " if (chars >= args.min_chars and defs) else "thin"
        print(f"    [{mark}] {d:<34} {n} refs, {chars:>5} chars, "
              f"{defs} defining")
    print("\n  ten of the thinnest:")
    for d, n, chars, defs in sorted(supply, key=lambda x: (x[3], x[2]))[:10]:
        mark = "ok  " if (chars >= args.min_chars and defs) else "thin"
        print(f"    [{mark}] {d:<34} {n} refs, {chars:>5} chars, "
              f"{defs} defining")
    same = [s for s in supply if s[2] >= args.min_chars and not s[3]]
    if same:
        print(f"\n  {len(same)} concepts sit in plenty of text but are never "
              f"DEFINED in it -\n  they are used, not explained. Character "
              f"count alone would pass them;\n  that is the difference "
              f"between having context and having a definition.")

    if args.show:
        print(f"\n=== the matched sentences, judge them yourself ===")
        for d, n, chars, defs in sorted(
                supply, key=lambda x: -x[3])[:args.show]:
            if not defs:
                continue
            chunks = mention_text(idx, next(
                r["target_key"] for r in rows if r["d"] == d), d)
            print(f"\n  {d}  ({n} refs, {defs} matched)")
            for sent in defining_sentences(chunks, d)[:3]:
                print(f"    - {sent[:220]}")

    # ---- projection ------------------------------------------------------
    branch = statistics.median([n for _, n, _ in per_page])
    print(f"\n=== PROJECTION ===")
    print(f"  generation 1: {len(writable)} pages could be written from the "
          f"text as it stands")
    print(f"  each analysed page contributes a median of {branch:.0f} links")
    print(f"  so generation 1 surfaces roughly "
          f"{int(len(writable) * branch)} concept slots, almost all at ONE "
          f"reference")
    print("\n  Whether that terminates depends on how many of those single-"
          "reference\n  concepts arrive with enough text to be written from. "
          "That is the\n  number in the '1 references' row above - if it is "
          "near zero, the loop\n  starves by itself and no threshold is "
          "needed. If it is not, the\n  supply gate is what stops it, and it "
          "has to be a measured gate rather\n  than an assumption.")


if __name__ == "__main__":
    main()
