"""Field test 7 - the whole read side against the REAL vault. Read-only.

Nothing here writes to the vault. The index is built in a scratch database and
the linker runs in memory; the comparison is against text on disk, never over
it.

This is the step every earlier finding came from. Field test 3 produced three
bugs and not one of them showed up in synthetic text - the leftover-bracket
stripper that deleted `[n_0]`, the two-step marker that dropped every concept
spanning a line break, and a character class excluding \\n in three places. The
test suites written so far are all synthetic, so they cannot catch that class
of thing. This can.

    python3 -m sentinel.tests.field_test_vault --vault ~/obsidian
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp

from ..index import Index
from ..regression import check_linker
from ..text import content_hash, link_body, normalize, split_frontmatter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="A folder to leave out of the index, relative to the "
                         "vault root. Repeat for several. Dot-folders are "
                         "always skipped.")
    ap.add_argument("--pages", type=int, default=5,
                    help="How many of the largest pages to run the linker over.")
    args = ap.parse_args()

    vault = Path(args.vault).expanduser()
    idx = Index(vault, Path(mkdtemp()) / "field.db", exclude=args.exclude)

    t0 = time.perf_counter()
    stats = idx.sync()
    build = time.perf_counter() - t0
    n_files = idx.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
    n_links = idx.db.execute("SELECT COUNT(*) c FROM links").fetchone()["c"]
    n_chunks = idx.db.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]

    print(f"=== index ===")
    print(f"  {n_files} pages, {n_links} links, {n_chunks} chunks, "
          f"built in {build:.1f}s")
    print(f"  link density {n_links / max(n_files, 1):.1f} per page "
          f"(the sizing assumption was 10)")
    if stats["unreadable"]:
        # Loud failure over silent failure: a scan that could not read once
        # reported a CLEAN vault.
        print(f"  [note] {len(stats['unreadable'])} UNREADABLE:")
        for path, err in stats["unreadable"][:5]:
            print(f"      {path}: {err}")
    else:
        print("  no unreadable file")

    missing = [r["path"] for r in idx.db.execute(
        "SELECT path FROM files WHERE summary IS NULL OR summary=''")]
    print(f"  pages with no extractable summary: {len(missing)}"
          + (f" e.g. {missing[:3]}" if missing else ""))
    prov = idx.db.execute(
        "SELECT COUNT(*) c FROM files WHERE summary_provisional=1").fetchone()["c"]
    print(f"  provisional summaries: {prov} of {n_files}")

    print(f"\n=== query timings ===")
    sample = idx.db.execute("SELECT path, page_key FROM files LIMIT 1").fetchone()
    if sample:
        def ms(fn, n=50):
            t = time.perf_counter()
            for _ in range(n):
                fn()
            return (time.perf_counter() - t) * 1000 / n
        for label, fn in [
            ("links out of a page", lambda: idx.links_out(sample["path"])),
            ("links into a concept", lambda: idx.links_in(sample["page_key"], 10)),
            ("growth queue ranked", lambda: idx.growth_queue(8)),
            ("orphan pages", idx.orphans),
            ("summaries for 5 hits", lambda: idx.summaries([sample["path"]])),
        ]:
            print(f"  {label:<24}{ms(fn):.3f} ms")

    dupes = list(idx.db.execute(
        """SELECT page_key, COUNT(*) n, GROUP_CONCAT(path, ' | ') paths
           FROM files GROUP BY page_key HAVING n > 1"""))
    print(f"\n=== pages sharing a stem: {len(dupes)} ===")
    for d in dupes[:5]:
        print(f"  {d['page_key']}: {d['paths']}")
    if dupes:
        print("  a bare-name read of these returns [RETRY] listing both, which "
              "is the designed behaviour")

    print(f"\n=== normalisation on your real names ===")
    folds = defaultdict(list)
    for r in idx.db.execute("SELECT DISTINCT display FROM links"):
        folds[normalize(r["display"])].append(r["display"])
    merged = {k: v for k, v in folds.items() if len(set(v)) > 1}
    print(f"  {len(folds)} keys from "
          f"{sum(len(set(v)) for v in folds.values())} display forms")
    for k, v in list(merged.items())[:8]:
        print(f"    {k}: {sorted(set(v))}")

    gq, total = idx.growth_queue(10)
    print(f"\n=== growth queue: {total} unresolved targets ===")
    for r in gq:
        print(f"  {r['display']}  referenced by {r['n']}")
    if gq and all(r["n"] == 1 for r in gq):
        print("  [note] every entry at count 1 - the ranking signal that makes "
              "this queue useful does not exist until there is scale")

    # ---- the linker, on real text ----------------------------------------
    print(f"\n=== linker regression on your {args.pages} largest pages ===")
    print("  concepts are taken from the links the page ALREADY has, so this")
    print("  measures idempotency and corruption, not extraction quality.\n")
    resolution = {}
    for r in idx.db.execute("SELECT path, page_key, type FROM files"):
        name = r["path"].rsplit("/", 1)[-1][:-3]
        if r["page_key"] not in resolution or r["type"] == "concept":
            resolution[r["page_key"]] = name

    pages = sorted(vault.rglob("*.md"), key=lambda p: p.stat().st_size,
                   reverse=True)[:args.pages]
    all_ok = True
    for f in pages:
        raw = f.read_text(encoding="utf-8")
        body = split_frontmatter(raw)[1]
        concepts = sorted({t.split("|")[0].strip()
                           for t in re.findall(r"\[\[([^\]]+)\]\]", body)})
        if not concepts:
            print(f"  {f.name}: no links to re-place, skipped")
            continue
        after, report = link_body(body, concepts, resolution)
        # Honest about what this proves: the concepts come from links the page
        # already has, so the seed suppresses every one of them. That measures
        # non-corruption and idempotency, NOT placement.
        fails = check_linker(body, after,
                             relink=lambda b: link_body(b, concepts, resolution)[0])
        status = "PASS" if not fails else "FAIL"
        all_ok &= not fails
        print(f"  {f.name} ({len(body)} chars, {len(concepts)} concepts): {status}")
        print(f"      {len(report.resolved)} resolved, "
              f"{len(report.unresolved)} unresolved, "
              f"{report.suppressed} repeats suppressed, "
              f"{report.in_protected} inside math or code left alone")
        if report.absent:
            print(f"      [note] reported absent: {report.absent[:5]}")
        for x in fails:
            print(f"      FAIL {x}")
        if content_hash(body) != content_hash(after):
            print("      FAIL content_hash moved")

    print("\n" + ("all linker checks hold on real text"
                  if all_ok else "SOMETHING BROKE - the detail is above"))


if __name__ == "__main__":
    main()
