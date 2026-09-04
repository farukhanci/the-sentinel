"""The Sentinel - repair nested-bracket damage.

Found on the real vault: the `jwst` key collected three display forms,
`JWST`, `[JWST` and `[[JWST`, which means the files carry `[[[JWST]]` and
`[[[[JWST]]]]`. That is the accumulation the earlier system was recorded as
producing, and it is still in the text.

Normalisation folds all three to one key, so the graph is correct in spite of
it. What is NOT correct is the text: `strip_links` substitutes once, so
`[[[[JWST]]]]` reduces to `[[JWST]]` rather than to `JWST`, and every derived
value computed from it - the content hash, the chunk that gets embedded, the
concept-presence check - is computed over markup that should not be there.

    python3 -m sentinel.repair_brackets --vault ~/obsidian          # dry run
    python3 -m sentinel.repair_brackets --vault ~/obsidian --apply

DRY RUN BY DEFAULT. This is the only tool in the system that rewrites files it
was not asked to write, so it shows the exact change first and touches nothing
until told twice.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .text import content_hash, split_frontmatter

# Two or more brackets on each side collapse to exactly two. A legitimate
# `[[X]]` matches and is rewritten to itself, which is harmless and keeps the
# pattern simple. A markdown link `[text](url)` has ONE bracket and cannot
# match.
DAMAGE = re.compile(r"\[{2,}([^\[\]]+)\]{2,}")

# Code and math only - deliberately NOT the wikilink span from text.PROTECT,
# which would protect the very thing being repaired. A fenced block may
# legitimately contain `[[x]]` as an example, and this file is about not
# touching the user's content.
SKIP = re.compile(
    r"```.*?```|`[^`\n]+`|\$\$[^$]*\$\$|\$[^$\n]+\$", re.S | re.M)


def planned(body: str) -> list[tuple[str, str]]:
    """Exactly the changes `repair` will make, and nothing else.

    The preview and the edit MUST come from one function. A first draft
    listed every DAMAGE match, including ones inside a code fence that the
    repair then skipped - so the dry run advertised changes that would never
    happen. Showing a change you will not make is its own kind of silent
    failure.
    """
    spans = [(m.start(), m.end()) for m in SKIP.finditer(body)]
    out = []
    for m in DAMAGE.finditer(body):
        if any(s <= m.start() < e for s, e in spans):
            continue
        fixed = f"[[{m.group(1)}]]"
        if fixed != m.group(0):
            out.append((m.group(0), fixed))
    return out


def repair(body: str) -> tuple[str, int]:
    spans = [(m.start(), m.end()) for m in SKIP.finditer(body)]
    fixed = 0

    def sub(m: re.Match) -> str:
        if any(s <= m.start() < e for s, e in spans):
            return m.group(0)
        nonlocal fixed
        out = f"[[{m.group(1)}]]"
        if out != m.group(0):
            fixed += 1
        return out

    return DAMAGE.sub(sub, body), fixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write. Without this nothing is touched.")
    args = ap.parse_args()

    vault = Path(args.vault).expanduser()
    total_files = total_fixes = 0

    for f in sorted(vault.rglob("*.md")):
        if any(p.startswith(".") for p in f.relative_to(vault).parts):
            continue
        raw = f.read_text(encoding="utf-8")
        fm, body = split_frontmatter(raw)
        new_body, n = repair(body)
        if not n:
            continue
        total_files += 1
        total_fixes += n
        rel = f.relative_to(vault)
        print(f"\n{rel}  ({n} to repair)")
        for was, now in planned(body):
            print(f"    {was!r}  ->  {now!r}")
        if content_hash(body) != content_hash(new_body):
            # Expected and correct: the rendered text really does change, so
            # the page is genuinely re-analysed afterwards.
            print("    [note] content hash moves - this page will be re-analysed")
        if args.apply:
            f.write_text(raw[:len(raw) - len(body)] + new_body, encoding="utf-8")

    if not total_files:
        print("No nested-bracket damage found.")
        return
    print(f"\n{total_fixes} repairs across {total_files} files")
    print("applied" if args.apply
          else "DRY RUN - nothing written. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
