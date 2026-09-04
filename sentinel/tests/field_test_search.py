"""Field test 4 - the search substrate on the real vault, with the real model.

Settles the question left open in the spec: the `query:` / `passage:` prefixes
were measured on THREE query pairs, which is not enough to conclude anything.
This runs both configurations over the same vault and the same queries and
reports where the expected page landed.

    python3 -m sentinel.tests.field_test_search \\
        --vault ~/obsidian --model ~/models/multilingual-e5-small \\
        --queries queries.json

queries.json is a list of {"query": "...", "expect": "wiki/page.md"}. Write it
by hand from pages you know - twenty pairs is worth far more than three, and
mixing Turkish queries against English pages is the case that killed
bge-small-en-v1.5, so include some.

Reports RANK, never a score. Every score on this model falls in a narrow band
whether the answer is right or wrong, so rank is the only signal.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from ..embed import E5Encoder, embed_pending
from ..index import Index
from ..search import search

CONFIGS = {
    "no prefix":  ("", ""),
    "e5 prefix":  ("query: ", "passage: "),
}


def rank_of(hits, expect: str) -> int | None:
    for i, h in enumerate(hits):
        if h.path == expect:
            return i + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    pairs = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    results: dict[str, list] = {}

    for name, (qp, pp) in CONFIGS.items():
        # A separate DB per configuration: changing the prefix changes every
        # vector, so the two cannot share a store.
        tmp = Path(tempfile.mkdtemp()) / f"{name.replace(' ', '_')}.db"
        idx = Index(args.vault, tmp)
        stats = idx.sync()
        if stats["unreadable"]:
            print(f"[note] {len(stats['unreadable'])} unreadable files: "
                  f"{stats['unreadable'][:3]}")

        enc = E5Encoder(args.model, query_prefix=qp, passage_prefix=pp)
        t0 = time.perf_counter()
        n = embed_pending(idx, enc)
        embed_s = time.perf_counter() - t0

        ranks = []
        t0 = time.perf_counter()
        for p in pairs:
            hits = search(idx, p["query"], encoder=enc, limit=args.limit)
            ranks.append((p["query"], p["expect"], rank_of(hits, p["expect"]),
                          [h.path for h in hits[:3]]))
        query_ms = (time.perf_counter() - t0) * 1000 / max(len(pairs), 1)
        results[name] = ranks

        found = sum(1 for _, _, r, _ in ranks if r)
        top1 = sum(1 for _, _, r, _ in ranks if r == 1)
        print(f"\n=== {name} ===")
        print(f"  {stats['seen']} pages, {n} chunks encoded in {embed_s:.1f}s, "
              f"{query_ms:.0f} ms per query")
        print(f"  top-1 {top1}/{len(pairs)}   in top-{args.limit} "
              f"{found}/{len(pairs)}")

    print("\n=== where they disagree ===")
    a, b = list(CONFIGS)
    disagreed = 0
    for (q, exp, ra, _), (_, _, rb, _) in zip(results[a], results[b]):
        if ra != rb:
            disagreed += 1
            print(f"  {q!r} -> expected {exp}: {a} rank {ra}, {b} rank {rb}")
    if not disagreed:
        print("  identical ranking on every query - the prefixes make no "
              "difference here, and the simpler configuration wins")

    print("\n=== misses (both configurations) ===")
    for (q, exp, ra, top), (_, _, rb, _) in zip(results[a], results[b]):
        if not ra and not rb:
            print(f"  {q!r} -> expected {exp}, got {top}")


if __name__ == "__main__":
    main()
