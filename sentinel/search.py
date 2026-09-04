"""The Sentinel - the search ladder.

Cheapest rung first; each answers outright or hands down.

  0. exact page lookup   - normalise, check the index. Free. (Index.resolve)
  1. summary-level       - both systems, fused, return the hits' summary lines.
                           THE NORMAL STOPPING PLACE.
  2. body-level          - only when the summaries showed a page is relevant
                           but did not contain the answer. Verbatim chunks.
  3. open the file       - through the bounded depths of `read`, never a dump.

NEVER SHOW A SIMILARITY SCORE. Every score measured on this model fell between
0.740 and 0.861, wrong answers included, with winning margins of 0.015 to
0.064. Printing `(similarity 0.82)` reads as confident when the wrong answer
scored 0.79. Only RANK carries information - which is also why the fusion is
RRF, which is rank-based rather than score-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .embed import from_blob

RRF_K = 60
SUMMARY_DISPLAY = 160


def _short(summary: str | None) -> str:
    """See tools._short - the same cap, applied where search formats its hits."""
    if not summary:
        return "-"
    s = " ".join(summary.split())
    return s if len(s) <= SUMMARY_DISPLAY else s[:SUMMARY_DISPLAY - 1] + "…"


@dataclass
class Hit:
    path: str
    summary: str | None = None
    provisional: bool = True
    from_conversation: bool = False
    literal: bool = False   # did the WORDING match, or only the meaning?
    chunks: list[tuple[str | None, str]] = field(default_factory=list)


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Fuse ranked lists by reciprocal rank. Ties broken by first appearance."""
    score: dict[str, float] = {}
    order: dict[str, int] = {}
    for lst in rankings:
        for rank, key in enumerate(lst):
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank + 1)
            order.setdefault(key, len(order))
    return sorted(score, key=lambda x: (-score[x], order[x]))


# ---------------------------------------------------------------------------
# The two systems
# ---------------------------------------------------------------------------


def _transcript_paths(index) -> set:
    return {r["path"] for r in index.db.execute(
        "SELECT path FROM files WHERE type='transcript'")}


def literal_ranking(index, query: str, limit: int = 50):
    """BM25 over the link-stripped chunk text."""
    q = " OR ".join(f'"{w}"' for w in query.split() if w.strip('"'))
    if not q:
        return []
    try:
        rows = index.db.execute(
            "SELECT path, seq FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?", (q, limit))
    except Exception:
        return []                       # a malformed query is not a crash
    return [(r["path"], r["seq"]) for r in rows]


def semantic_ranking(index, encoder, query: str, limit: int = 50):
    import numpy as np

    rows = list(index.db.execute(
        "SELECT c.path, c.seq, v.vec FROM chunks c "
        "JOIN vectors v ON v.stripped_hash = c.stripped_hash"))
    if not rows:
        return []
    mat = np.array([from_blob(r["vec"]) for r in rows], dtype=np.float32)
    q = np.asarray(encoder.encode([query], kind="query")[0], dtype=np.float32)
    scores = mat @ q                     # both sides are L2-normalised
    top = np.argsort(-scores)[:limit]
    return [(rows[i]["path"], rows[i]["seq"]) for i in top]


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def search(index, query: str, encoder=None, depth: str = "summary",
           limit: int = 5, chunks_per_file: int = 2) -> list[Hit]:
    """`depth` is DEPTH OF ONE INTENT, which is why it is allowed to exist.

    Both depths run the same retrieval; what differs is how much of the hit is
    handed back. A `kind` parameter that switched between semantic retrieval
    and deterministic enumeration would be a different intent, and that is the
    anti-pattern - it lives in `listing` instead.
    """
    lit = literal_ranking(index, query)
    lit_paths = {p for p, _ in lit}
    lists = [lit]
    if encoder is not None:
        lists.append(semantic_ranking(index, encoder, query))

    # A TRANSCRIPT is out of rung 1 and in at rung 2.
    #
    # Blocking it outright would be wrong: it holds what the summary page
    # dropped, and that is the reason it is kept. Returning it at rung 1 would
    # be wrong too - the same conversation would come back twice, once
    # summarised and once raw, and the raw copy is a poor thing to read a
    # one-line description of.
    #
    # The ladder already draws this distinction. Cheap rung: the summary is
    # enough. Deeper rung: it was not, so read what was actually said.
    if depth == "summary":
        skip = _transcript_paths(index)
        lists = [[(p, s) for p, s in lst if p not in skip] for lst in lists]

    fused = rrf([[f"{p}\x00{s}" for p, s in lst] for lst in lists])

    pages: list[str] = []
    per_page: dict[str, list[int]] = {}
    for key in fused:
        path, seq = key.split("\x00")
        if path not in per_page:
            per_page[path] = []
            pages.append(path)
        if len(per_page[path]) < chunks_per_file:
            per_page[path].append(int(seq))
        if len(pages) >= limit and all(
                len(per_page[p]) >= chunks_per_file for p in pages[:limit]):
            break
    pages = pages[:limit]
    if not pages:
        return []

    meta = {r["path"]: r for r in index.summaries(pages)}
    hits = []
    for p in pages:
        row = meta.get(p)
        hit = Hit(path=p,
                  summary=row["summary"] if row else None,
                  provisional=bool(row["summary_provisional"]) if row else True,
                  from_conversation=bool(row and row["origin"] == "conversation"),
                  literal=p in lit_paths)
        if depth == "body":
            for seq in per_page[p]:
                c = index.db.execute(
                    "SELECT heading, content FROM chunks WHERE path=? AND seq=?",
                    (p, seq)).fetchone()
                if c:
                    hit.chunks.append((c["heading"], c["content"]))
        hits.append(hit)
    return hits


def format_hits(hits: list[Hit], query: str, depth: str = "summary") -> str:
    """Ranked shape: state the total and STOP. The remainder is less relevant,
    so do not invite a next page - that shape belongs to `read`.

    A page awaiting analysis still carries its provisional first-sentence
    summary, and it is MARKED. Without the mark the ladder presents a weak
    description as a real one and the model stops at rung 1 on a page it
    should have opened.
    """
    if not hits:
        # A real answer. Do not fill the gap from training knowledge and
        # present it as the vault's.
        return f"[STOP] nothing in the vault matches '{query}'"

    # DENSE RETRIEVAL NEVER RETURNS NOTHING. It hands back its nearest
    # neighbours whether or not any of them is related, and no similarity
    # threshold can separate the two - every score measured on this model,
    # right answers and wrong ones alike, fell between 0.740 and 0.861. So
    # "the vault has nothing on this" cannot come from the retriever. What CAN
    # be reported honestly is provenance: if no page matched the WORDING, say
    # so and let the summaries settle it.
    out = [f"[OK] '{query}' - {len(hits)} hits, ranked"]
    if not any(h.literal for h in hits):
        out.append("[note] no wording match - nearest by meaning only, "
                   "may be unrelated")
    for h in hits:
        mark = (" (provisional summary)" if h.provisional else "") + \
               (" (from conversation)" if h.from_conversation else "")
        out.append(f"  {h.path}{mark}: {_short(h.summary)}")
        for heading, content in h.chunks:
            body = " ".join(content.split())
            out.append(f"      [{heading or 'body'}] {body}")
    if depth == "summary":
        out.append("[OK -> read] open one with read(path, depth=\"outline\")")
    return "\n".join(out)
