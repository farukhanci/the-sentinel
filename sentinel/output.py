"""The Sentinel - the output policy. ONE rule for all seven primitives.

The old system had five read tools and four different policies: one refused
over 8000 chars, one truncated silently at 3000, one capped at 10 hits without
saying so, one stated its cap, one dumped without limit. Explaining that
inconsistency is part of why its docstrings cost 1302 tokens.

NEVER REFUSE, NEVER TRUNCATE SILENTLY. Refusal existed to push the model up
the ladder; defaults do that now. Silent truncation is the recurring defect
class - a cut answer looks exactly like a complete one.
"""

from __future__ import annotations

import re

# ~2000 tokens. Measured against his real sections (1962-7148 chars): 8 of 9
# arrive whole, so the natural unit fits one call, while a full page still
# splits into four parts - the right friction for a read that should be rare.
BUDGET = 6400

# The full status set. A marker outside this list appearing anywhere is a
# mistake, not an extension.
STATUSES = ("[OK]", "[OK ->", "[DONE]", "[RETRY]", "[STOP]", "[more]", "[note]")

_HEADING = re.compile(r"^#{1,6} ", re.M)
_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")


def cut_point(text: str, budget: int = BUDGET) -> tuple[int, bool]:
    """(index to cut at, whether the cut was hard).

    heading -> paragraph -> sentence -> hard, rejecting any boundary earlier
    than 25% of the budget. At 50% a valid paragraph break at 37% was rejected
    and the text was hard-cut mid-sentence instead - worse on both counts.

    A hard cut is FLAGGED by the caller, because a sentence severed mid-clause
    is how a truncated answer starts looking whole.
    """
    if len(text) <= budget:
        return len(text), False
    window = text[:budget]
    floor = budget * 0.25
    for pat in (_HEADING, _PARA, _SENT):
        spots = [m.start() for m in pat.finditer(window) if m.start() > floor]
        if spots:
            return spots[-1], False
    return budget, True


def split_parts(text: str, budget: int = BUDGET) -> list[tuple[str, bool]]:
    """[(part text, was the cut hard)]. Reassembling every part recovers 100%
    of the input - tested on his real page shape, where every part begins at a
    heading."""
    parts, rest = [], text
    while rest:
        i, hard = cut_point(rest, budget)
        parts.append((rest[:i], hard))
        rest = rest[i:]
    return parts or [("", False)]


def sequential(header: str, body: str, part: int, total: int,
               next_call: str | None, hard: bool = False) -> str:
    """The remainder is EQUALLY important, so name the exact next call.

    This is `read`'s shape and only `read`'s. Continuation overhead measured
    at ~28 tokens.
    """
    lines = [f"[OK] {header} part {part}/{total}"]
    if hard:
        lines.append("[note] cut mid-sentence - no clean boundary was available")
    lines.append(body)
    if next_call:
        lines.append(f"[more] {next_call}")
    else:
        lines.append("[DONE] last part")
    return "\n".join(lines)


def ranked(header: str, shown: int, total: int, body_lines: list[str],
           next_hint: str | None = None) -> str:
    """The remainder is LESS relevant, so state the total and STOP.

    Do NOT invite a next page. This is the shape for `search`, `listing` and
    `graph`; offering pagination here would spend the budget walking down a
    ranking the model already has the top of.
    """
    head = f"[OK] {header} - showing {shown} of {total}, ranked"
    if total > shown:
        head += f"; {total - shown} not shown"
    out = [head, *body_lines]
    if next_hint:
        out.append(f"[OK -> {next_hint}")
    return "\n".join(out)


def health(notes: list[str]) -> str:
    """Health checks are NOT an eighth tool. Every output carries this line
    ONLY when abnormal - zero tokens normally, loud when it matters."""
    return f"\n[note] {'; '.join(notes)}" if notes else ""


def opaque(name: str) -> bool:
    """Titles only where the name carries nothing. `arxiv-2504.11743` is
    opaque; `gamma-ray burst` says what it is.

    Full summaries on every entry were measured and rejected: they pay off
    only if the model opens more than three of five, and it opens nought or
    one. ~27 tokens spent here saves a ~30-token `read(depth="meta")` per
    opaque entry.
    """
    return bool(re.search(r"\d{3,}", name))
