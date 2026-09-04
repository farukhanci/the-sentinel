"""The Sentinel - name resolution.

Three mechanisms decide, everything else DEFERS. The property that matters is
zero wrong merges: a wrong merge destroys information irreversibly, a missed
merge costs one unresolved link and is repaired by a later alias entry. The
cost is one-directional, so the system errs in one direction.

What code must NOT decide: modifier-containment with no number.
`gamma-ray burst` / `short gamma-ray burst` (a subclass) and
`Ariel` / `Ariel Space Telescope` (an expansion) have an IDENTICAL shape. A
heuristic on leading adjectives would merge one of them wrongly. Both defer.
"""

from __future__ import annotations

import re

from .text import normalize, tokens

SAME = "same"
DIFFERENT = "different"
DEFER = "defer"


def initials(long: str) -> str:
    return "".join(t[0] for t in re.split(r"[^A-Za-z0-9]+", long.lower()) if t)


def acronym_match(short: str, long: str) -> bool:
    """`GRB` <-> `gamma-ray burst`, `JWST` <-> `James Webb Space Telescope`.

    The plural `s` is stripped from the acronym side FIRST. Without that the
    rule solved NOTHING on real data: `GRBs` failed `[A-Z]{3,6}` outright even
    though the initials of `gamma-ray burst` are exactly `grb`.
    """
    s = short.strip()
    if re.fullmatch(r"[A-Z]{2,6}s", s):
        s = s[:-1]
    if not re.fullmatch(r"[A-Z]{3,6}", s):
        return False
    return s.lower() == initials(long)


def designation_split(a: str, b: str) -> bool:
    """`GRB` / `GRB 200826A`, `K2-18b` / `K2-18c`, `LM2576` / `LM2577`,
    `version1` / `version2`, `Ariel` / `Ariel Tier 2`.

    Both sides are tokenised AND plural-folded before comparison, or `GRBs`
    and `GRB 221009A` fail to line up and fall through to the queue.

    One side carrying a number is enough - class vs instance is still not the
    same thing. Requiring BOTH sides to carry one was the recorded bug.
    """
    ta, tb = tokens(a), tokens(b)
    da = [t for t in ta if any(c.isdigit() for c in t)]
    db = [t for t in tb if any(c.isdigit() for c in t)]

    if da and db:                      # both specific: do the designations differ?
        rest_a = [t for t in ta if t not in da]
        rest_b = [t for t in tb if t not in db]
        return rest_a == rest_b and da != db

    if bool(da) != bool(db):           # class vs instance
        spec, gen = (ta, tb) if da else (tb, ta)
        return set(gen) <= {t for t in spec if not any(c.isdigit() for c in t)}

    return False


def resolve_pair(a: str, b: str) -> str:
    """`same` / `different` / `defer`.

    Order is load-bearing: the designation guard runs BEFORE the acronym rule,
    so `GRB` / `GRB 221009A` splits rather than being tested for expansion.

    A forced binary would be a supply constraint - the third answer costs
    nothing and leaves the pair queued.
    """
    if normalize(a) == normalize(b):
        return SAME
    if designation_split(a, b):
        return DIFFERENT
    if acronym_match(a, b) or acronym_match(b, a):
        return SAME
    return DEFER
