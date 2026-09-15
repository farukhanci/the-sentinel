"""Run: python3 -m sentinel.tests.run_tests"""

from __future__ import annotations

import sys

from ..names import DEFER, DIFFERENT, SAME, resolve_pair
from ..regression import check_linker
from ..text import (
    content_hash,
    link_body,
    normalize,
    shape_ok,
    strip_links,
)

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


# ---------------------------------------------------------------------------
# The fixture: every trap the real page contained
# ---------------------------------------------------------------------------

BODY = """---
title: Gamma-ray bursts
summary: A provisional first sentence.
---

# Gamma-ray bursts and the forward shock

Markdown hard-wraps at 80 columns, so the gamma-ray
burst population divides into two classes. See [[afterglow]] for the
late-time emission; the afterglow itself is modelled separately.

The initial Lorentz factor $\\Gamma_0$ is estimated from the onset bump
feature. The compactness problem gives `[n_0]` as the density, and the
initial Lorentz factor follows.

The gamma-ray burst rate is quoted per unit volume.

## Method

We fit with a Markov Chain Monte Carlo sampler.

```python
gamma_ray_burst = 1  # synchrotron radiation everywhere in here
```

$$
E_{iso} = 10^{52} \\; \\mathrm{erg}
$$

See [the review](https://example.com/gamma-ray-burst) for context.
"""

CONCEPTS = [
    "gamma-ray burst",
    "forward shock",
    "initial Lorentz factor",
    "Markov Chain Monte Carlo",
    "onset bump feature",
    "compactness problem",
    "afterglow",
    "synchrotron radiation",
]

INDEX = {
    normalize("gamma-ray burst"): "Gamma-ray burst",
    normalize("afterglow"): "afterglow",
}

after, report = link_body(BODY, CONCEPTS, INDEX)

# --- the regression assertions -------------------------------------------
fails = check_linker(BODY, after, relink=lambda b: link_body(b, CONCEPTS, INDEX)[0])
ok("linker: 6 corruption checks + hash + idempotency", not fails, "; ".join(fails))

# --- placement behaviour -------------------------------------------------
ok(
    "wrapped concept is matched across the line break",
    "[[Gamma-ray burst|gamma-ray\nburst]]" in after,
)
ok(
    "no pipe when display equals canonical",
    "[[afterglow]]" in after and "[[afterglow|" not in after,
)
ok(
    "second, unwrapped occurrence is suppressed",
    after.count("[[Gamma-ray burst") == 1,
    f"got {after.count('[[Gamma-ray burst')}",
)
ok(
    "unresolved concepts are KEPT as links (the growth queue)",
    "[[initial Lorentz factor]]" in after and "[[compactness problem]]" in after,
)
ok(
    "first occurrence only, per page",
    after.count("[[initial Lorentz factor]]") == 1,
    f"got {after.count('[[initial Lorentz factor]]')}",
)
ok(
    "existing link seeds the set - afterglow is not linked twice",
    after.count("[[afterglow]]") == 1,
    f"got {after.count('[[afterglow]]')}",
)
ok("heading is protected - forward shock unlinked", "[[forward shock]]" not in after)
ok(
    "code fence is protected - synchrotron radiation unlinked",
    "[[synchrotron radiation]]" not in after,
)
ok("user's `[n_0]` survives untouched", "`[n_0]`" in after)
ok("markdown link untouched", "[the review](https://example.com/gamma-ray-burst)" in after)
ok(
    "nothing reported absent that is merely already linked",
    "afterglow" not in report.absent,
    f"absent={report.absent}",
)

# --- hash behaviour ------------------------------------------------------
ok("hash survives linking", content_hash(BODY) == content_hash(after))
ok(
    "hash still moves on a real content edit",
    content_hash(BODY) != content_hash(BODY.replace("two classes", "three classes")),
)
ok(
    "strip_links handles a link spanning a newline",
    "\n" in strip_links("[[Canon|two\nwords]]"),
)

# --- the hash is about content, not layout -------------------------------
from ..text import page_hash, render_page, split_frontmatter  # noqa: E402

SHAPES = {
    "blank line after fence": "---\ntype: note\n---\n\n# I\n\nText.\n",
    "no blank line":          "---\ntype: note\n---\n# I\n\nText.\n",
    "no frontmatter":         "# I\n\nText.\n",
    "leading blank lines":    "\n\n# I\n\nText.\n",
}
for name, doc in SHAPES.items():
    fm, b = split_frontmatter(doc)
    fm["summary"] = "Written by the analysis pass."
    ok(f"rewriting a page with {name} does not move the hash",
       page_hash(doc) == page_hash(render_page(fm, b)))
ok("but a real edit still does",
   content_hash("# A\n\nText.") != content_hash("# A\n\nOther."))
ok("and placing a link still does not",
   content_hash("the afterglow here") == content_hash("the [[afterglow]] here"))

# --- normalisation -------------------------------------------------------
for a, b in [
    ("Gamma-ray bursts", "Gamma-Ray Burst"),
    ("GRB", "GRBs"),
    ("phases", "phase"),
    ("categories", "category"),
]:
    ok(f"normalise folds {a!r} = {b!r}", normalize(a) == normalize(b))
ok("normalise keeps digits apart", normalize("version1") != normalize("version2"))

# --- shape filter --------------------------------------------------------
for good in ["gamma-ray burst", "Markov Chain Monte Carlo", "Amati relation"]:
    ok(f"shape keeps {good!r}", shape_ok(good))
for bad in ["Tier 2", "Figure 3", "two classes", "a", "10^52", "Section IV"]:
    ok(f"shape rejects {bad!r}", not shape_ok(bad))

# --- name resolution -----------------------------------------------------
CASES = [
    ("GRB", "gamma-ray burst", SAME),
    ("GRBs", "Gamma-ray bursts", SAME),
    ("JWST", "James Webb Space Telescope", SAME),
    ("GRB", "GRB 200826A", DIFFERENT),
    ("GRBs", "GRB 221009A", DIFFERENT),
    ("K2-18b", "K2-18c", DIFFERENT),
    ("LM2576", "LM2577", DIFFERENT),
    ("version1", "version2", DIFFERENT),
    ("Ariel", "Ariel Tier 2", DIFFERENT),
    ("Ariel", "Ariel Space Telescope", DEFER),
    ("gamma-ray burst", "short gamma-ray burst", DEFER),
    # measured: the acronym rule cannot reach this - initials are "im",
    # not "ism". It is exactly why mention-context proximity exists as a
    # second candidate generator.
    ("ISM", "interstellar medium", DEFER),
    ("forward shock", "FS model", DEFER),
    ("reverse shock", "RS model", DEFER),
    ("initial Lorentz factor", "Lorentz factor", DEFER),
    ("GRB", "GRB fireball", DEFER),
]
for a, b, want in CASES:
    got = resolve_pair(a, b)
    ok(f"resolve {a!r} ? {b!r} -> {want}", got == want, f"got {got}")

# ---------------------------------------------------------------------------
# --- frontmatter a YAML parser can read back ------------------------------
from ..text import yaml_value  # noqa: E402

_colon = ("The text summarizes the generator's mechanics and offers three "
          "directions for deeper exploration: mathematical formulas, DIY "
          "construction, and history.")
_out = render_page({"type": "note", "summary": _colon, "tags": "a, b"},
                   "# t\n\nBody.\n")
_block = _out.split("---")[1]
try:
    import yaml as _yaml
    ok("a summary containing ': ' still parses as YAML",
       _yaml.safe_load(_block)["summary"] == _colon, _block)
except ImportError:
    ok("a summary containing ': ' is quoted", '"' + _colon + '"' in _out, _block)

ok("and it round-trips through our own parser",
   split_frontmatter(_out)[0]["summary"] == _colon,
   repr(split_frontmatter(_out)[0].get("summary")))
ok("ordinary values are left unquoted",
   "type: note" in _out and "tags: a, b" in _out, _block)
for _v in ("a: b", "#tag", " leading", "trailing ", "", "- item", "*star"):
    ok(f"{_v!r} is quoted", yaml_value(_v).startswith('"'), yaml_value(_v))
ok("a value with a quote in it is escaped",
   yaml_value('say "hi": now') == '"say \\"hi\\": now"', yaml_value('say "hi": now'))

# --- a foreign writer's YAML list is skipped, not misread -----------------
# the-searcher writes `sources:` as a real YAML list. This reader is flat by
# design (see split_frontmatter's docstring); the list lines must be dropped
# whole, not partitioned into a `- https`/`- http` key holding the last URL.
_list_doc = (
    "---\n"
    "type: source\n"
    "sources: \"\"\n"
    "  - https://a.example.com/one\n"
    "  - https://b.example.com/two\n"
    "  - http://c.example.com/three\n"
    "---\n\n# Body\n"
)
_list_fm, _list_body = split_frontmatter(_list_doc)
ok("a YAML list line is not read as a key",
   "- https" not in _list_fm and "- http" not in _list_fm, str(_list_fm))
ok("scalar fields around a foreign list still parse",
   _list_fm.get("type") == "source", str(_list_fm))
ok("the list lines are dropped, not collapsed into one bogus key",
   len(_list_fm) == 2, str(_list_fm))
ok("the body is unaffected by a list in frontmatter",
   _list_body == "\n# Body\n", repr(_list_body))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
if not FAIL:
    print("  linker report:", report)
    print("  unresolved (growth queue):", report.unresolved)
sys.exit(1 if FAIL else 0)
