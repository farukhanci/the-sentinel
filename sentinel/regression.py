"""The Sentinel - linker regression assertions.

Earned during field test 3 on the real page. Run these after EVERY linker
change. They are a library and not a test file on purpose: the checks that
matter have to run against the real vault, and none of the bugs they caught
showed up in synthetic text.

    from sentinel.regression import check_linker
    failures = check_linker(before, after, relink=lambda b: link_body(b, cs, ix)[0])
"""

from __future__ import annotations

import re

from .text import content_hash, strip_links

_INLINE_MATH = re.compile(r"\$[^$\n]+\$")
_DISPLAY_MATH = re.compile(r"\$\$[^$]*\$\$")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_HEADING = re.compile(r"^#{1,6}[^\n]*$", re.M)
_SINGLE_BRACKET = re.compile(r"(?<!\[)\[[^\[\]\n]+\](?!\])")


def check_linker(before: str, after: str, relink=None) -> list[str]:
    """Returns a list of failure descriptions. Empty list means all pass."""
    fails: list[str] = []

    def count(pat, s):
        return len(pat.findall(s))

    if count(_DISPLAY_MATH, before) != count(_DISPLAY_MATH, after):
        fails.append("display math spans changed")
    if count(_INLINE_MATH, before) != count(_INLINE_MATH, after):
        fails.append("inline math spans changed")
    if count(_INLINE_CODE, before) != count(_INLINE_CODE, after):
        fails.append("inline code spans changed")
    if _HEADING.findall(before) != _HEADING.findall(after):
        fails.append("headings changed")
    if "[[[" in after:
        fails.append("nested brackets produced")
    if count(_SINGLE_BRACKET, before) != count(_SINGLE_BRACKET, after):
        # This is the one that caught the v8 leftover-bracket stripper
        # deleting `[n_0]` from a real page.
        fails.append("single-bracket count changed - user content was touched")
    if strip_links(before) != strip_links(after):
        fails.append("text differs once links are stripped")

    # The two properties. Neither showed up in synthetic text.
    if content_hash(before) != content_hash(after):
        fails.append("content_hash moved - every outstanding `expect` is now stale")
    if relink is not None and relink(after) != after:
        fails.append("not idempotent - a second pass changes the body")

    return fails
