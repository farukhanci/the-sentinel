"""The Sentinel - pure text layer.

No I/O, no database, no model calls. Everything above this file depends on it
and must never re-implement any of it.

Every function carries the mistake it corrects. Without that note the same
mistake gets made again.

One rule runs through the whole file: HASH THE CONTENT, CHUNK THE CONTENT,
EMBED THE CONTENT, SEARCH THE CONTENT - never the markup. `strip_links` is the
single place that rule is implemented.
"""

from __future__ import annotations

import re
from hashlib import sha1

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def fold_plural(token: str) -> str:
    """ies -> y, trailing s -> drop. Deliberately only these two.

    A `(?:s|x|z|ch|sh)es$` branch was tried and REMOVED: it folds `classes`
    to `class` correctly and `phases` to `phas` wrongly, and the two are
    indistinguishable without a lexicon. Since the cost is one-directional -
    a missed fold is one unresolved link repaired by a later alias, a wrong
    fold is a wrong merge - the conservative rule wins.

    `len(token) > 3` guards short words: without it `gas` folds to `ga`.
    """
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def normalize(name: str) -> str:
    """The resolution key for a page or concept name.

    Digits and word order are PRESERVED - `version1` / `version2` are two
    different things. Only plurals fold. Measured: this alone collapsed 20
    display forms to 17 keys for free.
    """
    return fold_plural(re.sub(r"[^a-z0-9]+", "", name.lower()))


def tokens(name: str) -> list[str]:
    """Plural-folded tokens, used by the name-resolution rules."""
    return [fold_plural(t) for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t]


# ---------------------------------------------------------------------------
# Content vs markup
# ---------------------------------------------------------------------------

_PIPED_LINK = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
_PLAIN_LINK = re.compile(r"\[\[([^\]]+)\]\]")


def strip_links(body: str) -> str:
    """`[[Canonical|shown]]` -> `shown`, `[[concept]]` -> `concept`.

    `[^\\]]` and NOT `[^\\]\\n]`: markdown hard-wraps at ~80 columns, so a
    multi-word link lands across a line break constantly.
    """
    return _PLAIN_LINK.sub(r"\1", _PIPED_LINK.sub(r"\2", body))


# Bump this whenever the DEFINITION of content_hash changes - what gets
# stripped, what gets excluded, how whitespace is treated.
#
# It exists because the definition already changed once, silently. Adding
# `.strip()` moved every page's hash, so `analyzed_hash != content_hash`
# everywhere and seventeen analysed pages went back into the queue with
# nothing said. They were fixed by hand with SQL, which is not something a
# system should ask of anyone.
#
# The same class of cost is recorded for the embedding model: changing it
# invalidates every vector. Hashes were simply never expected to change.
HASH_VERSION = 2


def content_hash(body: str) -> str:
    """One value, three uses: the `expect` write guard, the files-table hash,
    and `analyzed_hash`.

    Hashing the rendered CONTENT rather than the markup is what makes that
    possible. Placing links does not move it, so the linking pass never
    invalidates an outstanding read and never marks its own page pending
    again; a real content edit still moves it.
    """
    # `.strip()` because whitespace at the page edges is not content, and
    # rewriting a page normalises it: `render_page` lstrips the body and puts
    # exactly one blank line after the fence. Measured on three of four real
    # page shapes, that alone moved the hash - which failed the analysis pass
    # on a page it had not otherwise touched. Same principle as stripping the
    # link markup and excluding frontmatter: hash what the page SAYS.
    return sha1(strip_links(body).strip().encode("utf-8")).hexdigest()


def concept_present(body: str, concept: str) -> bool:
    """Is this concept verbatim in the text? Runs on the LINK-STRIPPED body.

    Against the raw body the lookbehind fails on a preceding `[`, so three
    concepts were reported absent when they were present and merely already
    linked. The analysis gate uses this, not a raw-body test.
    """
    pat = re.compile(rf"(?<![\[\w]){_flex(concept)}(?![\w\]])", re.I)
    return pat.search(strip_links(body)) is not None


# ---------------------------------------------------------------------------
# Shape filter - one positive test, replacing nine negative rules
# ---------------------------------------------------------------------------

SHAPE = re.compile(r"^[A-Za-z][A-Za-z0-9\-' ]{2,49}$")

_ORDINAL_LABEL = re.compile(
    r"(?:Tier|Cycle|Phase|Stage|Level|Step|Figure|Table|Section)\s*[\dIVX]+", re.I
)
_COUNTED_NOUN = re.compile(
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+\w+s?", re.I
)


def shape_ok(concept: str) -> bool:
    """18/18 noise rejected, 19/19 concepts kept on real candidates.

    A whitelist is stable; a blacklist grows forever.
    """
    c = concept.strip()
    if not SHAPE.match(c):
        return False
    if len(c.split()) > 6:
        return False
    if _ORDINAL_LABEL.fullmatch(c):
        return False
    if _COUNTED_NOUN.fullmatch(c):
        return False
    return True


# ---------------------------------------------------------------------------
# Protected spans
# ---------------------------------------------------------------------------

# Compiled with re.S | re.M. Because re.S is on, the heading pattern must use
# [^\n] and NOT '.', or DOTALL eats the following line. Order matters: display
# math before inline math, frontmatter and fences before everything.
PROTECT = re.compile(
    "|".join(
        [
            r"\A---\n.*?\n---",          # frontmatter
            r"```.*?```",                # code fence
            r"`[^`\n]+`",                # inline code
            r"\$\$[^$]*\$\$",            # display math
            r"\$[^$\n]+\$",              # inline math
            r"^#{1,6}[^\n]*$",           # heading
            r"\[\[[^\]]*\]\]",           # existing wikilink - links wrap
            r"\[[^\]\n]*\]\([^)\n]*\)",  # markdown link
        ]
    ),
    re.S | re.M,
)


def protected_spans(body: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in PROTECT.finditer(body)]


# ---------------------------------------------------------------------------
# The one-pass linker - direct placement
# ---------------------------------------------------------------------------


def _flex(concept: str) -> str:
    """A concept's spaces match ANY whitespace - markdown wraps at 80 columns."""
    return r"\s+".join(re.escape(w) for w in concept.split())


class LinkReport:
    """What the pass did, for the growth queue and for the health line."""

    def __init__(self) -> None:
        self.resolved: list[tuple[str, str]] = []   # (display, canonical)
        self.unresolved: list[str] = []             # kept as [[...]], become the queue
        self.suppressed: int = 0                    # repeat occurrences
        self.in_protected: int = 0                  # inside math, code, headings
        self.rejected_shape: int = 0
        self.absent: list[str] = []                 # not in the text at all

    def __repr__(self) -> str:
        return (
            f"LinkReport(resolved={len(self.resolved)}, "
            f"unresolved={len(self.unresolved)}, suppressed={self.suppressed}, "
            f"in_protected={self.in_protected}, absent={len(self.absent)})"
        )


def link_body(body: str, concepts: list[str], index: dict[str, str]):
    """Place `[[...]]` for every concept, in ONE substitution pass.

    `index` maps normalised key -> canonical page name, built from FILENAMES
    only. Resolution is exact match; no similarity is ever consulted, because
    the measured ordering is inverted (`version1`~`version2` scores 87.5,
    `Ariel`~`Ariel Space Telescope` 38.5).

    Returns (new_body, LinkReport).

    NEVER strips a bracket from the body: the analysis pass does not touch the
    text, so every single bracket there is the user's content. v8's leftover
    stripper deleted `` `[n_0]` `` from a real page and broke the hash.

    No intermediate single-bracket step: it was a v8 artefact from when the
    MODEL did the marking, and it silently dropped every concept spanning a
    line break.
    """
    report = LinkReport()
    if not concepts:
        return body, report

    stripped = strip_links(body)
    for c in concepts:
        if not concept_present(stripped, c):
            report.absent.append(c)

    # Longest first, so `gamma-ray burst` wins over `burst`.
    ordered = sorted(set(concepts), key=len, reverse=True)
    pat = re.compile(
        rf"(?<![\[\w])({'|'.join(_flex(c) for c in ordered)})(?![\w\]])", re.I
    )

    spans = protected_spans(body)

    def protected(pos: int) -> bool:
        return any(s <= pos < e for s, e in spans)

    # Seed the first-occurrence set from links ALREADY in the body. Without
    # this, re-running double-links: the protected-span check skips an existing
    # link but does not COUNT it, so the first-occurrence rule permits a second.
    # This omission alone broke idempotency.
    already = {
        normalize(t.split("|")[0]) for t in re.findall(r"\[\[([^\]]+)\]\]", body)
    }

    def repl(m: re.Match) -> str:
        hit = m.group(0)
        if protected(m.start()):
            report.in_protected += 1
            return hit                       # LEAVE IT. Exactly as it is.
        if not shape_ok(" ".join(hit.split())):
            report.rejected_shape += 1
            return hit
        key = normalize(hit)
        if key in already:
            report.suppressed += 1
            return hit                       # first occurrence only, per page
        already.add(key)
        target = index.get(key)              # EXACT match, no similarity
        if target is None:
            report.unresolved.append(hit)    # unresolved links are KEPT
            return f"[[{hit}]]"
        report.resolved.append((hit, target))
        # Obsidian's resolver does not consult frontmatter aliases - confirmed
        # by the Obsidian team as intentional. The pipe form is mandatory.
        return f"[[{target}]]" if hit == target else f"[[{target}|{hit}]]"

    return pat.sub(repl, body), report


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

_FM = re.compile(r"\A---\n(.*?)\n---\n?", re.S)


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """(frontmatter dict, body). Missing or malformed frontmatter -> ({}, text).

    Deliberately a flat `key: value` reader and not a YAML parser: the fields
    this system owns are all scalars, and a dependency here would be paid on
    every sync.
    """
    m = _FM.match(text)
    if not m:
        return {}, text
    fm: dict[str, str] = {}
    for line in m.group(1).split("\n"):
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip().strip("'\"")
    return fm, text[m.end():]


def page_hash(text: str) -> str:
    """`content_hash` for a whole page file: body only, frontmatter EXCLUDED.

    Found by placing the three hash uses side by side. The analysis pass writes
    `summary:` into frontmatter; if the hash covered frontmatter, that write
    would move it, every outstanding `expect` would go stale and the model's
    next write would be refused although it edited nothing - the same failure
    the link-stripping fix removed. Frontmatter is metadata, not content.

    Consequence, and it is the right one: editing `type:` by hand does not
    queue the page for re-analysis, because it did not change what the page
    says.
    """
    _, body = split_frontmatter(text)
    return content_hash(body)


# Frontmatter key order, so a rewrite never produces a spurious diff.
FM_ORDER = ["type", "tags", "summary", "summary_provisional", "origin",
            "created", "updated"]


# YAML characters that make an unquoted scalar ambiguous or invalid.
_NEEDS_QUOTES = re.compile(r": |:$|^[-?\s>|&*!%@`\[\]{}#,'\"]|#\s|\s$|^$")


def yaml_value(v) -> str:
    """A frontmatter value that YAML can read back.

    Measured on a real page: a summary reading "...three directions for deeper
    exploration: mathematical formulas..." contains `: `, which ends the key on
    that line. Obsidian could not parse the block at all and rendered the
    entire frontmatter as a code fence - the page still held its content, but
    every property vanished from the panel.

    Summaries are written by a model and will contain colons, hashes, quotes
    and leading dashes sooner or later, so the fix belongs here rather than in
    a prompt asking the model to avoid punctuation.
    """
    text = str(v)
    if not _NEEDS_QUOTES.search(text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_page(fm: dict, body: str) -> str:
    keys = [k for k in FM_ORDER if k in fm] + [k for k in fm if k not in FM_ORDER]
    lines = "\n".join(f"{k}: {yaml_value(fm[k])}" for k in keys)
    return f"---\n{lines}\n---\n\n{body.lstrip()}"


def first_sentence(body: str, cap: int = 300) -> str:
    """The cheapest possible summary, and the only one that CANNOT fabricate.

    Measured about half useful on real pages - one page got a good line, the
    other got the opening of a figure caption - which is the concrete case for
    the analysis pass being mandatory rather than an improvement. It is a
    placeholder that keeps the page visible to the cheap path, nothing more.
    """
    body = split_frontmatter(body)[1]      # tolerate a whole file being passed
    text = re.sub(r"^#{1,6}[^\n]*$", "", strip_links(body), flags=re.M)
    text = " ".join(text.split())
    m = re.search(r"(?<=[.!?])\s", text)
    out = (text[:m.start()] if m else text)[:cap].strip()
    return out or "(no text yet)"
