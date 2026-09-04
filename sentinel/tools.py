"""The Sentinel - the primitives. Read side.

Seven tools, not thirty-two. Tool count is an ACCURACY problem and not only a
token one: production accuracy degrades past 15-20 tools in rotation, and on
the Berkeley Function Calling Leaderboard accuracy fell 43% -> 2% when the
tool count went 4 -> 51. The failure mode is not "I don't know which tool" -
the model picks a plausible wrong one, or fills arguments borrowed from a
DIFFERENT tool's schema.

Every `depth` and `by` here is DEPTH OF ONE INTENT, which is the only form of
mode parameter that survives the literature. A combined find(query, kind) was
tried and failed, because semantic retrieval and deterministic enumeration are
two different intents; they are `search` and `listing`.

Defaults matter more than they look. Router designs fail through routing
error, and a wrong rung degrades everything downstream, so every parameter
here has the common case as its default and the model rarely has to choose.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from functools import wraps

from .index import parse_page
from .output import BUDGET, health, opaque, ranked, sequential, split_parts
from .search import format_hits
from .search import search as _search
from .text import (
    first_sentence,
    normalize,
    render_page,
    split_frontmatter,
)

_HEADING_LINE = re.compile(r"^(#{1,6})\s*(.+)$", re.M)

SUMMARY_DISPLAY = 160

# References a concept needs before its absence is worth mentioning unasked.
#
# The health line's rule is to speak when something is ABNORMAL and to cost
# nothing otherwise. A concept with no page is not abnormal - most of them
# never get one, and saying "12 concepts are waiting" every turn is a line
# that stops being read. What IS worth a sentence is a concept the vault keeps
# leaning on and still cannot define: measured on the real vault, `GRB` was
# referenced by six pages with no page of its own, while `JSON` was referenced
# by one. The first is a gap; the second is noise.
GAP_REFERENCES = 4

# Fields code owns and the model may never set. `summary` has exactly one
# writer, the analysis pass; the rest are facts about the write itself, which
# only code is in a position to know.
OWNED_FIELDS = {"summary", "summary_provisional", "origin", "created",
                "updated"}

# A repeated search, caught in the TOOL rather than in the loop, because in
# Open WebUI the harness does not run - that loop belongs to Open WebUI.
# Measured there, the model searched ten times in one turn, each query a
# reworded version of the last.
#
# THE GUARD WARNS; IT DOES NOT BLOCK. A first version returned [STOP] on a
# near-repeat, and it was worse than the problem: the tool object outlives a
# conversation, so a query from the PREVIOUS conversation blocked the first
# search of the next one. Starved of results, the model reported that the
# search had found a note on the subject. It had found nothing, because it
# had not searched.
#
# Blocking a read is how a model ends up inventing. So a near-repeat returns
# the results AND says they are the same ones; only a sustained run is
# refused, by which point the model is demonstrably not reading what it gets.
REPEAT_WINDOW = 120.0        # seconds
REPEAT_OVERLAP = 0.6         # of the shorter query's content words
REPEAT_LIMIT = 5             # near-repeats before the tool does refuse


def _query_key(query: str) -> frozenset:
    return frozenset(normalize(w) for w in re.findall(r"\w+", query.lower())
                     if len(w) > 2)


def _overlap(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def origin_mark(row) -> str:
    """`(from conversation)` where a page's claims came from talking rather
    than from a source that can be re-read.

    Two independent projects working on this problem arrived at the same
    device - a flag marking agent-written content, kept for trust calibration -
    and the reasoning holds here: a claim settled in conversation is
    legitimate and is NOT the same kind of thing as a claim taken from a
    paper. Marking it costs a few tokens and appears only where it applies.
    """
    try:
        return " (from conversation)" if row["origin"] == "conversation" else ""
    except (IndexError, KeyError):
        return ""


def _short(summary: str | None) -> str:
    """Cap a summary line in RANKED output.

    Measured: ten hits cost 2566 characters, about 64 tokens each, against a
    design target of ~15. The cause is that a provisional summary is the
    page's first sentence verbatim, and on a paper that runs to 300
    characters. The analysis pass fixes this properly by writing a real one;
    until it has run, the map the model reads on every search should not cost
    four times what it was sized for.

    The ellipsis is deliberate - a cut that looks complete is the defect this
    system keeps guarding against.
    """
    if not summary:
        return "-"
    s = " ".join(summary.split())
    return s if len(s) <= SUMMARY_DISPLAY else s[:SUMMARY_DISPLAY - 1] + "…"

def _locked(fn):
    """Serialise a whole primitive against the index.

    The lock spans the entire call because a primitive is a sequence - refresh
    the index, then read it - and a half-applied sequence is exactly the
    partial refresh the design refuses to treat as a warning.
    """
    @wraps(fn)
    def inner(self, *a, **kw):
        with self.index.lock:
            return fn(self, *a, **kw)
    return inner


class Sentinel:
    def __init__(self, index, encoder=None, origin: str = "conversation",
                 pages: str = "wiki"):
        self.index = index
        self.encoder = encoder
        # `origin` is code's to set because only code knows which path a write
        # arrived on: `conversation` while talking, `user` written directly,
        # later `ingestion`.
        self.origin = origin
        # Where a page with no folder in its path goes. The model was writing
        # to the vault root because nothing said otherwise, and a root that
        # fills with pages loses the one place a person looks first.
        self.pages = pages.strip("/")
        # Keyed by scope so one conversation's searches never speak for
        # another's. A caller that knows the conversation sets `scope`; one
        # that does not shares a bucket, which is why the guard only warns.
        self._searches: dict[str, list[tuple[float, frozenset, str]]] = {}
        self.scope = "default"

    # -- shared ----------------------------------------------------------

    def _sync(self) -> None:
        """Bring the index up to date. Called at the TOP of every primitive.

        Position is load-bearing: the health line is appended at the END of a
        response, so syncing there would refresh the index only in time for
        the NEXT call, and the current one would answer from stale rows. That
        defect was real and is what this split fixes.

        Every primitive calls this, so the index is never more than one tool
        call behind the vault. Measured at 300 pages a no-op walk is ~12 ms,
        against a conversational turn measured in seconds - cheap enough that
        no TTL is needed, and a TTL would only be a staleness window with a
        number on it.

        For `write` this is not a nicety but a CORRECTNESS requirement: the
        `expect` guard compares against the index's content_hash, so a stale
        index would compare against a hash the file no longer has and wave
        through exactly the destructive overwrite the guard exists to stop.
        """
        self._notes = self.index.health_notes()

    def _health(self) -> str:
        """Report only - the walk already happened in _sync."""
        notes = list(getattr(self, "_notes", []))
        pending = len(self.index.pending_analysis())
        if pending:
            notes.append(f"{pending} pages awaiting analysis")
        try:
            failed = self.index.db.execute(
                "SELECT COUNT(*) c FROM analysis WHERE status='failed'"
            ).fetchone()["c"]
            if failed:
                notes.append(f"{failed} pages failed analysis")
        except Exception:
            pass          # the table appears with the first analysis pass
        try:
            rows, _ = self.index.growth_queue(3)
            gaps = [r for r in rows if r["n"] >= GAP_REFERENCES]
            if gaps:
                named = ", ".join(f"{r['display']} ({r['n']})" for r in gaps)
                notes.append(f"referenced but still undefined: {named}")
        except Exception:
            pass
        try:
            from .concepts import bloated
            big = bloated(self.index)
            if big:
                # Reported, never acted on. Whether a page has become two
                # subjects is a judgement about the subject.
                notes.append(f"{len(big)} concept pages have grown past "
                             f"{max(n for _, n in big)} sections and may want "
                             f"splitting")
        except Exception:
            pass
        return health(notes)

    def _resolve(self, path_or_name: str):
        """A path OR a bare page name. `graph` prints page names while `read`
        wanted paths, and bridging that gap by hand was a hidden step for the
        model. Fuzzy resolution for reads, exact paths for writes."""
        if self.index.meta(path_or_name):
            return [path_or_name]
        return self.index.resolve(path_or_name)

    def write_path(self, path: str) -> str:
        """The path a write will actually land on.

        Callers that need to look at the target BEFORE writing have to resolve
        it the same way `write` does, and the way to guarantee that is to ask
        rather than to repeat the rule. Measured: the tool layer checked the
        page for an existing record link using the path as the model gave it,
        `settled decisions.md`, while the write itself put the page in
        `wiki/`. The check looked at a file that did not exist, concluded
        there was no link, and added a second one.
        """
        if "/" not in path and self.pages:
            return f"{self.pages}/{path}"
        return path

    def _is_record(self, path: str) -> bool:
        """A transcript, or a new page in the folder where transcripts live.

        The type is the real test; the folder catches a page that does not
        exist yet, which is when there is no type to read.
        """
        row = self.index.meta(path)
        if row and row["type"] == "transcript":
            return True
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        if not folder:
            return False
        return bool(self.index.db.execute(
            "SELECT 1 FROM files WHERE type='transcript' AND path LIKE ? "
            "LIMIT 1", (folder + "/%",)).fetchone())

    def _body(self, path: str) -> str:
        raw = (self.index.vault / path).read_text(encoding="utf-8")
        return split_frontmatter(raw)[1]

    # -- read ------------------------------------------------------------

    @_locked
    def read(self, path: str, depth: str = "meta", target: str | None = None,
             from_part: int = 1) -> str:
        """Read one note. Accepts a path (`wiki/afterglow.md`) or a bare page
        name (`afterglow`).

        depth:
          meta    (default) frontmatter, tags and link counts. ~30 tokens.
                  SKIP THIS AFTER `search` - you already hold the summary and
                  reading meta pays for it twice. It is for a path that came
                  from `graph` or `listing`.
          outline the heading tree with the size of each section. Use this to
                  choose a section before reading any prose.
          part    one section, named by `target` (a heading).
          window  the text around `target` (a word or phrase).
          full    the whole page, split into parts; `from_part` selects which.

        target:    a heading (depth=part) or a word to centre on (depth=window).
                   Ignored by the other depths.
        from_part: which part of a split read to return. Starts at 1.

        Use `search` instead when you do not yet know which page you want.
        """
        self._sync()
        hits = self._resolve(path)
        if not hits:
            # If the name is in the growth queue, say so. Measured: a model
            # read `graph`'s vault shape as a list of pages and tried to open
            # six of them in a row. The information to explain the refusal was
            # already in the index.
            n = self.index.db.execute(
                "SELECT COUNT(*) c FROM links WHERE target_key=? AND resolved=0",
                (normalize(path),)).fetchone()["c"]
            if n:
                return (f"[STOP] '{path}' has no page. {n} pages link to it, "
                        f"which is why `graph` lists it as something to write. "
                        f"There is nothing to read yet.")
            return f"[STOP] no page named '{path}'"
        if len(hits) > 1:
            listed = "\n".join(f"  {h}" for h in hits)
            return f"[RETRY] '{path}' matches {len(hits)} pages:\n{listed}"
        p = hits[0]
        row = self.index.meta(p)

        if depth == "meta":
            tags = [r["tag"] for r in self.index.db.execute(
                "SELECT tag FROM tags WHERE path=?", (p,))]
            out_n = len(self.index.links_out(p))
            _, in_n = self.index.links_in(row["page_key"], limit=1)
            mark = " (provisional)" if row["summary_provisional"] else ""
            lines = [f"[OK] {p}  ({row['type'] or 'untyped'}"
                     f"{', from conversation' if row['origin'] == 'conversation' else ''})",
                     f"  summary{mark}: {row['summary'] or '-'}"]
            if tags:
                lines.append(f"  tags: {', '.join(tags)}")
            lines.append(f"  links out: {out_n}, linked from: {in_n}")
            # The `expect` guard needs a hash the model can hand back, so the
            # read has to hand one out. Eight characters: enough to be unique
            # across a vault, short enough not to be a line of its own cost.
            lines.append(f"  expect: {row['content_hash'][:8]}")
            return "\n".join(lines) + self._health()

        body = self._body(p)

        if depth == "outline":
            rows = []
            marks = list(_HEADING_LINE.finditer(body))
            for i, m in enumerate(marks):
                end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
                rows.append(f"  {'  ' * (len(m.group(1)) - 1)}{m.group(2)}"
                            f"  ({end - m.start()} chars)")
            if not rows:
                return f"[OK] {p} has no headings ({len(body)} chars)\n" \
                       f"[OK -> read] read(path=\"{p}\", depth=\"full\")"
            return (f"[OK] {p} outline, {len(marks)} sections\n"
                    + "\n".join(rows)
                    + f"\n[OK -> read] read(path=\"{p}\", depth=\"part\", "
                      f"target=\"<heading>\")")

        if depth == "part":
            if not target:
                return "[RETRY] depth=\"part\" needs target=<a heading>"
            marks = list(_HEADING_LINE.finditer(body))
            for i, m in enumerate(marks):
                if normalize(m.group(2)) == normalize(target):
                    end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
                    text = body[m.start():end]
                    break
            else:
                names = ", ".join(m.group(2) for m in marks[:8]) or "none"
                return f"[RETRY] no heading '{target}' in {p}. Headings: {names}"
        elif depth == "window":
            if not target:
                return "[RETRY] depth=\"window\" needs target=<a word or phrase>"
            m = re.search(re.escape(target), body, re.I)
            if not m:
                return f"[STOP] '{target}' does not appear in {p}"
            half = BUDGET // 2
            text = body[max(0, m.start() - half): m.start() + half]
        elif depth == "full":
            text = body
        else:
            hint = ""
            if depth in ("summary", "body"):
                # Measured on the first real conversation: the model called
                # read(depth="summary"), borrowing `search`'s enum for a
                # parameter of the same name. That is the documented failure -
                # not "I don't know which tool" but arguments filled from a
                # DIFFERENT tool's schema. Naming the confusion turns a wasted
                # call into a recovery.
                hint = (f' - "{depth}" is a `search` depth, not a `read` one. '
                        f'For a cheap look at one page use depth="meta".')
            return ("[RETRY] depth must be one of: meta, outline, part, "
                    "window, full" + hint)

        parts = split_parts(text)
        i = max(1, min(from_part, len(parts)))
        chunk, hard = parts[i - 1]
        nxt = (f'read(path="{p}", depth="{depth}", '
               + (f'target="{target}", ' if target else "")
               + f"from_part={i + 1})") if i < len(parts) else None
        return sequential(f"{p} {depth} [expect {row['content_hash'][:8]}]",
                          chunk, i, len(parts), nxt, hard) + self._health()

    # -- search ----------------------------------------------------------

    @_locked
    def search(self, query: str, depth: str = "summary", limit: int = 5) -> str:
        """Find pages by meaning AND by wording, fused. Use this whenever you
        do not already know the page name.

        depth:
          summary (default) one line per hit saying what that page is. Five
                  hits cost about 75 tokens. THIS IS THE NORMAL STOPPING
                  PLACE - go further only if the summaries showed a page is
                  relevant but did not contain the answer.
          body    verbatim passages from the hits, capped per file.
        limit:    how many pages to return. 5 is right for a conversation.

        Results are RANKED, and rank is the only signal - no similarity score
        is shown because none is meaningful here. If no page matched the
        wording, the output says so; treat those hits as unconfirmed.

        Use `listing` instead to enumerate by an exact attribute (a tag, a
        folder, recency) - that is not a search.
        """
        self._sync()

        if depth not in ("summary", "body"):
            hint = ""
            if depth in ("meta", "outline", "part", "window", "full"):
                # The same borrowing, in the other direction. Both messages
                # exist because the collision is symmetric: two tools share a
                # parameter NAME and not its values.
                hint = f' - "{depth}" is a `read` depth, not a `search` one.'
            return '[RETRY] depth must be "summary" or "body"' + hint

        # AFTER validating the call. A malformed call is not a search: it
        # never reached the index, so counting it would spend the budget on
        # something that returned nothing and block the corrected retry.
        now = time.monotonic()
        recent = [x for x in self._searches.get(self.scope, [])
                  if now - x[0] < REPEAT_WINDOW]
        key = _query_key(query)
        repeats = [q for _, prev, q in recent
                   if _overlap(key, prev) >= REPEAT_OVERLAP]
        recent.append((now, key, query))
        self._searches[self.scope] = recent

        if len(repeats) >= REPEAT_LIMIT:
            return (f"[STOP] {len(repeats)} rewordings of this query already, "
                    f"all returning the same pages. If none of the summaries "
                    f"were about what you asked, the vault does not contain "
                    f"it - say so rather than searching again.")
        repeat_note = (f"\n[note] nearly the query you just ran ('{repeats[-1]}'), "
                       f"so these are the same pages. Rewording will not change "
                       f"them." if repeats else "")
        hits = _search(self.index, query, encoder=self.encoder, depth=depth,
                       limit=limit)
        return format_hits(hits, query, depth) + repeat_note + self._health()

    # -- listing ---------------------------------------------------------

    @_locked
    def listing(self, by: str, value: str | None = None, limit: int = 20) -> str:
        """Enumerate pages by an exact attribute. Deterministic, not ranked by
        relevance.

        by:
          tag      pages carrying `value` as a tag
          path     pages under the folder `value` (recursive)
          recent   most recently changed pages; `value` is ignored
          all_tags every tag in the vault with a count; `value` is ignored
        value: the tag or folder. Required for `tag` and `path`.
        limit: how many rows. 20 by default.

        Use `search` instead when you are looking by meaning rather than by an
        attribute you can name exactly.
        """
        self._sync()
        if by == "all_tags":
            rows, total = self.index.all_tags(limit)
            body = [f"  {r['tag']} ({r['n']})" for r in rows]
            return ranked("all tags", len(rows), total, body) + self._health()
        if by == "recent":
            rows, total = self.index.recent(limit)
        elif by == "tag":
            if not value:
                return '[RETRY] by="tag" needs value=<a tag>'
            rows, total = self.index.by_tag(value, limit)
        elif by == "path":
            if not value:
                return '[RETRY] by="path" needs value=<a folder>'
            rows, total = self.index.by_path(value, limit)
        else:
            return ("[RETRY] by must be one of: tag, path, recent, all_tags")
        if not rows:
            return f"[STOP] nothing matches {by}={value!r}"
        body = [f"  {r['path']}"
                f"{' (provisional summary)' if r['summary_provisional'] else ''}"
                f"{origin_mark(r)}"
                f": {_short(r['summary'])}" for r in rows]
        head = f"{by}={value}" if value else by
        return ranked(head, len(rows), total, body) + self._health()

    # -- graph -----------------------------------------------------------

    @_locked
    def graph(self, subject: str | None = None, limit: int = 10) -> str:
        """Link structure. With no `subject`, the shape of the whole vault.

        subject: a page name or a concept, resolved or not.
        limit:   how many incoming links to show. 10 by default.

        Grouped by what you would DO with each group rather than by a status
        flag: a link whose page exists is something to read, a link with no
        page yet is something to write.

        Asking about a concept that has no page is NORMAL here - unresolved
        links are the growth queue, not errors.
        """
        self._sync()
        # An omitted subject and an empty one mean the same thing. Measured:
        # asked which concepts had no page yet, the model chose exactly the
        # right call - `graph` with no subject - and sent subject="". That
        # fell through to the lookup path and came back [STOP], so the one
        # question the growth queue exists to answer could not be asked.
        if not subject or not subject.strip():
            rows, total = self.index.growth_queue(8)
            # One per line, with the count in brackets. The comma-joined
            # `name n` form was measured to be unparseable: concept names
            # contain spaces, so `GRB 6, isotropic energy 5` reads as a
            # concept called "GRB 6" - and that is exactly what came back,
            # reported with an invented count of 8.
            listed = "\n".join(f"    {r['display']} ({r['n']} pages)"
                               for r in rows)
            rest = max(total - len(rows), 0)
            # Naming the remainder stops it being filled in from elsewhere.
            # Measured: the model followed this output with seven further
            # "unresolved concepts" taken from earlier turns, not from here.
            tail = (f"\n[note] the other {rest} are not listed; do not name "
                    f"them from memory" if rest else "")
            # The heading says NO PAGE EXISTS, and says it before the names.
            # An earlier version led with "to write next", and measured on a
            # real run the model spent eight of its twenty calls trying to
            # `read` the entries - first by bare name, then guessing at
            # `wiki/<name>` - and got a refusal every time. The list was
            # correct; what it did not say was that these are the absences.
            return ("[OK] vault link shape\n"
                    f"  NAMES WITH NO PAGE, most referenced first - showing "
                    f"{len(rows)} of {total}. These cannot be read; they are "
                    f"what there is to write:\n{listed}\n"
                    f"  orphan pages: {self.index.orphans()}"
                    + tail) + self._health()

        # A PATH or a bare name, the same as `read`. Measured: `search`
        # returns paths, and handing one straight to `graph` came back [STOP] -
        # so the obvious next step after a search was a dead end and the model
        # had to translate a path back into a name by itself.
        paths = self._resolve(subject)
        key = (self.index.meta(paths[0])["page_key"] if paths
               else normalize(subject))
        rows_in, total_in = self.index.links_in(key, limit)

        if not paths and not total_in:
            return f"[STOP] '{subject}' has no page and nothing links to it"

        out = []
        if paths:
            row = self.index.meta(paths[0])
            out.append(f"[OK] {subject}  ({row['type'] or 'untyped'})")
            exists, missing = [], []
            for r in self.index.links_out(paths[0]):
                (exists if r["resolved"] else missing).append(r["display"])
            if exists:
                out.append(f"  links to, page exists ({len(exists)}): "
                           + ", ".join(exists))
            if missing:
                out.append(f"  links to, no page yet ({len(missing)}): "
                           + ", ".join(missing))
        else:
            # No page but referenced: the growth-queue view, and [OK] because
            # this is a normal question, not a failure.
            out.append(f"[OK] {subject} - no page yet, referenced by "
                       f"{total_in} pages")

        if rows_in:
            more = f"; {total_in - len(rows_in)} more not shown" \
                if total_in > len(rows_in) else ""
            out.append(f"  linked from ({total_in}{more}):")
            for r in rows_in:
                stem = r["source"].rsplit("/", 1)[-1][:-3]
                # One per line. Comma-joining names that carry summaries -
                # which themselves contain commas and full stops - produced a
                # single 700-character line with no structure in it. Same
                # defect as the vault-shape list, and the same fix.
                #
                # Titles only where the name is opaque: `arxiv-2504.11743`
                # says nothing, `gamma-ray burst` says what it is. Measured,
                # full summaries on every entry pay off only if the model
                # opens more than three of five, and it opens nought or one.
                if opaque(stem) and r["summary"]:
                    out.append(f"    {stem}\n      {_short(r['summary'])}")
                else:
                    out.append(f"    {stem}")
        return "\n".join(out) + self._health()

    # =====================================================================
    # The write side
    #
    # Reading only reads; writing TRIGGERS. When a page changes, its summary
    # goes stale, its chunks go stale, its links change. The earlier system
    # wrote and did nothing else, which is why index.md drifted, [[[[JWST]]]]
    # accumulated, and a stale summary was not even a concept.
    #
    # `write` CONTAINS NO MODEL CALL. The page is written now, unlinked;
    # linking is a separate pass. Calling the analysis model at write time
    # would mean a VRAM swap per write - fine in a batch, ruinous mid
    # conversation. An unlinked page is not broken, just not yet woven in,
    # exactly as an unresolved link is not an error.
    # =====================================================================

    def _refresh(self, path: str) -> str | None:
        """Returns None on success, or a loud failure line.

        PARTIAL REFRESH IS A FAILURE, NOT A WARNING. If the write lands but
        the index does not, the index lies and every later search is quietly
        wrong, so the refresh is part of the write's success criteria.
        """
        try:
            f = self.index.vault / path
            if f.exists():
                self.index._replace_page(
                    parse_page(path, f.read_text(encoding="utf-8")),
                    f.stat().st_mtime)
            else:
                for t in ("links", "chunks", "chunks_fts", "tags"):
                    col = "source" if t == "links" else "path"
                    self.index.db.execute(f"DELETE FROM {t} WHERE {col}=?", (path,))
                self.index.db.execute("DELETE FROM files WHERE path=?", (path,))
            self.index.refresh_resolved()
            self.index.db.commit()
            return None
        except Exception as e:
            self.index.db.rollback()
            return (f"[STOP] {path} was written to disk but the index refresh "
                    f"FAILED ({e}). Search results are now unreliable until "
                    f"the index is rebuilt.")

    @_locked
    def write(self, path: str, content: str, where: str = "section",
              target: str | None = None, expect: str | None = None) -> str:
        """Put content into a note. Exact paths only - no name resolution here.

        USE THIS WHEN THE USER ASKS YOU TO RECORD SOMETHING - "kaydet", "save
        this", "write that down". Not on your own judgement of what matters.

        When they ask, do it in ONE call. Do not search first, do not ask
        which page, do not describe what you are about to write. If they name
        a page, use it; otherwise choose a name and say which one you used.

        Deciding unprompted was tried and does not work. Across five phrasings
        of the instruction the model would restate the rule correctly in its
        own reasoning and then end the turn with an offer instead of a call.
        Waiting for the user costs nothing, because the alternative was
        nothing happening.

        WHERE IT GOES: `wiki/<name>.md`. Nothing in the vault root - measured,
        a page was written there because no instruction said otherwise, and a
        vault whose root fills up with pages loses the one place a person
        looks first.

        WRITE WHAT THE CONVERSATION SETTLED, NOT WHAT YOU KNOW ABOUT THE
        SUBJECT.

        If they name what to record, record that. If they only say "kaydet",
        write what was worked out in the conversation - including a
        conclusion you drew from what the vault told you, which is a real
        result of the exchange. What does not belong is the rest of what you
        know about the topic.

        Measured: asked to record a conversation about the Van de Graaff
        generator, the model produced an encyclopaedia entry - the 1929 date,
        the capacitance formula, the 3 MV/m breakdown field, a section on
        building one. None of it had been discussed. The page carries the
        user's own origin marking, so six months on it reads as something
        they established.

        The line is the source, not the speaker: what came from the
        conversation or from the vault belongs, what came from your training
        does not.

        Write it the way you understood it - organised, with headings, in your
        own arrangement. The concepts come from the conversation record, not
        from this page, so shaping the page costs nothing.

        TWO THINGS DO NOT BELONG. No task list: three ticked boxes read as a
        commitment the user made, and six months later they cannot tell it
        from their own plan. No number that was not said: a figure you worked
        out reads as a measurement, because everything around it is one. A
        DATE IS A NUMBER - you do not know today's, and the system writes it
        into the page for you, so never put one in the text.

        where:
          section (default) replace the section named by `target`
          whole   replace the whole body
          end     append to the end of the body
          frontmatter set the field named by `target` to `content`
        target: the heading (where=section) or the field (where=frontmatter).
        expect: the `expect` value from your last read of this page, or "new"
                for a page that does not exist yet. REQUIRED.

        If the file changed since you read it, the write is refused so you can
        merge rather than overwrite something you never saw. The same guard
        makes a repeated identical call harmless.

        On a page that does not exist, `where` is ignored and the content
        becomes the page.

        Links are NOT placed here. That is the analysis pass, and a page
        without them is not broken.
        """
        if not expect:
            return ('[RETRY] expect is required: pass the value from your last '
                    'read of this page, or "new"')
        if where not in ("section", "whole", "end", "frontmatter"):
            return ("[RETRY] where must be one of: section, whole, end, "
                    "frontmatter")

        self._sync()               # before the expect comparison, not after
        path = self.write_path(path)

        # A CONVERSATION RECORD IS NOT YOURS TO EDIT.
        #
        # It is written by code, from the messages themselves, precisely so
        # that it is a record and not an account. Measured: asked to add two
        # findings, the model laid out a plan whose second step was to
        # reconstruct the transcript "entirely with the additions" - replacing
        # what was said with what it remembered being said. Every concept in
        # the vault is gated against that file.
        #
        # [STOP], not [RETRY]: there is no phrasing of this that works.
        if self._is_record(path):
            return (f"[STOP] {path} is a conversation record, written by the "
                    f"system from the messages themselves. It cannot be "
                    f"edited. Write the finding to a page in "
                    f"{self.pages or 'wiki'}/ instead.")
        f = self.index.vault / path
        exists = f.exists()
        row = self.index.meta(path)

        if exists and expect == "new":
            # A page written from conversation GROWS when the conversation
            # comes back to it. Refusing here would send the model down a
            # three-step path - read, take the hash, write the whole body -
            # which it does not do reliably, so in practice the second finding
            # would land in a second page or nowhere.
            #
            # Appending, not overwriting: whatever was written the first time
            # is still true, and whatever the user has edited into the page
            # survives. The same shape a concept page grows in.
            if (row["origin"] or "") == "conversation":
                # A repeated identical call must still do nothing. Appending
                # made `expect` stop guarding that, so the check moves here:
                # if the text is already on the page, the write has already
                # happened.
                if " ".join(content.split()) in " ".join(
                        self._body(path).split()):
                    return (f"[DONE] {path} already says this "
                            f"(expect {row['content_hash'][:8]})")
                where, expect = "end", row["content_hash"][:8]
                if not content.lstrip().startswith("#"):
                    # The heading is code's: the model does not know today's
                    # date and should not be writing one.
                    content = (f"## {datetime.now(timezone.utc):%Y-%m-%d}\n\n"
                               + content.strip())
            else:
                return (f"[RETRY] {path} already exists (expect "
                        f"{row['content_hash'][:8]} if you meant to change it)")
        if not exists and expect != "new":
            return f'[RETRY] {path} does not exist - pass expect="new" to create it'
        if exists and not row["content_hash"].startswith(expect):
            head = self._body(path)[:1200]
            return (f"[RETRY] {path} changed since you read it. It is now "
                    f"{row['content_hash'][:8]}, you passed {expect}. Current "
                    f"opening:\n{head}")

        if not exists:
            # Content may lead with its own frontmatter, so `type` and `tags`
            # can be set as the page is created rather than in a second call.
            # It matters: a concept page created as `type: note` never gets
            # the precedence the resolution index gives concepts, and until
            # now EVERY page a model created was typed note.
            given, given_body = split_frontmatter(content)
            owned = OWNED_FIELDS & set(given)
            if owned:
                return ("[STOP] " + ", ".join(sorted(owned)) + " "
                        + ("is" if len(owned) == 1 else "are")
                        + " set by the system, not by you. Leave "
                        + ("it" if len(owned) == 1 else "them") + " out.")
            fm = {"type": "note", **given}
            body = (given_body if given else content).strip() + "\n"
        else:
            fm, body = split_frontmatter(f.read_text(encoding="utf-8"))
            if where == "whole":
                body = content.strip() + "\n"
            elif where == "end":
                body = body.rstrip() + "\n\n" + content.strip() + "\n"
            elif where == "frontmatter":
                if not target:
                    return '[RETRY] where="frontmatter" needs target=<a field>'
                if target == "summary":
                    # `summary` has exactly ONE writer. Two authors - a manual
                    # write and the analysis pass - means the pass silently
                    # overwrites a deliberate one. Retrying never fixes that,
                    # so this is [STOP].
                    return ("[STOP] summary is written by the analysis pass, "
                            "not by hand")
                fm[target] = content.strip()
            else:
                if not target:
                    return '[RETRY] where="section" needs target=<a heading>'
                marks = list(_HEADING_LINE.finditer(body))
                for i, m in enumerate(marks):
                    if normalize(m.group(2)) == normalize(target):
                        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
                        body = body[:m.start()] + content.strip() + "\n\n" + body[end:]
                        break
                else:
                    names = ", ".join(m.group(2) for m in marks[:8]) or "none"
                    return f"[RETRY] no heading '{target}' in {path}. Headings: {names}"

        # Code's fields, not the model's. It has no clock, and only code knows
        # which path this write arrived on.
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fm.setdefault("created", now)
        fm["updated"] = now
        fm.setdefault("origin", self.origin)

        # A page can NEVER be written without a summary: without one it is
        # invisible to the cheap path and gets opened in full whenever it is a
        # candidate. The first sentence cannot fabricate, so it is the safe
        # provisional value until the analysis pass replaces it.
        if not fm.get("summary"):
            fm["summary"] = first_sentence(body)
            fm["summary_provisional"] = "1"

        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(render_page(fm, body), encoding="utf-8")

        failure = self._refresh(path)
        if failure:
            return failure
        new = self.index.meta(path)
        lines = [f"[DONE] {'created' if not exists else 'added to'} {path} "
                 f"(expect {new['content_hash'][:8]})"]
        if not exists:
            lines += self._naming(path)
        if new["summary_provisional"]:
            lines.append("[note] provisional summary; page is queued for "
                         "analysis")
        return "\n".join(lines) + self._health()

    def _naming(self, path: str) -> list[str]:
        """What a new filename did, and what it may have missed.

        Only on creation. The filename is the canonical name, so this is the
        one moment where the choice is still cheap to change.
        """
        from .resolve_queue import on_page_created
        out = []
        r = on_page_created(self.index, path)
        for display, n in r["merged"]:
            out.append(f"[note] '{display}' resolves here now - {n} earlier "
                       f"links came alive, no file rewritten")
        for display, n in r["near"]:
            out.append(f"[note] {n} links are waiting for the spelling "
                       f"'{display}', which this name does not match. "
                       f"relocate() if that is the page they meant.")
        return out

    @_locked
    def relocate(self, path: str, to: str) -> str:
        """Rename or move a note, redirecting every link that points at it.

        An alias entry would look like the elegant fix and is WRONG: Obsidian's
        resolver does not consult frontmatter aliases - confirmed by the
        Obsidian team as intentional - so the user would see broken links in
        the UI even though this system resolved them fine.

        Incoming links are rewritten to the pipe form, which keeps the words
        on the page exactly as they were. Their content hashes therefore do
        not move and none of those pages is queued for re-analysis.
        """
        self._sync()
        src, dst = self.index.vault / path, self.index.vault / to
        if not src.exists():
            return f"[STOP] no page at {path}"
        if dst.exists():
            return f"[RETRY] {to} already exists"

        old_key = self.index.meta(path)["page_key"]
        new_name = to.rsplit("/", 1)[-1][:-3]
        sources = [r["source"] for r in self.index.db.execute(
            "SELECT DISTINCT source FROM links WHERE target_key=?", (old_key,))]

        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)

        rewritten = 0
        for s in sources:
            sf = self.index.vault / s
            if not sf.exists():
                continue
            text = sf.read_text(encoding="utf-8")
            def sub(m):
                inner = m.group(1)
                canon, _, disp = inner.partition("|")
                if normalize(canon) != old_key:
                    return m.group(0)
                return f"[[{new_name}|{disp or canon}]]"
            new_text = re.sub(r"\[\[([^\]]+)\]\]", sub, text)
            if new_text != text:
                sf.write_text(new_text, encoding="utf-8")
                rewritten += 1

        for p in [to, *sources]:
            failure = self._refresh(p)
            if failure:
                return failure
        self.index.db.execute("DELETE FROM files WHERE path=?", (path,))
        for t, col in (("links", "source"), ("chunks", "path"),
                       ("chunks_fts", "path"), ("tags", "path")):
            self.index.db.execute(f"DELETE FROM {t} WHERE {col}=?", (path,))
        self.index.refresh_resolved()
        self.index.db.commit()
        return "\n".join([f"[DONE] {path} -> {to}, {rewritten} pages had "
                          f"links redirected", *self._naming(to)]) \
            + self._health()

    @_locked
    def remove(self, path: str) -> str:
        """Delete a note.

        Links pointing AT it are left alone: they become unresolved and rejoin
        the growth queue, which is correct, since the concept is still
        referenced - it just has no page again.
        """
        self._sync()
        f = self.index.vault / path
        if not f.exists():
            return f"[STOP] no page at {path}"
        key = self.index.meta(path)["page_key"]
        f.unlink()
        failure = self._refresh(path)
        if failure:
            return failure
        orphaned = self.index.db.execute(
            "SELECT COUNT(*) c FROM links WHERE target_key=?", (key,)).fetchone()["c"]
        tail = (f"; {orphaned} links to it are now unresolved and back in the "
                f"growth queue") if orphaned else ""
        return f"[DONE] removed {path}{tail}" + self._health()
