"""The Sentinel - the derived index.

Three of the four read primitives were doing a vault walk per call. Everything
they need fits here, in ONE SQLite alongside the embeddings.

DERIVED, NEVER AUTHORITATIVE. A full rebuild by scanning the vault must always
be possible and supported. If the two disagree, the vault wins.

Maintenance is nearly free: the embedding sync already walks the vault and
compares mtimes, so extracting a page's wikilinks and its `summary:` in that
same pass costs a few lines - no extra requests, no second sync, no separate
invalidation to keep correct.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from .text import (
    HASH_VERSION,
    content_hash,
    first_sentence,
    normalize,
    page_hash,
    split_frontmatter,
    strip_links,
)

WORD_CAP = 400  # per chunk, heading-aware


def default_db(vault) -> Path:
    """Where the index lives for a vault, decided in ONE place.

    It was decided in three, and they disagreed: the maintenance pass wrote to
    `<vault>/.sentinel/index.db` while the chat and server read
    `<vault>/../.sentinel.db`. Everything worked, separately, on two different
    indexes - so a night of analysis was invisible to the next conversation,
    and the only symptom was a page count that would not go down.

    Inside the vault, so it travels with it and survives a container rebuild.
    Dot-prefixed, so the walk skips it.
    """
    return Path(vault) / ".sentinel" / "index.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path      TEXT PRIMARY KEY,
    page_key  TEXT NOT NULL,          -- derived from the filename
    mtime     REAL NOT NULL,
    summary   TEXT,
    summary_provisional INTEGER NOT NULL DEFAULT 1,
    type      TEXT,                   -- concept | note
    origin    TEXT,                   -- conversation | user  (later: ingestion)
    content_hash  TEXT NOT NULL,
    analyzed_hash TEXT
);
CREATE TABLE IF NOT EXISTS links (
    source     TEXT NOT NULL,
    target_key TEXT NOT NULL,
    display    TEXT NOT NULL,
    resolved   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    path     TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    heading  TEXT,
    content  TEXT NOT NULL,           -- the ORIGINAL text, for display
    is_summary INTEGER NOT NULL DEFAULT 0,
    stripped_hash TEXT NOT NULL       -- what the vector is actually keyed on
);
CREATE TABLE IF NOT EXISTS vectors (
    stripped_hash TEXT PRIMARY KEY,
    vec  BLOB NOT NULL
);
-- Literal half of the search substrate. Standalone rather than
-- external-content FTS5: chunks are delete-and-reinsert per page anyway, so
-- there is nothing for a shadow table to buy, and this way the indexed text
-- is the LINK-STRIPPED text - the same rule as the hash, the chunk vector and
-- the concept-presence check.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    path UNINDEXED, seq UNINDEXED, text
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tags (
    path TEXT NOT NULL,
    tag  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS aliases (
    alias_key     TEXT PRIMARY KEY,
    canonical_key TEXT NOT NULL,
    kind          TEXT NOT NULL,      -- acronym | user | judged
    added         TEXT
);
CREATE INDEX IF NOT EXISTS ix_src ON links(source);
CREATE INDEX IF NOT EXISTS ix_tgt ON links(target_key);
CREATE INDEX IF NOT EXISTS ix_res ON links(resolved);
-- One mention row per (page, concept). The first-occurrence rule already
-- guarantees that from the linker; making it a constraint lets the growth
-- queue use COUNT(*) instead of COUNT(DISTINCT source), which was the whole
-- cost of that query. A hand-written duplicate link collapses to the first
-- display form, which is what the mention record wants anyway.
CREATE UNIQUE INDEX IF NOT EXISTS ix_mention ON links(source, target_key);
-- Covering, so the growth queue never returns to the table.
CREATE INDEX IF NOT EXISTS ix_unres ON links(resolved, target_key, source);
CREATE INDEX IF NOT EXISTS ix_key ON files(page_key);
CREATE INDEX IF NOT EXISTS ix_chunk ON chunks(path);
CREATE INDEX IF NOT EXISTS ix_shash ON chunks(stripped_hash);
CREATE UNIQUE INDEX IF NOT EXISTS ix_tag ON tags(path, tag);
"""

# `page_key` is not in the recorded schema. It is a derived column: the
# resolution index is built from FILENAMES ONLY - reading every page's
# frontmatter for type precedence was O(n) requests and bought almost nothing.
# Storing the key means the resolved flag is one SQL statement instead of a
# Python pass over every link.

_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")  # NOT [^\]\n] - links wrap
_HEADING_LINE = re.compile(r"^(#{1,6})\s*(.*)$")


# ---------------------------------------------------------------------------
# Parsing a page
# ---------------------------------------------------------------------------


@dataclass
class Page:
    path: str
    page_key: str
    body: str
    summary: str | None
    summary_provisional: bool
    type: str | None
    origin: str | None
    content_hash: str
    tags: list[str]


def parse_page(rel_path: str, text: str) -> Page:
    fm, body = split_frontmatter(text)
    return Page(
        path=rel_path,
        page_key=normalize(Path(rel_path).stem),
        body=body,
        # A page written by hand in Obsidian has no `summary:`, and without
        # one it is invisible to the cheap path - every search that considers
        # it has to open it in full. Measured on the real vault: 40 of 40
        # pages had none, so rung 1 answered nothing at all.
        #
        # So the index DERIVES a provisional first sentence when frontmatter
        # has none. Derived, never written: the file is not touched, and the
        # analysis pass still replaces it. `write` continues to place one in
        # the file itself, because a page it creates should carry its own.
        summary=fm.get("summary") or first_sentence(body) or None,
        summary_provisional=(not fm.get("summary")) or
        str(fm.get("summary_provisional", "1")).lower() in ("1", "true", "yes"),
        type=fm.get("type"),
        origin=fm.get("origin"),
        content_hash=content_hash(body),
        tags=parse_tags(fm.get("tags", "")),
    )


def parse_tags(raw: str) -> list[str]:
    """`tags: a, b` and `tags: [a, b]` both work; an inline YAML list is the
    form Obsidian's own UI writes, so refusing it would reject the vault's
    own output."""
    return [t.strip().lstrip("#") for t in raw.strip("[]").split(",") if t.strip()]


def extract_links(body: str) -> list[tuple[str, str]]:
    """[(target_key, display)] for every wikilink in the body.

    `[[Canonical|shown]]` records the key of Canonical and the display form,
    which is what makes the `links` table the MENTION RECORD - which concept
    appeared in which page, resolved or not, in which spelling.
    """
    out = []
    for raw in _WIKILINK.findall(body):
        target, _, display = raw.partition("|")
        out.append((normalize(target), " ".join((display or target).split())))
    return out


def chunk_page(page: Page) -> list[tuple[int, str | None, str, int]]:
    """[(seq, heading, content, is_summary)], heading-aware, 400-word cap.

    The summary is indexed as its OWN chunk. That separate chunk is what lets
    semantic search answer "what is this page about" - a query-less question
    that chunk retrieval cannot otherwise serve, because it returns a body
    paragraph rather than a description.
    """
    rows: list[tuple[int, str | None, str, int]] = []
    seq = 0
    if page.summary:
        rows.append((seq, None, page.summary, 1))
        seq += 1

    heading: str | None = None
    buf: list[str] = []

    def flush():
        nonlocal seq, buf
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return
        words = text.split()
        if len(words) <= WORD_CAP:
            pieces = [text]
        else:
            pieces, cur, n = [], [], 0
            for para in text.split("\n\n"):
                pn = len(para.split())
                if cur and n + pn > WORD_CAP:
                    pieces.append("\n\n".join(cur))
                    cur, n = [], 0
                cur.append(para)
                n += pn
            if cur:
                pieces.append("\n\n".join(cur))
        for p in pieces:
            rows.append((seq, heading, p, 0))
            seq += 1

    for line in page.body.split("\n"):
        m = _HEADING_LINE.match(line)
        if m:
            flush()
            heading = m.group(2).strip()
        else:
            buf.append(line)
    flush()
    return rows


def embed_text(content: str, heading: str | None = None,
               title: str | None = None) -> str:
    """What actually goes to the encoder and to the full-text index.

    The chunk row keeps the ORIGINAL text for display; this is the searchable
    form: link-stripped, with the HEADING and, for the summary chunk, the PAGE
    NAME prepended.

    The heading was missing and it made a page unfindable by the words that
    describe it best. Measured: a page whose H1 is `afterglow` and whose
    section is `Decay physics` returned nothing for either word - only for
    `fades`, a word in the body. Headings sat in their own column and never
    reached the index.

    Link syntax stays out for the same reason as everywhere else: it is
    markup, and if the vector carried it then placing a link would move the
    vector for no semantic reason.
    """
    parts = [p for p in (title, heading, content) if p]
    return strip_links("\n".join(parts))


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


class Index:
    def __init__(self, vault: str | Path, db_path: str | Path,
                 exclude: list[str] | None = None):
        # Without the trailing slash the walk builds `wiki/Astronomyfoo.md`,
        # every read 404s, and the scan reports a CLEAN vault while having
        # read nothing.
        self.vault = Path(str(vault).rstrip("/") + "/")
        # Folders that live in the vault directory but are not part of the
        # second brain: another system's scratch space, an ingestion staging
        # area, a mirror of pages that already exist elsewhere. Dot-folders are
        # always skipped and do not need listing here.
        #
        # Deliberately a parameter and not a built-in list: which folders count
        # is a decision about the vault, not about this code.
        self.exclude = [e.strip("/") for e in (exclude or [])]
        # `check_same_thread=False` with a lock, rather than a connection per
        # thread. A tool call is a SEQUENCE - sync, then query - and it has to
        # be atomic as a whole, which a per-thread connection would not give.
        #
        # Needed because a server runs synchronous endpoints in a threadpool:
        # without this the first request raised ProgrammingError and the whole
        # tool server was unusable.
        # The index is derived data and its directory is ours to create. A
        # missing folder is not a reason to refuse to start - measured, and
        # the message it produced ("unable to open database file") says
        # nothing about what to do.
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.executescript(SCHEMA)
        self.hash_version_changed = self._check_hash_version()

    def _check_hash_version(self) -> int | None:
        """Returns the OLD version if it moved, else None.

        Derived data may be rebuilt at any time; what cannot be rebuilt is the
        knowledge that a page was already analysed. A hash definition change
        breaks the link between the two, so it has to be visible rather than
        inferred from a queue that suddenly grew.
        """
        row = self.db.execute(
            "SELECT value FROM meta WHERE key='hash_version'").fetchone()
        old = int(row["value"]) if row else None
        if old != HASH_VERSION:
            self.db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES "
                "('hash_version', ?)", (str(HASH_VERSION),))
            self.db.commit()
        return old if old is not None and old != HASH_VERSION else None

    # -- writing ----------------------------------------------------------

    def _replace_page(self, page: Page, mtime: float) -> None:
        self.db.execute("DELETE FROM tags       WHERE path=?", (page.path,))
        self.db.execute("DELETE FROM chunks     WHERE path=?", (page.path,))
        self.db.execute("DELETE FROM chunks_fts WHERE path=?", (page.path,))
        self.db.execute("DELETE FROM links  WHERE source=?", (page.path,))
        self.db.execute(
            """INSERT INTO files (path, page_key, mtime, summary,
                                  summary_provisional, type, origin, content_hash)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET
                 page_key=excluded.page_key, mtime=excluded.mtime,
                 summary=excluded.summary,
                 summary_provisional=excluded.summary_provisional,
                 type=excluded.type, origin=excluded.origin,
                 content_hash=excluded.content_hash""",
            (page.path, page.page_key, mtime, page.summary,
             int(page.summary_provisional), page.type, page.origin,
             page.content_hash),
        )
        self.db.executemany(
            "INSERT OR IGNORE INTO tags (path, tag) VALUES (?,?)",
            [(page.path, t) for t in page.tags],
        )
        self.db.executemany(
            "INSERT OR IGNORE INTO links (source, target_key, display) VALUES (?,?,?)",
            [(page.path, k, d) for k, d in extract_links(page.body)],
        )
        rows = chunk_page(page)
        title = Path(page.path).stem
        indexed = [
            (s, h, c, i, embed_text(c, h, title if i else None))
            for s, h, c, i in rows
        ]
        self.db.executemany(
            "INSERT INTO chunks (path, seq, heading, content, is_summary, "
            "stripped_hash) VALUES (?,?,?,?,?,?)",
            [(page.path, s, h, c, i, content_hash(t))
             for s, h, c, i, t in indexed],
        )
        self.db.executemany(
            "INSERT INTO chunks_fts (path, seq, text) VALUES (?,?,?)",
            [(page.path, s, t) for s, h, c, i, t in indexed],
        )

    def sync(self, full: bool = False) -> dict:
        """Walk on mtime, compare `content_hash` per changed file.

        The two triggers are NOT the same thing. `mtime` is the FILESYSTEM
        signal - which files changed on disk. `content_hash` is the SEMANTIC
        signal - whether what the page says changed. If only links were placed,
        mtime moved but the hash did not, and there is nothing to redo.
        """
        stats = {"seen": 0, "touched": 0, "relinked": 0, "reindexed": 0,
                 "deleted": 0, "unreadable": [], "vault_unreachable": False,
                 "refused_delete": 0}
        if not self.vault.is_dir():
            # A scan that cannot read must SAY SO. Returning quietly here is
            # how a missing vault becomes a CLEAN report - and worse, how the
            # deletion pass below would erase the whole index.
            stats["vault_unreachable"] = True
            return stats
        known = {r["path"]: (r["mtime"], r["content_hash"])
                 for r in self.db.execute("SELECT path, mtime, content_hash FROM files")}
        on_disk = set()

        for f in sorted(self.vault.rglob("*.md")):
            rel = str(f.relative_to(self.vault))
            # `.trash` is Obsidian's own deleted-pages folder, `.obsidian` its
            # config, `.git` history. Measured on the real vault: three of the
            # first pages the scan reported were deleted ones, whose links
            # were inflating the growth queue and whose text would have come
            # back from search.
            parts = Path(rel).parts
            if any(part.startswith(".") for part in parts):
                continue
            if self.exclude and any(
                    rel == e or rel.startswith(e + "/") for e in self.exclude):
                continue
            on_disk.add(rel)
            stats["seen"] += 1
            mtime = f.stat().st_mtime
            if not full and rel in known and known[rel][0] == mtime:
                continue
            stats["touched"] += 1
            try:
                text = f.read_text(encoding="utf-8")
            except Exception as e:
                # Loud failure over silent failure: a scan that could not read
                # once reported a CLEAN vault.
                stats["unreadable"].append((rel, str(e)))
                continue
            h = page_hash(text)
            unchanged = not full and rel in known and known[rel][1] == h
            # Re-parse either way. An unchanged hash does NOT mean "nothing to
            # redo": the linking pass writes links into the body and the
            # analysis pass writes `summary:` into frontmatter, and BOTH leave
            # the hash where it was. What the hash actually guards is the
            # expensive work - re-embedding and re-analysis - not the SQL.
            self._replace_page(parse_page(rel, text), mtime)
            stats["relinked" if unchanged else "reindexed"] += 1

        gone_set = set(known) - on_disk
        if gone_set and len(gone_set) == len(known) and len(known) > 2:
            # Every page vanished at once. That is an unmounted drive or a
            # wrong path, not an edit, and acting on it would destroy the
            # index over a filesystem hiccup. Refuse and be loud; a real
            # emptying is recovered with an explicit rebuild().
            stats["refused_delete"] = len(gone_set)
            gone_set = set()
        for gone in gone_set:
            # Its OWN outgoing rows go with it. Links pointing AT it are left
            # alone: they become unresolved and rejoin the growth queue, which
            # is correct - the concept is still referenced, it just has no page.
            self.db.execute("DELETE FROM links  WHERE source=?", (gone,))
            self.db.execute("DELETE FROM tags       WHERE path=?", (gone,))
            self.db.execute("DELETE FROM chunks     WHERE path=?", (gone,))
            self.db.execute("DELETE FROM chunks_fts WHERE path=?", (gone,))
            self.db.execute("DELETE FROM files  WHERE path=?", (gone,))
            stats["deleted"] += 1

        if not (stats["reindexed"] or stats["relinked"] or stats["deleted"]
                or full):
            # Nothing changed: no resolution to recompute, nothing to commit.
            # Measured at 300 pages, this is the difference between a 19 ms
            # and a 12 ms no-op - and a no-op runs on every tool call.
            return stats
        self.refresh_resolved()
        if full or stats["reindexed"] > 50:
            # Not on every sync - a conversational write reindexes one page.
            # The first build of a fresh DB is not `full`, but it does cross
            # this floor, which is where the planner statistics matter most.
            self.db.execute("ANALYZE")   # measured 1.87 -> 0.89 ms
        self.db.commit()
        return stats

    def carry_analysis_forward(self) -> int:
        """After a hash-definition change, re-mark pages that were analysed.

        Safe because it keys on `summary_provisional = 0`: that flag is set
        only by the analysis pass writing a real summary, and a hash change
        does not touch the page. It restores a fact that was true before the
        definition moved rather than asserting a new one.
        """
        n = self.db.execute(
            "UPDATE files SET analyzed_hash = content_hash "
            "WHERE summary_provisional = 0 "
            "AND (analyzed_hash IS NULL OR analyzed_hash != content_hash)"
        ).rowcount
        self.db.commit()
        return n

    def health_notes(self) -> list[str]:
        """What the health line says. Abnormal only - zero tokens normally."""
        notes = []
        if self.hash_version_changed is not None:
            notes.append(
                f"the content-hash definition changed (v"
                f"{self.hash_version_changed} -> v{HASH_VERSION}); analysed "
                f"pages have been re-queued - run "
                f"index.carry_analysis_forward() to restore them")
        st = self.sync()
        if st["vault_unreachable"]:
            notes.append(f"VAULT UNREACHABLE at {self.vault} - results below "
                         f"are from a stale index")
        if st["refused_delete"]:
            notes.append(f"every one of {st['refused_delete']} pages vanished "
                         f"from disk; refused to delete them from the index")
        if st["unreadable"]:
            notes.append(f"{len(st['unreadable'])} unreadable files, e.g. "
                         f"{st['unreadable'][0][0]}")
        return notes

    def rebuild(self) -> dict:
        """The escape hatch that keeps the index derived rather than a source
        of truth. Must always work."""
        for t in ("files", "links", "chunks", "chunks_fts", "tags"):
            self.db.execute(f"DELETE FROM {t}")   # aliases are NOT derived
        return self.sync(full=True)

    def refresh_resolved(self) -> None:
        self.db.execute(
            """UPDATE links SET resolved = (
                 target_key IN (SELECT page_key FROM files)
                 OR target_key IN (
                      SELECT alias_key FROM aliases
                      WHERE canonical_key IN (SELECT page_key FROM files)))"""
        )

    def page_created(self, page_key: str) -> int:
        """One statement brings every earlier link to that concept to life.

        This is the return on keeping unresolved links: measured at 228 links
        across 228 pages, zero files rewritten.
        """
        cur = self.db.execute(
            "UPDATE links SET resolved=1 WHERE target_key=? AND resolved=0",
            (page_key,),
        )
        self.db.commit()
        return cur.rowcount

    # -- reading ----------------------------------------------------------

    def resolve(self, name: str) -> list[str]:
        """Paths whose filename resolves from this name. Two hits -> [RETRY]."""
        key = normalize(name)
        row = self.db.execute(
            "SELECT canonical_key FROM aliases WHERE alias_key=?", (key,)
        ).fetchone()
        if row:
            key = row["canonical_key"]
        return [r["path"] for r in
                self.db.execute("SELECT path FROM files WHERE page_key=? ORDER BY path",
                                (key,))]

    def links_out(self, path: str) -> list[sqlite3.Row]:
        return list(self.db.execute(
            "SELECT target_key, display, resolved FROM links WHERE source=?", (path,)))

    def links_in(self, page_key: str, limit: int = 10):
        """Ranked by recency - what recently touched this concept is what is
        currently relevant. Returns (rows, total)."""
        total = self.db.execute(
            "SELECT COUNT(*) c FROM links WHERE target_key=?", (page_key,)
        ).fetchone()["c"]
        rows = list(self.db.execute(
            """SELECT l.source, l.display, f.summary, f.mtime
               FROM links l JOIN files f ON f.path = l.source
               WHERE l.target_key=? ORDER BY f.mtime DESC LIMIT ?""",
            (page_key, limit)))
        return rows, total

    def growth_queue(self, limit: int = 8):
        """Unresolved targets ranked by how many pages reference each."""
        total = self.db.execute(
            "SELECT COUNT(DISTINCT target_key) c FROM links WHERE resolved=0"
        ).fetchone()["c"]
        rows = list(self.db.execute(
            """SELECT target_key, MIN(display) display, COUNT(*) n
               FROM links WHERE resolved=0
               GROUP BY target_key ORDER BY n DESC, target_key LIMIT ?""", (limit,)))
        return rows, total

    def orphans(self) -> int:
        """A COUNT, not a list - in a young vault orphans are normal."""
        return self.db.execute(
            """SELECT COUNT(*) c FROM files f
               WHERE NOT EXISTS (SELECT 1 FROM links WHERE source = f.path)
                 AND NOT EXISTS (SELECT 1 FROM links WHERE target_key = f.page_key)"""
        ).fetchone()["c"]

    def summaries(self, paths: list[str]):
        """The interface. Five hits ~ 75 tokens, each saying what it is."""
        q = ",".join("?" * len(paths))
        return list(self.db.execute(
            f"SELECT path, summary, summary_provisional, origin FROM files "
            f"WHERE path IN ({q})",
            paths))

    # -- listing: enumerate by exact attribute ---------------------------
    # Deterministic enumeration, kept OUT of `search`. A combined
    # find(query, kind) was tried and FAILED: it mixed semantic retrieval with
    # exact enumeration, which are two different INTENTS, not two depths of
    # one. Depth is safe in a tool schema; intent is the anti-pattern.

    def by_tag(self, tag: str, limit: int = 20):
        total = self.db.execute(
            "SELECT COUNT(*) c FROM tags WHERE tag=?", (tag,)).fetchone()["c"]
        rows = list(self.db.execute(
            """SELECT f.path, f.summary, f.summary_provisional, f.origin FROM tags t
               JOIN files f ON f.path = t.path
               WHERE t.tag=? ORDER BY f.mtime DESC LIMIT ?""", (tag, limit)))
        return rows, total

    def by_path(self, prefix: str, limit: int = 20):
        # Trailing slash or the walk builds `wiki/Astronomyfoo.md`, every read
        # 404s, and the scan reports a CLEAN vault while reading nothing.
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        like = prefix + "%"
        total = self.db.execute(
            "SELECT COUNT(*) c FROM files WHERE path LIKE ?", (like,)).fetchone()["c"]
        rows = list(self.db.execute(
            "SELECT path, summary, summary_provisional, origin FROM files "
            "WHERE path LIKE ? ORDER BY path LIMIT ?", (like, limit)))
        return rows, total

    def recent(self, limit: int = 20):
        total = self.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
        rows = list(self.db.execute(
            "SELECT path, summary, summary_provisional, origin FROM files "
            "ORDER BY mtime DESC LIMIT ?", (limit,)))
        return rows, total

    def all_tags(self, limit: int = 50):
        total = self.db.execute(
            "SELECT COUNT(DISTINCT tag) c FROM tags").fetchone()["c"]
        rows = list(self.db.execute(
            "SELECT tag, COUNT(*) n FROM tags GROUP BY tag "
            "ORDER BY n DESC, tag LIMIT ?", (limit,)))
        return rows, total

    def meta(self, path: str):
        return self.db.execute(
            "SELECT * FROM files WHERE path=?", (path,)).fetchone()

    def pending_embed(self):
        """Chunks with no vector yet, deduplicated by stripped text.

        Because the vector is keyed on the LINK-STRIPPED hash and not on the
        chunk row, placing a link re-inserts the chunk and the existing vector
        is still there waiting for it. Nothing is re-encoded for markup - the
        same rule as the content hash, arriving here as free deduplication.
        """
        return list(self.db.execute(
            """SELECT DISTINCT c.stripped_hash, f.text AS content
               FROM chunks c JOIN chunks_fts f
                 ON f.path = c.path AND f.seq = c.seq
               WHERE NOT EXISTS (
                   SELECT 1 FROM vectors v
                   WHERE v.stripped_hash = c.stripped_hash)"""))

    def prune_vectors(self) -> int:
        return self.db.execute(
            "DELETE FROM vectors WHERE stripped_hash NOT IN "
            "(SELECT stripped_hash FROM chunks)").rowcount

    def pending_analysis(self) -> list[str]:
        """The queue derives itself - no separate table, the state is already
        in the index."""
        return [r["path"] for r in self.db.execute(
            "SELECT path FROM files "
            "WHERE analyzed_hash IS NULL OR analyzed_hash != content_hash "
            "ORDER BY path")]
