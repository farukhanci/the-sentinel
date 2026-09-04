"""Run: python3 -m sentinel.tests.test_index"""

from __future__ import annotations

import random
import sys
import tempfile
import time
from pathlib import Path

from ..index import Index
from ..text import normalize

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


CONCEPTS = [
    "gamma-ray burst", "afterglow", "forward shock", "reverse shock",
    "host galaxy", "peak energy", "Lorentz factor", "synchrotron radiation",
    "Amati relation", "fireball model", "interstellar medium", "jet break",
    "prompt emission", "redshift", "circumburst medium",
]

tmp = Path(tempfile.mkdtemp())
vault = tmp / "vault"
(vault / "wiki").mkdir(parents=True)
random.seed(7)

# 300 pages, ~10 links each - the measured link density.
for i in range(300):
    picks = random.sample(CONCEPTS, 10)
    body = "\n\n".join(
        f"## Section {j}\n\nParagraph about [[{c}]] and its role in the model. "
        + "filler " * 30
        for j, c in enumerate(picks)
    )
    (vault / "wiki" / f"page-{i:03d}.md").write_text(
        f"---\ntype: note\norigin: user\nsummary: Page {i} on {picks[0]}.\n"
        f"summary_provisional: 0\n---\n\n# Page {i}\n\n{body}\n",
        encoding="utf-8",
    )

# Two of the concepts exist as real pages, the rest do not.
for c in ["afterglow", "gamma-ray burst"]:
    (vault / "wiki" / f"{c}.md").write_text(
        f"---\ntype: concept\nsummary: The {c}.\nsummary_provisional: 0\n---\n\n"
        f"# {c}\n\nA concept page.\n", encoding="utf-8")

idx = Index(vault, tmp / "sentinel.db")
t0 = time.perf_counter()
stats = idx.sync()
build_ms = (time.perf_counter() - t0) * 1000

ok("full sync indexed every page", stats["reindexed"] == 302, str(stats))
ok("no unreadable file", not stats["unreadable"], str(stats["unreadable"]))

n_links = idx.db.execute("SELECT COUNT(*) c FROM links").fetchone()["c"]
ok("link density ~10/page", 2900 <= n_links <= 3100, f"{n_links} links")

# --- Obsidian's own folders are not part of the vault ---------------------
(vault / ".trash").mkdir(exist_ok=True)
(vault / ".trash" / "deleted.md").write_text(
    "---\ntype: note\nsummary: Was deleted.\n---\n\n# gone\n\n"
    "It still links [[afterglow]] and would inflate the queue.\n",
    encoding="utf-8")
(vault / ".obsidian").mkdir(exist_ok=True)
(vault / ".obsidian" / "notes.md").write_text("config-ish\n", encoding="utf-8")
s = idx.sync()
ok("a deleted page in .trash is not indexed",
   idx.meta(".trash/deleted.md") is None, str(s))
ok("nor anything under .obsidian", idx.meta(".obsidian/notes.md") is None)

# --- every entry point agrees where the index lives ----------------------
import inspect as _i  # noqa: E402

from ..index import default_db  # noqa: E402
from .. import chat as _chat, maintain as _maint, server as _srv  # noqa: E402

ok("the path is decided in one place",
   str(default_db("/x/vault")) == "/x/vault/.sentinel/index.db",
   str(default_db("/x/vault")))
for _m in (_chat, _maint, _srv):
    src = _i.getsource(_m)
    ok(f"{_m.__name__.split('.')[-1]} uses it",
       "default_db(vault)" in src and ".sentinel.db" not in src)
ok("and it is inside the vault, so the walk skips it",
   default_db("/x/vault").name.startswith("index")
   and default_db("/x/vault").parent.name.startswith("."))

# --- the index makes its own home ----------------------------------------
deep = tmp / "deepvault"
(deep / "wiki").mkdir(parents=True)
di = Index(deep, deep / ".sentinel" / "nested" / "index.db")
ok("a missing index directory is created, not an error",
   (deep / ".sentinel" / "nested").is_dir() and di.sync()["seen"] == 0)

# --- a change in what the hash MEANS is visible, not silent --------------
from ..text import HASH_VERSION  # noqa: E402

hv = tmp / "hv"
(hv / "wiki").mkdir(parents=True)
(hv / "wiki" / "done.md").write_text(
    "---\ntype: note\nsummary: A real summary.\nsummary_provisional: 0\n"
    "---\n\n# done\n\nText.\n", encoding="utf-8")
(hv / "wiki" / "todo.md").write_text(
    "---\ntype: note\nsummary: First line.\nsummary_provisional: 1\n"
    "---\n\n# todo\n\nText.\n", encoding="utf-8")
hidx = Index(hv, tmp / "hv.db")
hidx.sync()
ok("a fresh store reports no version change", hidx.hash_version_changed is None)
hidx.db.execute("UPDATE files SET analyzed_hash = content_hash "
                "WHERE path='wiki/done.md'")
hidx.db.commit()
ok("the analysed page is out of the queue",
   "wiki/done.md" not in hidx.pending_analysis())

# What a definition change looks like: the stored hashes no longer match.
hidx.db.execute("UPDATE meta SET value='0' WHERE key='hash_version'")
hidx.db.execute("UPDATE files SET analyzed_hash='stale'")
hidx.db.commit()
hidx2 = Index(hv, tmp / "hv.db")
hidx2.sync()
ok("reopening after a definition change reports it",
   hidx2.hash_version_changed == 0, str(hidx2.hash_version_changed))
ok("and says so in the health line, naming the fix",
   any("content-hash definition changed" in n and "carry_analysis_forward" in n
       for n in hidx2.health_notes()), str(hidx2.health_notes()))
ok("the analysed page was re-queued by the change",
   "wiki/done.md" in hidx2.pending_analysis())
n = hidx2.carry_analysis_forward()
ok("carrying forward restores exactly the analysed pages", n == 1, str(n))
ok("the analysed page is out of the queue again",
   "wiki/done.md" not in hidx2.pending_analysis())
ok("and the provisional one stays in it",
   "wiki/todo.md" in hidx2.pending_analysis())
ok("a second open reports nothing", Index(hv, tmp / "hv.db").hash_version_changed is None)

# --- folders the vault holds but the second brain should not -------------
(vault / "staging").mkdir(exist_ok=True)
(vault / "staging" / "part-1.md").write_text(
    "---\ntype: note\nsummary: A fragment.\n---\n\n# part\n\n"
    "A section of a page that already exists elsewhere.\n", encoding="utf-8")
idx2 = Index(vault, tmp / "excl.db", exclude=["staging"])
st2 = idx2.sync()
ok("an excluded folder is not indexed", idx2.meta("staging/part-1.md") is None)
ok("but the rest of the vault still is", st2["seen"] > 0)
plain = Index(vault, tmp / "noexcl.db")
plain.sync()
ok("without the exclusion it IS indexed",
   plain.meta("staging/part-1.md") is not None)
ok("a trailing slash in the exclusion is tolerated",
   Index(vault, tmp / "sl.db", exclude=["staging/"]).sync()
   and Index(vault, tmp / "sl.db").meta("staging/part-1.md") is None)
import shutil as _sh2
_sh2.rmtree(vault / "staging")
idx.sync()

# --- a page with no frontmatter summary still gets one, derived not written
(vault / "wiki" / "bare.md").write_text(
    "# bare\n\nThe opening sentence of a page nobody summarised. More text.\n",
    encoding="utf-8")
idx.sync()
row = idx.meta("wiki/bare.md")
ok("the index derives a provisional summary so the cheap path works",
   row["summary"].startswith("The opening sentence"), str(dict(row)))
ok("and marks it provisional", row["summary_provisional"] == 1)
ok("without touching the file",
   (vault / "wiki" / "bare.md").read_text().startswith("# bare"))
(vault / "wiki" / "bare.md").unlink()
import shutil as _sh
_sh.rmtree(vault / ".trash"); _sh.rmtree(vault / ".obsidian")
idx.sync()

# --- the two invalidation triggers are not the same thing -----------------
p = vault / "wiki" / "page-000.md"
orig = p.read_text()

# What the linking pass actually does: the rendered text is identical.
linked = orig.replace("[[afterglow]]", "[[Afterglow|afterglow]]")
p.write_text(linked)
s = idx.sync()
ok("placing a link does not move the hash", s["reindexed"] == 0, str(s))
ok("but the links table IS refreshed", s["relinked"] == 1, str(s))
ok("the new display form is recorded",
   any(r["display"] == "afterglow" for r in idx.links_out("wiki/page-000.md")))

# What the analysis pass does: frontmatter only.
resummarised = linked.replace("summary: Page 0", "summary: Rewritten for page 0")
p.write_text(resummarised)
s = idx.sync()
ok("writing a summary does not queue re-analysis", s["reindexed"] == 0, str(s))
ok("but the stored summary IS refreshed",
   idx.summaries(["wiki/page-000.md"])[0]["summary"].startswith("Rewritten"),
   idx.summaries(["wiki/page-000.md"])[0]["summary"])

# A real content edit.
p.write_text(resummarised.replace("Paragraph about", "A different paragraph about"))
s = idx.sync()
ok("a real content edit does reindex", s["reindexed"] == 1, str(s))

p.write_text(orig)
idx.sync()

# --- what gets indexed is not what gets displayed ------------------------
row = idx.db.execute(
    "SELECT c.content, f.text FROM chunks c JOIN chunks_fts f "
    "ON f.path=c.path AND f.seq=c.seq WHERE c.path='wiki/page-001.md' "
    "AND c.is_summary=0 LIMIT 1").fetchone()
ok("the indexed text carries the heading the display text does not",
   row["text"].startswith("S") and not row["content"].startswith("S"),
   f"{row['text'][:30]!r} vs {row['content'][:30]!r}")
srow = idx.db.execute(
    "SELECT c.content, f.text FROM chunks c JOIN chunks_fts f "
    "ON f.path=c.path AND f.seq=c.seq WHERE c.path='wiki/page-001.md' "
    "AND c.is_summary=1").fetchone()
ok("the summary chunk carries the page name, so a page is findable by it",
   srow["text"].startswith("page-001"), repr(srow["text"][:40]))
ok("but the stored summary itself is untouched",
   not srow["content"].startswith("page-001"), repr(srow["content"][:40]))

# --- vectors are keyed on the stripped text, not the chunk row -----------
idx.db.executemany("INSERT OR IGNORE INTO vectors (stripped_hash, vec) VALUES (?,?)",
                   [(r["stripped_hash"], b"x") for r in idx.pending_embed()])
idx.db.commit()
ok("every chunk has a vector once encoded", not idx.pending_embed())

p.write_text(orig.replace("[[forward shock]]", "[[Forward shock|forward shock]]"))
idx.sync()
ok("placing a link re-encodes nothing", not idx.pending_embed(),
   f"{len(idx.pending_embed())} chunks would be re-encoded")
p.write_text(orig)
idx.sync()

# --- resolution -----------------------------------------------------------
ok("resolve by bare name", idx.resolve("Afterglow") == ["wiki/afterglow.md"],
   str(idx.resolve("Afterglow")))
ok("plural folds to the same page", idx.resolve("afterglows") == ["wiki/afterglow.md"])

res = idx.db.execute(
    "SELECT COUNT(*) c FROM links WHERE resolved=1").fetchone()["c"]
ok("only existing pages count as resolved", 0 < res < n_links, f"{res}/{n_links}")

# --- page creation is one statement ---------------------------------------
(vault / "wiki" / "host galaxy.md").write_text(
    "---\ntype: concept\nsummary: The host galaxy.\nsummary_provisional: 0\n---\n\n"
    "# host galaxy\n\nA concept page.\n", encoding="utf-8")
idx.sync()
t0 = time.perf_counter()
n = idx.page_created(normalize("host galaxy"))
create_ms = (time.perf_counter() - t0) * 1000
ok("creating a page brought its links to life with zero files rewritten", n == 0,
   f"{n} still unresolved after sync - sync should already have flipped them")

# --- deletion -------------------------------------------------------------
before_in, _ = idx.links_in(normalize("afterglow"), limit=5)
(vault / "wiki" / "afterglow.md").unlink()
idx.sync()
still = idx.db.execute(
    "SELECT COUNT(*) c FROM links WHERE target_key=?", (normalize("afterglow"),)
).fetchone()["c"]
unres = idx.db.execute(
    "SELECT COUNT(*) c FROM links WHERE target_key=? AND resolved=0",
    (normalize("afterglow"),)).fetchone()["c"]
ok("incoming links survive deletion and go unresolved",
   still > 0 and still == unres, f"{unres}/{still}")
ok("the deleted page's own outgoing links are gone",
   not idx.links_out("wiki/afterglow.md"))

# --- rebuild --------------------------------------------------------------
snapshot = idx.db.execute(
    "SELECT COUNT(*) a, (SELECT COUNT(*) FROM files) b FROM links").fetchone()
idx.rebuild()
after_rb = idx.db.execute(
    "SELECT COUNT(*) a, (SELECT COUNT(*) FROM files) b FROM links").fetchone()
ok("a full rebuild reproduces the same index",
   tuple(snapshot) == tuple(after_rb), f"{tuple(snapshot)} vs {tuple(after_rb)}")

# --- pending analysis is a query, not a table -----------------------------
ok("every page is pending until analysed", len(idx.pending_analysis()) == 302,
   str(len(idx.pending_analysis())))

# --- timings --------------------------------------------------------------
def ms(fn, n=50):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) * 1000 / n


timings = {
    "links out of a page": ms(lambda: idx.links_out("wiki/page-042.md")),
    "links into a concept": ms(lambda: idx.links_in(normalize("peak energy"), 10)),
    "growth queue ranked": ms(lambda: idx.growth_queue(8)),
    "orphan pages": ms(idx.orphans),
    "summaries for 5 hits": ms(lambda: idx.summaries(
        [f"wiki/page-{i:03d}.md" for i in range(5)])),
}
for name, t in timings.items():
    ok(f"{name} under 2 ms", t < 2.0, f"{t:.3f} ms")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
print(f"  302 pages / {n_links} links, full build {build_ms:.0f} ms")
for name, t in timings.items():
    print(f"    {name:<24} {t:.3f} ms")
sys.exit(1 if FAIL else 0)
