"""Run: python3 -m sentinel.tests.test_write"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ..index import Index
from ..text import normalize
from ..tools import Sentinel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


tmp = Path(tempfile.mkdtemp())
vault = tmp / "vault"
(vault / "wiki").mkdir(parents=True)
(vault / "wiki" / "afterglow.md").write_text(
    "---\ntype: concept\nsummary: Late-time emission.\nsummary_provisional: 0\n---\n\n"
    "# afterglow\n\n## Origin\n\nIt follows the burst.\n\n## Decay\n\nIt fades.\n",
    encoding="utf-8")
(vault / "wiki" / "cites.md").write_text(
    "---\ntype: note\nsummary: Cites things.\nsummary_provisional: 0\n---\n\n"
    "# cites\n\nWe discuss the [[afterglow]] and also [[afterglow|its decay]].\n",
    encoding="utf-8")

idx = Index(vault, tmp / "w.db")
idx.sync()
S = Sentinel(idx, encoder=None, origin="conversation")

# --- the expect guard -----------------------------------------------------
ok("expect is required", S.write("wiki/new.md", "x").startswith("[RETRY]"))
ok("creating something that exists is refused",
   S.write("wiki/afterglow.md", "x", expect="new").startswith("[RETRY]"))
ok("writing to something that does not exist is refused",
   S.write("wiki/ghost.md", "x", where="whole", expect="abc123").startswith("[RETRY]"))

h = idx.meta("wiki/afterglow.md")["content_hash"][:8]
stale = S.write("wiki/afterglow.md", "x", where="whole", expect="deadbeef")
ok("a stale expect is refused and names both hashes",
   stale.startswith("[RETRY]") and h in stale, stale[:120])
ok("the refusal hands back enough to merge from", "It follows the burst" in stale)

# --- creating -------------------------------------------------------------
r = S.write("wiki/jet break.md",
            "# jet break\n\nThe light curve steepens when the cone widens.",
            expect="new")
ok("a new page is created", r.startswith("[DONE] created"), r)
ok("where is ignored on a page that does not exist yet",
   (vault / "wiki" / "jet break.md").exists())

row = idx.meta("wiki/jet break.md")
ok("a page can never be written without a summary", bool(row["summary"]), str(dict(row)))
ok("that summary is marked provisional", row["summary_provisional"] == 1)
ok("the provisional summary is the first sentence, verbatim",
   row["summary"].startswith("The light curve steepens"), row["summary"])
ok("code sets created, updated and origin",
   "created:" in (vault / "wiki" / "jet break.md").read_text()
   and "origin: conversation" in (vault / "wiki" / "jet break.md").read_text())
ok("the write is announced with the new expect value", "expect " in r)
ok("and says the page is queued for analysis", "provisional summary" in r)

# --- the page's own frontmatter, set as it is created --------------------
r = S.write("wiki/afterglow shock.md",
            "---\ntype: concept\ntags: astro, shocks\n---\n\n"
            "# afterglow shock\n\nThe boundary that lights up.", expect="new")
ok("a page can be created as a concept, not only as a note",
   idx.meta("wiki/afterglow shock.md")["type"] == "concept", r)
ok("and its tags come with it",
   {t["tag"] for t in idx.db.execute(
       "SELECT tag FROM tags WHERE path='wiki/afterglow shock.md'")}
   == {"astro", "shocks"})
ok("the body is the body, frontmatter is not duplicated into it",
   "type: concept" not in idx.db.execute(
       "SELECT content FROM chunks WHERE path='wiki/afterglow shock.md' "
       "AND is_summary=0").fetchone()["content"])
ok("code still owns created, updated and origin",
   all(k in (vault / "wiki" / "afterglow shock.md").read_text()
       for k in ("created:", "updated:", "origin:")))

for owned in ("summary", "origin", "created"):
    r = S.write(f"wiki/try-{owned}.md",
                f"---\ntype: concept\n{owned}: mine\n---\n\n# t\n\nText.",
                expect="new")
    ok(f"setting {owned} in the content is refused", r.startswith("[STOP]"), r)
    ok(f"and the page is not created", not (vault / f"wiki/try-{owned}.md").exists())

r = S.write("wiki/plain.md", "# plain\n\nNo frontmatter here.", expect="new")
ok("content with no frontmatter still works",
   idx.meta("wiki/plain.md")["type"] == "note", r)
ok("and its body survives intact",
   "No frontmatter here" in (vault / "wiki" / "plain.md").read_text())

# --- a page written in conversation is marked as such --------------------
ok("read meta says where the page came from",
   "from conversation" in S.read("wiki/plain.md"), S.read("wiki/plain.md"))
ok("and so does a listing line",
   "(from conversation)" in S.listing(by="path", value="wiki"),
   S.listing(by="path", value="wiki"))

(vault / "wiki" / "sourced.md").write_text(
    "---\ntype: source\nsummary: From a paper.\nsummary_provisional: 0\n"
    "origin: source\n---\n\n# sourced\n\nText.\n", encoding="utf-8")
idx.sync()
line = [x for x in S.listing(by="path", value="wiki").splitlines()
        if "sourced" in x]
ok("a sourced page carries no such mark",
   line and "from conversation" not in line[0], str(line))

# --- a folderless path lands in the pages folder, not the vault root -----
r = S.write("loose.md", "# loose\n\nA finding with no folder given.",
            expect="new")
ok("a page with no folder goes to wiki/", "wiki/loose.md" in r, r)
ok("and not to the vault root", not (vault / "loose.md").exists())

# --- one place decides where a write lands -------------------------------
ok("a folderless path resolves to the pages folder",
   S.write_path("loose2.md") == "wiki/loose2.md", S.write_path("loose2.md"))
ok("a path with a folder is left alone",
   S.write_path("wiki/x.md") == "wiki/x.md")
ok("and the same rule decides what write does",
   "wiki/loose2.md" in S.write("loose2.md", "# l\n\nText here.",
                               expect="new"))

# --- a conversation record cannot be written to ---------------------------
(vault / "conversations").mkdir(exist_ok=True)
(vault / "conversations" / "rec-1.md").write_text(
    "---\ntype: transcript\norigin: conversation\nsummary: Talk.\n"
    "summary_provisional: 0\n---\n\n# rec-1\n\n## You\n\nSaid a thing.\n",
    encoding="utf-8")
idx.sync()
h = idx.meta("conversations/rec-1.md")["content_hash"][:8]
for label, kw in (("rewritten", dict(where="whole", expect=h)),
                  ("appended to", dict(where="end", expect=h))):
    out = S.write("conversations/rec-1.md", "x", **kw)
    ok(f"a record cannot be {label}", out.startswith("[STOP]"), out)
ok("nor can a new page be created beside one",
   S.write("conversations/rec-2.md", "x", expect="new").startswith("[STOP]"))
ok("the refusal is [STOP], because no rephrasing would work",
   "[RETRY]" not in S.write("conversations/rec-1.md", "x", where="end",
                            expect=h))
ok("and it says where the finding should go instead",
   "wiki/ instead" in S.write("conversations/rec-1.md", "x", where="end",
                              expect=h))
ok("the record itself is untouched",
   "Said a thing." in (vault / "conversations" / "rec-1.md").read_text())

# --- the write description says where pages go ---------------------------
from ..schema import schemas as _schemas  # noqa: E402

_w = [x for x in _schemas(Sentinel) if x["name"] == "write"][0]["description"]
ok("the tool says pages go in wiki/", "wiki/<name>.md" in _w, _w[:200])
ok("and says not the vault root", "vault root" in _w)
_flat = " ".join(_w.split())
ok("the line is drawn at the source, not the speaker",
   "NOT WHAT YOU KNOW ABOUT THE SUBJECT" in _flat
   and "from your training does not" in _flat, _flat[:400])
ok("a conclusion drawn from the vault is allowed to be recorded",
   "vault told you" in _flat, _flat[:400])
ok("capture waits for the user to ask",
   "WHEN THE USER ASKS YOU TO RECORD" in _w, _w[:300])
ok("and says to do it in one call",
   "ONE call" in _w and "Do not search first" in _w)

# --- a conversation page grows instead of refusing the second write ------
r1 = S.write("wiki/finding.md", "# finding\n\nThe first thing we settled.",
             expect="new")
ok("the first write creates it", r1.startswith("[DONE] created"), r1)
r2 = S.write("wiki/finding.md", "The second thing, weeks later.", expect="new")
ok("the second write adds rather than being refused",
   r2.startswith("[DONE] added to"), r2)
grown = (vault / "wiki" / "finding.md").read_text()
ok("the first writing survives", "The first thing we settled." in grown, grown)
ok("and the second is a section under it",
   grown.index("first thing") < grown.index("second thing"), grown)
ok("with a dated heading written by code, not by the model",
   "\n## 20" in grown, grown)

r3 = S.write("wiki/finding.md", "## My own heading\n\nMore.", expect="new")
ok("a heading the model supplied is kept",
   "## My own heading" in (vault / "wiki" / "finding.md").read_text())

(vault / "wiki" / "sourced2.md").write_text(
    "---\ntype: note\norigin: source\nsummary: From a paper.\n"
    "summary_provisional: 0\n---\n\n# sourced2\n\nText.\n", encoding="utf-8")
idx.sync()
ok("a page that did NOT come from conversation is still refused",
   S.write("wiki/sourced2.md", "x", expect="new").startswith("[RETRY]"))

# --- idempotency ----------------------------------------------------------
again = S.write("wiki/jet break.md",
                "The light curve steepens when the cone widens.", expect="new")
ok("repeating the same content adds nothing",
   again.startswith("[DONE]") and "already says this" in again, again)
ok("and the body is not duplicated",
   (vault / "wiki" / "jet break.md").read_text().split("---", 2)[2]
   .count("steepens") == 1)

# --- section, end, frontmatter -------------------------------------------
h = idx.meta("wiki/afterglow.md")["content_hash"][:8]
r = S.write("wiki/afterglow.md", "## Decay\n\nIt fades as a power law.",
            where="section", target="Decay", expect=h)
body = (vault / "wiki" / "afterglow.md").read_text()
ok("section replaces only that section",
   "power law" in body and "It follows the burst" in body and "It fades.\n" not in body,
   body)

h = idx.meta("wiki/afterglow.md")["content_hash"][:8]
ok("a missing heading is [RETRY] and lists the real ones",
   S.write("wiki/afterglow.md", "x", where="section", target="Nope",
           expect=h).startswith("[RETRY]"))

r = S.write("wiki/afterglow.md", "## Notes\n\nAdded at the end.",
            where="end", expect=h)
ok("end appends", "Added at the end" in
   (vault / "wiki" / "afterglow.md").read_text())

h = idx.meta("wiki/afterglow.md")["content_hash"][:8]
r = S.write("wiki/afterglow.md", "astro, grb", where="frontmatter",
            target="tags", expect=h)
ok("frontmatter sets a field",
   "astro" in [t["tag"] for t in idx.db.execute(
       "SELECT tag FROM tags WHERE path='wiki/afterglow.md'")], r)

h = idx.meta("wiki/afterglow.md")["content_hash"][:8]
r = S.write("wiki/afterglow.md", "a hand-written summary", where="frontmatter",
            target="summary", expect=h)
ok("summary has exactly one writer, and this is not it", r.startswith("[STOP]"), r)
ok("that refusal is [STOP], not [RETRY] - retrying could never work",
   "[RETRY]" not in r)

# --- the index is refreshed with the write, not after it ------------------
h = idx.meta("wiki/afterglow.md")["content_hash"][:8]
S.write("wiki/afterglow.md", "# afterglow\n\nIt mentions the [[jet break]].",
        where="whole", expect=h)
ok("links are reindexed as part of the write",
   any(r["target_key"] == normalize("jet break")
       for r in idx.links_out("wiki/afterglow.md")))
ok("and the new page resolves them",
   idx.db.execute("SELECT resolved FROM links WHERE source='wiki/afterglow.md' "
                  "AND target_key=?", (normalize("jet break"),)).fetchone()["resolved"] == 1)

# --- relocate -------------------------------------------------------------
before = (vault / "wiki" / "cites.md").read_text()
hash_before = idx.meta("wiki/cites.md")["content_hash"]
r = S.relocate("wiki/afterglow.md", "wiki/afterglow emission.md")
after = (vault / "wiki" / "cites.md").read_text()
ok("relocate moves the file", (vault / "wiki" / "afterglow emission.md").exists()
   and not (vault / "wiki" / "afterglow.md").exists(), r)
ok("incoming links are rewritten, not aliased",
   "[[afterglow emission|afterglow]]" in after, after)
ok("the display text is preserved, so the words on the page do not change",
   "[[afterglow emission|its decay]]" in after, after)
ok("which means the citing page's hash does not move",
   idx.meta("wiki/cites.md")["content_hash"] == hash_before,
   f"{hash_before[:8]} -> {idx.meta('wiki/cites.md')['content_hash'][:8]}")
ok("so the rewritten page is not queued for re-analysis by the rename",
   before != after and
   idx.meta("wiki/cites.md")["content_hash"] == hash_before)
ok("relocate reports how many pages it touched", "1 pages had links" in r, r)
ok("the old path is gone from the index", idx.meta("wiki/afterglow.md") is None)
ok("the links now resolve to the new name",
   idx.db.execute("SELECT resolved FROM links WHERE target_key=?",
                  (normalize("afterglow emission"),)).fetchone()["resolved"] == 1)
ok("relocating onto an existing path is refused",
   S.relocate("wiki/cites.md", "wiki/jet break.md").startswith("[RETRY]"))
ok("relocating something that is not there is [STOP]",
   S.relocate("wiki/ghost.md", "wiki/x.md").startswith("[STOP]"))

# --- remove ---------------------------------------------------------------
r = S.remove("wiki/afterglow emission.md")
ok("remove deletes the file", not (vault / "wiki" / "afterglow emission.md").exists())
ok("incoming links survive and rejoin the growth queue",
   "unresolved and back in the growth queue" in r, r)
ok("the page's own outgoing links are gone",
   not idx.links_out("wiki/afterglow emission.md"))
ok("removing what is not there is [STOP]",
   S.remove("wiki/ghost.md").startswith("[STOP]"))

gq, _ = idx.growth_queue(10)
ok("the removed concept is back in the queue",
   any(r["target_key"] == normalize("afterglow emission") for r in gq),
   str([r["target_key"] for r in gq]))

# --- a full rebuild still agrees with the vault ---------------------------
snapshot = (idx.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"],
            idx.db.execute("SELECT COUNT(*) c FROM links").fetchone()["c"])
idx.rebuild()
after_rb = (idx.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"],
            idx.db.execute("SELECT COUNT(*) c FROM links").fetchone()["c"])
ok("after every write path, the index still matches a fresh scan",
   snapshot == after_rb, f"{snapshot} vs {after_rb}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
sys.exit(1 if FAIL else 0)
