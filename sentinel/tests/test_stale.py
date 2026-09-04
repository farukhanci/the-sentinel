"""Run: python3 -m sentinel.tests.test_stale

The vault is edited BEHIND the system's back here - which is the normal case,
because the user has Obsidian open.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from ..index import Index
from ..tools import Sentinel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


def fresh():
    tmp = Path(tempfile.mkdtemp())
    v = tmp / "vault"
    (v / "wiki").mkdir(parents=True)
    for n in ("alpha", "beta", "gamma"):
        (v / "wiki" / f"{n}.md").write_text(
            f"---\ntype: note\nsummary: The {n} page.\nsummary_provisional: 0\n"
            f"---\n\n# {n}\n\nText about {n} and the afterglow.\n",
            encoding="utf-8")
    idx = Index(v, tmp / "s.db")
    idx.sync()
    return v, idx, Sentinel(idx)


# --- an external edit is picked up without anyone calling sync ------------
v, idx, S = fresh()
p = v / "wiki" / "alpha.md"
time.sleep(0.01)
p.write_text(p.read_text().replace("the afterglow", "the reverse shock"),
             encoding="utf-8")

ok("a tool call sees the edit, with no explicit sync",
   "reverse shock" in S.read("alpha", depth="full"), S.read("alpha", depth="full"))
ok("and search sees it too",
   "wiki/alpha.md" in S.search("reverse shock", limit=3))

# --- a new page created in Obsidian appears -------------------------------
(v / "wiki" / "delta.md").write_text(
    "---\ntype: note\nsummary: Written in Obsidian.\nsummary_provisional: 0\n"
    "---\n\n# delta\n\nBrand new.\n", encoding="utf-8")
ok("a page created outside the system is found",
   "wiki/delta.md" in S.listing(by="recent", limit=10))

# --- THE ONE THAT MATTERS: staleness must not disarm the expect guard -----
v, idx, S = fresh()
p = v / "wiki" / "alpha.md"
held = idx.meta("wiki/alpha.md")["content_hash"][:8]      # what the model holds
time.sleep(0.01)
p.write_text(p.read_text().replace("Text about", "COMPLETELY REWRITTEN text about"),
             encoding="utf-8")
# The index has not been told. Before the fix it still held the old hash, so
# the guard compared against a hash the file no longer had and let the write
# through, destroying the edit it was built to protect.
r = S.write("wiki/alpha.md", "# alpha\n\nOverwritten.", where="whole", expect=held)
ok("a write with a hash the file no longer has is REFUSED",
   r.startswith("[RETRY]"), r[:160])
ok("and the edit made outside the system survives",
   "COMPLETELY REWRITTEN" in p.read_text())
ok("the refusal hands back the current hash to retry with",
   idx.meta("wiki/alpha.md")["content_hash"][:8] in r, r[:160])

fresh_hash = idx.meta("wiki/alpha.md")["content_hash"][:8]
r = S.write("wiki/alpha.md", "# alpha\n\nNow overwritten deliberately.",
            where="whole", expect=fresh_hash)
ok("with the current hash the same write succeeds", r.startswith("[DONE]"), r)

# --- an unreachable vault is loud, and destroys nothing --------------------
v, idx, S = fresh()
before = idx.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
moved = v.parent / "vault-moved"
v.rename(moved)
out = S.listing(by="recent")
ok("an unreachable vault says so", "VAULT UNREACHABLE" in out, out[:200])
ok("and the index is NOT wiped",
   idx.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"] == before,
   str(idx.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]))
moved.rename(v)
ok("once it comes back, the health line goes quiet",
   "VAULT UNREACHABLE" not in S.listing(by="recent"))

# --- every page vanishing at once is refused, not obeyed -------------------
v, idx, S = fresh()
for f in (v / "wiki").glob("*.md"):
    f.unlink()
out = S.listing(by="recent")
ok("a vault that emptied all at once is reported, not acted on",
   "refused to delete" in out, out[:200])
ok("the rows are still there",
   idx.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"] == 3)
ok("an explicit rebuild is what actually clears them",
   idx.rebuild() is not None
   and idx.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"] == 0)

# --- one page deleted normally is still handled ---------------------------
v, idx, S = fresh()
(v / "wiki" / "beta.md").unlink()
S.listing(by="recent")
ok("deleting ONE page still works normally", idx.meta("wiki/beta.md") is None)
ok("and the other two are untouched",
   idx.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"] == 2)

# --- unreadable files are named, not skipped in silence -------------------
v, idx, S = fresh()
(v / "wiki" / "bad.md").write_bytes(b"---\ntype: note\n---\n\n\xff\xfe binary \xff")
out = S.listing(by="recent")
ok("an unreadable file is named in the health line",
   "unreadable" in out and "bad.md" in out, out[:250])

# --- cost -----------------------------------------------------------------
v, idx, S = fresh()
t0 = time.perf_counter()
for _ in range(20):
    S.read("alpha")
per = (time.perf_counter() - t0) * 1000 / 20
ok("a primitive with a fresh index stays cheap", per < 50, f"{per:.1f} ms per call")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
print(f"  {per:.2f} ms per call including the freshness walk")
sys.exit(1 if FAIL else 0)
