"""Run: python3 -m sentinel.tests.test_tools"""

from __future__ import annotations

import re
import sys
import tempfile
import zlib
from pathlib import Path

from ..embed import embed_pending
from ..index import Index
from ..output import STATUSES, cut_point, split_parts
from ..text import normalize
from ..tools import Sentinel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


class FakeEncoder:
    def encode(self, texts, kind="passage"):
        import numpy as np
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 384] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-9, None)


tmp = Path(tempfile.mkdtemp())
vault = tmp / "vault"
(vault / "wiki").mkdir(parents=True)

LONG = "\n\n".join(
    f"## Section {i}\n\n" + ("Sentence about the afterglow and its decay. " * 60)
    for i in range(4))
(vault / "wiki" / "gamma-ray burst.md").write_text(
    "---\ntype: concept\ntags: [astro, grb]\nsummary: What a gamma-ray burst is.\n"
    "summary_provisional: 0\n---\n\n# gamma-ray burst\n\n"
    "Intro paragraph mentioning the [[afterglow]] and the [[fireball model]].\n\n"
    + LONG, encoding="utf-8")
(vault / "wiki" / "arxiv-2504.11743.md").write_text(
    "---\ntype: note\ntags: [paper]\nsummary: Constraining initial Lorentz factors.\n"
    "summary_provisional: 0\n---\n\n# Paper\n\nWe cite [[gamma-ray burst]] work.\n",
    encoding="utf-8")
(vault / "wiki" / "afterglow.md").write_text(
    "---\ntype: concept\nsummary: Late-time emission.\nsummary_provisional: 0\n---\n\n"
    "# afterglow\n\nIt follows the [[gamma-ray burst]].\n", encoding="utf-8")
(vault / "wiki" / "lonely.md").write_text(
    "---\ntype: note\nsummary: Nothing links here.\nsummary_provisional: 1\n---\n\n"
    "# lonely\n\nNo links at all.\n", encoding="utf-8")

idx = Index(vault, tmp / "t.db")
idx.sync()
enc = FakeEncoder()
embed_pending(idx, enc)
S = Sentinel(idx, enc)

# --- the status set is closed --------------------------------------------
outs = [S.read("gamma-ray burst"), S.read("gamma-ray burst", depth="outline"), S.read("nope"),
        S.read("gamma-ray burst", depth="full"), S.read("gamma-ray burst", depth="part"),
        S.search("afterglow"), S.listing(by="all_tags"), S.listing(by="tag", value="astro"),
        S.graph(), S.graph("gamma-ray burst"), S.graph("fireball model"),
        S.graph("nothing at all"), S.read("gamma-ray burst", depth="sideways")]
markers = {m for o in outs for m in re.findall(r"^\[[a-z ->]+\]", o, re.M | re.I)}
ok("no marker outside the closed set",
   all(any(m.startswith(s) for s in STATUSES) for m in markers), str(markers))

# --- read ----------------------------------------------------------------
meta = S.read("gamma-ray burst")
ok("read resolves a bare page name",
   meta.startswith("[OK] wiki/gamma-ray burst.md"), meta[:60])
ok("meta shows the summary, tags and link counts",
   "gamma-ray burst is" in meta and "astro" in meta and "linked from" in meta, meta)
ok("meta is cheap", len(meta) < 300, f"{len(meta)} chars")
ok("an unknown page is [STOP], not an empty result",
   S.read("no such page").startswith("[STOP]"))

outline = S.read("gamma-ray burst", depth="outline")
ok("outline lists sections with sizes", outline.count("Section") >= 4, outline)
ok("outline points at the next call", 'depth="part"' in outline)

part = S.read("gamma-ray burst", depth="part", target="Section 1")
ok("part returns the named section", "Section 1" in part and "Section 2" not in part)
ok("a wrong heading is [RETRY] and lists the real ones",
   S.read("gamma-ray burst", depth="part", target="Nope").startswith("[RETRY]")
   and "Section 0" in S.read("gamma-ray burst", depth="part", target="Nope"))

full1 = S.read("gamma-ray burst", depth="full")
ok("a full read splits into parts", "[more]" in full1, full1[-200:])
ok("the continuation names the exact next call", 'from_part=2' in full1)
last = S.read("gamma-ray burst", depth="full", from_part=9)
ok("the last part says it is the last", "[DONE]" in last)

body = idx.db.execute("SELECT 1").fetchone()  # keep sqlite happy
whole = "".join(re.split(r"\n\[more\].*|\[OK\].*\n|\[DONE\].*", full1))
parts = split_parts((vault / "wiki" / "gamma-ray burst.md").read_text().split("---", 2)[2])
ok("reassembling every part recovers the text",
   "".join(p for p, _ in parts) ==
   (vault / "wiki" / "gamma-ray burst.md").read_text().split("---", 2)[2])

win = S.read("gamma-ray burst", depth="window", target="decay")
ok("window centres on the target", "decay" in win)
ok("a missing window target is [STOP]",
   S.read("gamma-ray burst", depth="window", target="zzzz").startswith("[STOP]"))
ok("a bad depth is [RETRY] listing the valid values",
   S.read("gamma-ray burst", depth="sideways").startswith("[RETRY]"))

# --- a reworded search warns, it does not starve -------------------------
S2 = Sentinel(idx, enc)
first = S2.search("thinking block generation slowdown")
again = S2.search("generation slowdown thinking block tokens")
ok("a reworded query still returns its results",
   again.startswith("[OK]") or again.startswith("[STOP] nothing"), again[:80])
ok("but it is told they are the same pages",
   "[note] nearly the query you just ran" in again, again[-160:])
ok("and the first query is named", "thinking block generation slowdown" in again)
ok("a genuinely different query gets no such note",
   "[note] nearly" not in S2.search("Markov Chain Monte Carlo sampler"))

S4 = Sentinel(idx, enc)
for _ in range(5):
    S4.search("afterglow decay curve shape")
last = S4.search("decay curve shape afterglow again")
ok("a sustained run of rewordings IS refused", last.startswith("[STOP]"), last[:90])
ok("and the refusal says what to do instead",
   "does not contain it" in last, last)

S5 = Sentinel(idx, enc)
S5.search("afterglow decay curve shape")
S5.scope = "another conversation"
ok("another conversation does not inherit the history",
   "[note] nearly" not in S5.search("afterglow decay curve shape"))

S6 = Sentinel(idx, enc)
ok("a malformed call is not counted as a search",
   S6.search("afterglow", depth="outline").startswith("[RETRY]")
   and "[note] nearly" not in S6.search("afterglow"))

# --- borrowing another tool's enum, the documented failure mode ----------
r = S.read("gamma-ray burst", depth="summary")
ok("a search depth passed to read is [RETRY], not silently accepted",
   r.startswith("[RETRY]"), r)
ok("and the confusion is named, so the next call can recover",
   "is a `search` depth" in r and 'depth="meta"' in r, r)
r = S.search("afterglow", depth="outline")
ok("the same confusion the other way round is named too",
   r.startswith("[RETRY]") and "is a `read` depth" in r, r)
ok("an enum value belonging to no tool gets the plain list",
   "is a `search` depth" not in S.read("gamma-ray burst", depth="sideways"))

# --- ranked output caps the summary it displays --------------------------
(vault / "wiki" / "wordy.md").write_text(
    "---\ntype: note\nsummary_provisional: 1\nsummary: "
    + "This opening sentence runs on and on the way the first line of a paper "
      "does, carrying clauses that say very little about what the page is for "
      "and everything about how the abstract was phrased. " * 2
    + "\n---\n\n# wordy\n\nBody.\n", encoding="utf-8")
idx.sync()
line = [x for x in S.listing(by="path", value="wiki").splitlines()
        if "wordy" in x][0]
ok("a long summary is capped in ranked output", len(line) < 220, f"{len(line)} chars")
ok("and the cut is visible rather than looking complete", "…" in line, line)
short = [x for x in S.listing(by="path", value="wiki").splitlines()
         if "afterglow.md" in x]
ok("a short summary is untouched", short and "…" not in short[0], str(short))
(vault / "wiki" / "wordy.md").unlink()
idx.sync()

# --- the cut point --------------------------------------------------------
i, hard = cut_point("x" * 3000 + "\n\n" + "y" * 8000, 6400)
ok("a paragraph boundary past the floor is preferred over a hard cut",
   not hard and i == 3000, f"cut at {i}, hard={hard}")
i, hard = cut_point("x" * 100 + "\n\n" + "y" * 8000, 6400)
ok("a boundary earlier than 25% of the budget is REJECTED", hard,
   f"cut at {i}")
i, hard = cut_point("z" * 20000, 6400)
ok("a hard cut is flagged", hard)
ok("a hard cut says so in the output",
   "[note] cut mid-sentence" in
   __import__("sentinel.output", fromlist=["x"]).sequential("t", "b", 1, 2, "n", True))

# --- listing --------------------------------------------------------------
tags = S.listing(by="all_tags")
ok("all_tags counts every tag", "astro" in tags and "paper" in tags)
ok("by=tag needs a value", S.listing(by="tag").startswith("[RETRY]"))
ok("by=path is recursive and folder-normalised",
   S.listing(by="path", value="wiki").count("wiki/") >= 4,
   S.listing(by="path", value="wiki"))
ok("an unknown tag is [STOP]", S.listing(by="tag", value="zzz").startswith("[STOP]"))
ok("a bad by is [RETRY]", S.listing(by="folder").startswith("[RETRY]"))
ok("ranked output states the total and does not invite a next page",
   "not shown" not in tags and "[more]" not in tags)

# --- graph ----------------------------------------------------------------
g = S.graph("gamma-ray burst")
ok("graph groups by what you would do with each group",
   "links to, page exists" in g and "links to, no page yet" in g, g)
ok("an opaque incoming name carries its title",
   "arxiv-2504.11743\n      Constraining" in g, g)
ok("a self-describing incoming name does not", "Late-time" not in g, g)
ok("incoming links are one per line, not comma-joined",
   "), " not in g and g.count("    ") >= 2, g)
ok("graph takes a path as well as a name",
   S.graph("wiki/gamma-ray burst.md").startswith("[OK] wiki/gamma-ray burst.md"),
   S.graph("wiki/gamma-ray burst.md")[:60])
ok("and a path from search leads straight into graph",
   "links to, page exists" in S.graph("wiki/gamma-ray burst.md"),
   S.graph("wiki/gamma-ray burst.md"))

gq = S.graph("fireball model")
ok("a referenced concept with no page is [OK], not an error",
   gq.startswith("[OK]") and "no page yet" in gq, gq)
ok("an unreferenced unknown is the only real failure",
   S.graph("nothing at all").startswith("[STOP]"))
ok("an isolated page is still [OK]", S.graph("lonely").startswith("[OK]"))

shape = S.graph()
ok("vault shape ranks the absences and counts orphans",
   "NAMES WITH NO PAGE" in shape and "orphan pages" in shape, shape)
ok("and says plainly that they cannot be read",
   "cannot be read" in shape, shape)
ok("an empty subject means the same as no subject",
   S.graph(subject="") == shape and S.graph(subject="   ") == shape,
   S.graph(subject="")[:80])
ok("each entry is on its own line with its count in brackets",
   all("(" in ln and "pages)" in ln
       for ln in shape.splitlines() if ln.startswith("    ")), shape)
ok("a name containing a space cannot be read as name-plus-number",
   "fireball model (" in shape, shape)
ok("nothing is claimed unlisted when everything is shown",
   "not listed" not in shape, shape)

(vault / "wiki" / "many.md").write_text(
    "---\ntype: note\nsummary: Many links.\n---\n\n# many\n\n"
    + " ".join(f"[[concept number {i}]]" for i in range(12)) + "\n",
    encoding="utf-8")
idx.sync()
big = S.graph()
ok("with more than fits, the unlisted remainder is named",
   "not listed" in big and "from memory" in big, big)
ok("and only the shown ones are listed",
   len([ln for ln in big.splitlines() if ln.startswith("    ")]) == 8, big)
(vault / "wiki" / "many.md").unlink()
idx.sync()

# --- filename-only resolution, and what rescues it ------------------------
(vault / "wiki" / "MCMC.md").write_text(
    "---\ntype: concept\nsummary: The sampler.\nsummary_provisional: 0\n---\n\n"
    "# MCMC\n\nA method.\n", encoding="utf-8")
(vault / "wiki" / "uses-mcmc.md").write_text(
    "---\ntype: note\nsummary: Uses it.\nsummary_provisional: 0\n---\n\n"
    "# Uses\n\nWe run [[Markov Chain Monte Carlo]] here.\n", encoding="utf-8")
idx.sync()
ok("a long-form link to an abbreviated filename does NOT resolve by itself",
   S.graph("Markov Chain Monte Carlo").startswith("[OK]")
   and "no page yet" in S.graph("Markov Chain Monte Carlo"),
   S.graph("Markov Chain Monte Carlo"))

idx.db.execute("INSERT OR REPLACE INTO aliases (alias_key, canonical_key, kind, added) "
               "VALUES (?,?,?,?)",
               (normalize("Markov Chain Monte Carlo"), normalize("MCMC"), "acronym", "now"))
idx.refresh_resolved()
idx.db.commit()
ok("one alias entry brings every earlier link to life, with no rewrite pass",
   idx.db.execute("SELECT resolved FROM links WHERE target_key=?",
                  (normalize("Markov Chain Monte Carlo"),)).fetchone()["resolved"] == 1)

# --- a gap the vault keeps leaning on is named, a passing one is not -----
gapv = tmp / "gaps"
(gapv / "wiki").mkdir(parents=True)
for i in range(5):
    extra = " And [[jet break]] once." if i == 0 else ""
    (gapv / "wiki" / f"p{i}.md").write_text(
        f"---\ntype: note\nsummary: Page {i}.\nsummary_provisional: 0\n"
        f"---\n\n# p{i}\n\nWe rely on the [[forward shock]] here.{extra}\n",
        encoding="utf-8")
gidx = Index(gapv, tmp / "gaps.db")
gidx.sync()
gidx.db.execute("UPDATE files SET analyzed_hash = content_hash")
gidx.db.commit()
gout = Sentinel(gidx).listing(by="recent")
ok("a concept five pages lean on is named unasked",
   "forward shock (5)" in gout, gout)
ok("one mentioned once is not", "jet break" not in gout, gout)
note = [ln for ln in gout.splitlines() if ln.startswith("[note]")]
ok("the gap is the only thing the line has to say",
   len(note) == 1 and note[0] == "[note] referenced but still undefined: "
                                 "forward shock (5)", str(note))

# --- the health line ------------------------------------------------------
ok("pending analysis surfaces itself in output the model already reads",
   "[note]" in meta and "awaiting analysis" in meta, meta)
idx.db.execute("UPDATE files SET analyzed_hash = content_hash")
idx.db.commit()
ok("and costs nothing once there is nothing to report",
   "[note]" not in S.read("gamma-ray burst"), S.read("gamma-ray burst"))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
if not FAIL:
    print(S.graph("gamma-ray burst"))
    print()
    print(S.listing(by="recent", limit=3))
sys.exit(1 if FAIL else 0)
