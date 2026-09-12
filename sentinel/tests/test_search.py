"""Run: python3 -m sentinel.tests.test_search

The encoder here is FAKE and deterministic. It verifies the plumbing - fusion,
depths, output shape - and says nothing about retrieval quality. That question
only has an answer on the real vault with the real model, which is what
field_test_search.py is for.
"""

from __future__ import annotations

import re
import sys
import zlib
import tempfile
from pathlib import Path

from ..index import Index
from ..search import format_hits, literal_ranking, rrf, search

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


class FakeEncoder:
    """Bag-of-words in a fixed 384-slot space. Deterministic, no download."""

    VOCAB = "gamma ray burst afterglow lorentz factor jet break turkce sonlanma".split()

    def encode(self, texts, kind="passage"):
        import numpy as np
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"\w+", t.lower())
            for w in words:
                out[i, zlib.crc32(w.encode()) % 384] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-9, None)


tmp = Path(tempfile.mkdtemp())
vault = tmp / "vault"
(vault / "wiki").mkdir(parents=True)

PAGES = {
    "wiki/lorentz.md": ("Constraining the initial Lorentz factor of bursts.", "0",
                        "# Lorentz\n\n## Method\n\nWe estimate the initial Lorentz "
                        "factor from the onset bump feature of the afterglow.\n"),
    "wiki/jets.md": ("Jet break timing and its use as a geometry probe.", "0",
                     "# Jets\n\n## Break\n\nThe jet break appears in the light "
                     "curve when the beaming cone widens.\n"),
    "wiki/notes-tr.md": ("Türkçe not: sonlanma koşulu ve döngü kontrolü.", "0",
                         "# Notlar\n\n## Döngü\n\nSonlanma koşulu sağlanmazsa "
                         "model kendisiyle tartışır.\n"),
    "wiki/draft.md": ("The first sentence of a page nobody analysed yet.", "1",
                      "# Draft\n\n## Body\n\nThe jet break is mentioned here too, "
                      "briefly.\n"),
}
for path, (summary, prov, body) in PAGES.items():
    (vault / path).write_text(
        f"---\ntype: note\norigin: user\nsummary: {summary}\n"
        f"summary_provisional: {prov}\n---\n\n{body}", encoding="utf-8")

idx = Index(vault, tmp / "s.db")
idx.sync()
enc = FakeEncoder()
from ..embed import embed_pending  # noqa: E402
n = embed_pending(idx, enc)
ok("every chunk got a vector", n > 0 and not idx.pending_embed(), f"{n} encoded")
ok("a second call re-encodes nothing", embed_pending(idx, enc) == 0)

# --- RRF is rank-based ----------------------------------------------------
ok("rrf prefers what both lists rank high",
   rrf([["a", "b", "c"], ["b", "a", "c"]])[0] in ("a", "b"))
ok("rrf promotes a consensus item over a single first place",
   rrf([["x", "b", "c"], ["y", "b", "c"]])[0] == "b",
   str(rrf([["x", "b", "c"], ["y", "b", "c"]])))

# --- headings and the page name are searchable ---------------------------
ok("a page is findable by its own name",
   any("notes-tr" in p for p, _ in literal_ranking(idx, "notes-tr")),
   str(literal_ranking(idx, "notes-tr")))
ok("a word that appears only in a heading is findable",
   any("lorentz" in p for p, _ in literal_ranking(idx, "Method")),
   str(literal_ranking(idx, "Method")))
ok("the heading is joined to its own section, not to another",
   literal_ranking(idx, "Break")[0][0] == "wiki/jets.md",
   str(literal_ranking(idx, "Break")))

# --- literal half ---------------------------------------------------------
lit = literal_ranking(idx, "jet break")
ok("literal search finds the exact wording", any("jets" in p for p, _ in lit), str(lit))
ok("a malformed query does not crash", literal_ranking(idx, '"') == [])

# --- the ladder -----------------------------------------------------------
hits = search(idx, "jet break", encoder=enc, depth="summary", limit=3)
ok("rung 1 returns the page whose wording matches",
   any(h.path == "wiki/jets.md" and h.literal for h in hits),
   str([(h.path, h.literal) for h in hits]))
ok("rung 1 carries no body text", all(not h.chunks for h in hits))

body_hits = search(idx, "jet break", encoder=enc, depth="body", limit=3)
ok("rung 2 carries verbatim chunks", any(h.chunks for h in body_hits))
ok("chunks are capped per file", all(len(h.chunks) <= 2 for h in body_hits))

long_summary = ("A first sentence taken verbatim from a paper, which runs to "
                "the length papers run to and says more about the abstract "
                "than about what this page is actually for in the vault. ")
capped = format_hits(
    [type(hits[0])(path="wiki/x.md", summary=long_summary, provisional=True,
                   literal=True)], "x")
ok("search caps a long summary line too", len(capped.splitlines()[1]) < 220,
   f"{len(capped.splitlines()[1])} chars")
ok("and marks the cut", "…" in capped)

out = format_hits(hits, "jet break")
ok("no similarity score is ever printed",
   not re.search(r"0\.\d|similarity|score", out, re.I), out)
ok("the summary is what the model reads", "geometry probe" in out)

prov = [h for h in search(idx, "draft", encoder=enc, limit=4) if h.path == "wiki/draft.md"]
ok("a provisional summary is marked as such",
   prov and "(provisional summary)" in format_hits(prov, "draft"),
   format_hits(prov, "draft") if prov else "not found")
ok("an analysed summary is not marked",
   "(provisional summary)" not in format_hits(
       [h for h in hits if h.path == "wiki/jets.md"], "jet break"))

none = search(idx, "quantum chromodynamics", encoder=enc)
ok("a query nothing matches is flagged as meaning-only, not passed off as a hit",
   "[note] no wording match" in format_hits(none, "quantum chromodynamics"),
   format_hits(none, "quantum chromodynamics"))
ok("a real hit is not flagged that way",
   "[note]" not in format_hits(hits, "jet break"))

# --- turkish --------------------------------------------------------------
tr = search(idx, "sonlanma koşulu", encoder=enc, limit=3)
ok("a Turkish query reaches the Turkish page", tr and tr[0].path == "wiki/notes-tr.md",
   str([h.path for h in tr]))

# --- the literal half alone still works (no encoder) -----------------------
ok("search degrades to literal-only without an encoder",
   {h.path for h in search(idx, "jet break", encoder=None, limit=3)}
   == {"wiki/jets.md", "wiki/draft.md"},
   str([h.path for h in search(idx, "jet break", encoder=None, limit=3)]))

# --- a filed source is held to the deeper rung, like a record -------------
qv = tmp / "quiet"
for d in ("notes", "sources", "conversations"):
    (qv / d).mkdir(parents=True)
(qv / "notes" / "kept.md").write_text(
    "---\ntype: note\nsummary: What was decided about shocks.\n"
    "summary_provisional: 0\n---\n\n# kept\n\nThe forward shock matters.\n",
    encoding="utf-8")
(qv / "sources" / "paper.md").write_text(
    "---\ntype: source\nsummary: A filed paper.\nsummary_provisional: 0\n"
    "---\n\n# paper\n\nThe forward shock is the outward boundary, and the "
    "Sedov length appears only here.\n", encoding="utf-8")
(qv / "conversations" / "c1.md").write_text(
    "---\ntype: transcript\nsummary: A talk.\nsummary_provisional: 0\n"
    "---\n\n# c1\n\nWe mentioned the forward shock once.\n",
    encoding="utf-8")
qidx = Index(qv, tmp / "quiet.db")
qidx.sync()
QUIET = ("conversations", "sources")

r1 = [h.path for h in search(qidx, "forward shock", depth="summary",
                             quiet=QUIET)]
ok("rung 1 returns only what was kept", r1 == ["notes/kept.md"], str(r1))
r2 = [h.path for h in search(qidx, "forward shock", depth="body", quiet=QUIET)]
ok("rung 2 reaches the source", "sources/paper.md" in r2, str(r2))
ok("and the record", "conversations/c1.md" in r2, str(r2))

only = [h.path for h in search(qidx, "Sedov length", depth="summary",
                               quiet=QUIET)]
ok("something only a source says is absent from rung 1", only == [], str(only))
ok("and present at rung 2",
   "sources/paper.md" in [h.path for h in search(qidx, "Sedov length",
                                                 depth="body", quiet=QUIET)])
none = [h.path for h in search(qidx, "forward shock", depth="summary")]
ok("with no quiet folders given, only the type rule applies",
   "sources/paper.md" in none and "conversations/c1.md" not in none, str(none))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
if not FAIL:
    print(format_hits(search(idx, "lorentz factor", encoder=enc, limit=2), "lorentz factor"))
sys.exit(1 if FAIL else 0)
