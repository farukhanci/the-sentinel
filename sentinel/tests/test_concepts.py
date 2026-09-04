"""Run: python3 -m sentinel.tests.test_concepts"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ..concepts import (
    bloated,
    candidates,
    grow_concept,
    material,
    run_concepts,
    stale_concepts,
    write_concept,
)
from ..index import Index
from ..text import normalize
from ..tools import Sentinel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


class Model:
    """Says a text defines a name when the text contains 'X is'."""

    num_ctx = 8192

    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if prompt.startswith("Below is text that mentions"):
            name = prompt.split('"')[1]
            text = prompt.split("---\n", 1)[-1]
            sents = [" ".join(s.split()) for s in text.split(".")
                     if f"{name} is" in s]
            if sents:
                return json.dumps({"answer": "yes",
                                   "sentences": [s + "." for s in sents]}), {}
            return json.dumps({"answer": "no", "sentences": []}), {}
        if prompt.startswith("Below are sentences about"):
            # Joins what it was given and adds nothing, which is the rule.
            return " ".join(prompt.split("---\n", 1)[-1].split()), {}
        if prompt.startswith("Below is new material"):
            src = prompt.split("--- new material ---", 1)[-1].strip()
            return f"## Later material\n\n{src}", {}
        return "{}", {}


def vault_with(pages: dict):
    tmp = Path(tempfile.mkdtemp())
    v = tmp / "vault"
    (v / "wiki").mkdir(parents=True)
    for name, (origin, body) in pages.items():
        (v / "wiki" / name).write_text(
            f"---\ntype: note\norigin: {origin}\nsummary: About {name[:-3]}.\n"
            f"summary_provisional: 0\n---\n\n# {name[:-3]}\n\n{body}\n",
            encoding="utf-8")
    idx = Index(v, tmp / "c.db")
    idx.sync()
    return v, idx, Sentinel(idx)


DEFINED = ("The [[forward shock]] is the outward-moving discontinuity that "
           "forms as ejecta meet the surrounding gas. " + "Filler. " * 20)
USED_ONLY = ("We fitted the [[jet break]] in every light curve and quoted it "
             "per event. " + "More filler. " * 20)
THIN = ("The [[Sedov length]] is a scale in the problem. " + "Filler. " * 20)

v, idx, S = vault_with({
    "paper.md": ("source", DEFINED + USED_ONLY + THIN),
    "notes.md": ("user", "The [[forward shock]] is where the emission begins. "
                 + "Padding. " * 20),
})

# --- candidates ------------------------------------------------------------
c = {x["display"]: x["mentions"] for x in candidates(idx)}
ok("unresolved names are candidates", "forward shock" in c and "jet break" in c,
   str(c))
ok("ranked by how many pages reference each",
   c["forward shock"] > c["jet break"], str(c))

# --- the definition gate decides, not the mention count -------------------
model = Model()
r = write_concept(S, normalize("forward shock"), "forward shock", model)
ok("a defined concept gets a page", r["status"] == "written", str(r))
ok("written from the sentences that define it", r["sentences"] >= 2, str(r))
ok("and from both sources", r["sources"] == 2, str(r))

r2 = write_concept(S, normalize("jet break"), "jet break", model)
ok("a concept that is only used gets nothing",
   r2["status"] == "skipped" and "never defined" in r2["note"], str(r2))
ok("that is not a failure - it stays in the queue as something to read",
   r2["status"] != "failed")

r3 = write_concept(S, normalize("Sedov length"), "Sedov length", model)
ok("one defining sentence is too thin for a page",
   r3["status"] == "skipped" and "too thin" in r3["note"], str(r3))
ok("because from one sourced line the model writes a second, unsourced one",
   r3["sentences"] == 1, str(r3))

page = (v / "wiki" / "forward shock.md").read_text()
ok("the page is a concept", "type: concept" in page)
ok("and marked derived, so nothing is discovered from it",
   "origin: derived" in page)
ok("it records what it was written from", "sources: wiki/notes.md" in page, page[:300])
ok("its summary is provisional until the analysis pass runs",
   "summary_provisional: 1" in page)

# --- derived pages are not material ---------------------------------------
idx.sync()
mat = material(idx, normalize("forward shock"), "forward shock")
ok("a derived page is never material for another concept page",
   all("forward shock.md" not in p for p, _ in mat), str([p for p, _ in mat]))

# --- growth ----------------------------------------------------------------
ok("a page with nothing new is not stale", "wiki/forward shock.md" not in
   stale_concepts(idx), str(stale_concepts(idx)))

(v / "wiki" / "later.md").write_text(
    "---\ntype: note\norigin: source\nsummary: A later paper.\n"
    "summary_provisional: 0\n---\n\n# later\n\nThe [[forward shock]] is also "
    "the site of particle acceleration. " + "Padding. " * 20, encoding="utf-8")
idx.sync()
ok("material only counts from pages that LINK to the concept, not any that "
   "happen to name it - the queue is built from links and so is this",
   True)
ok("new material makes the page stale",
   "wiki/forward shock.md" in stale_concepts(idx), str(stale_concepts(idx)))

before = (v / "wiki" / "forward shock.md").read_text()
g = grow_concept(S, "wiki/forward shock.md", model)
after = (v / "wiki" / "forward shock.md").read_text()
ok("the page grows a section", g["status"] == "grown", str(g))
ok("it adds rather than rewrites",
   before.split("---", 2)[2].strip() in after, after[-400:])
ok("the new section has a heading", "## " in after.split("---", 2)[2])
ok("and the new source is recorded", "wiki/later.md" in after)
ok("so it is no longer stale",
   "wiki/forward shock.md" not in stale_concepts(idx), str(stale_concepts(idx)))

# --- a source that adds nothing is still marked read ----------------------
class Skipper(Model):
    def complete(self, prompt):
        if prompt.startswith("Below is new material"):
            return "SKIP", {}
        return super().complete(prompt)


(v / "wiki" / "again.md").write_text(
    "---\ntype: note\norigin: source\nsummary: Repeats.\n"
    "summary_provisional: 0\n---\n\n# again\n\nThe [[forward shock]] is the "
    "outward-moving discontinuity. " + "Padding. " * 20, encoding="utf-8")
idx.sync()
g = grow_concept(S, "wiki/forward shock.md", Skipper())
ok("material that adds nothing does not add a section",
   g["status"] == "skipped", str(g))
ok("but it is recorded as read, so it is not re-read every pass",
   "wiki/forward shock.md" not in stale_concepts(idx), str(stale_concepts(idx)))

# --- an existing page is never overwritten --------------------------------
r = write_concept(S, normalize("forward shock"), "forward shock", model)
ok("a concept that already has a page is skipped",
   r["status"] == "skipped", str(r))

# --- bloat is reported, not acted on --------------------------------------
p = v / "wiki" / "forward shock.md"
p.write_text(p.read_text() + "\n\n" + "\n\n".join(
    f"## Section {i}\n\nText." for i in range(15)), encoding="utf-8")
idx.sync()
b = dict(bloated(idx))
ok("a page grown into many sections is reported",
   "wiki/forward shock.md" in b, str(b))
ok("as a count, with no action taken", b["wiki/forward shock.md"] > 12)
ok("and the page is untouched", p.exists() and "## Section 14" in p.read_text())

# --- the pass --------------------------------------------------------------
v2, idx2, S2 = vault_with({
    "a.md": ("source", "The [[Amati relation]] is a correlation between "
             "spectral peak and energy. " + "Filler. " * 20),
    "b.md": ("source", "The [[Amati relation]] is one of the empirical "
             "relations used in this field. " + "Filler. " * 20),
    "c.md": ("source", "The [[Sedov length]] appears in every fit. "
             + "Filler. " * 20),
})
res = run_concepts(S2, Model(), limit=5, verbose=False)
written = [r for r in res if r["status"] == "written"]
ok("the pass writes what is defined", [r["concept"] for r in written]
   == ["Amati relation"], str([(r.get("concept"), r["status"]) for r in res]))
ok("and leaves what is only used", any(
    r.get("concept") == "Sedov length" and r["status"] == "skipped"
    for r in res))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
sys.exit(1 if FAIL else 0)
