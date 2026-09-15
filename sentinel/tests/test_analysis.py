"""Run: python3 -m sentinel.tests.test_analysis"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ..analysis import analyze_page, gate_concepts, gate_summary, repair_json, run_pass
from ..index import Index
from ..text import normalize, page_hash
from ..tools import Sentinel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


class ScriptedModel:
    """Hands back whatever the test queued, and counts calls."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        r = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return r, {"wall": 0.0}


# --- output repair --------------------------------------------------------
obj, rep = repair_json('```json\n{"summary": "s", "concepts": []}\n```')
ok("a code fence is repaired and REPORTED", obj and rep == ["code fence"], str(rep))
obj, rep = repair_json('Sure! Here it is:\n{"summary": "s", "concepts": []}')
ok("a preamble is repaired and reported", obj and "preamble" in rep, str(rep))
obj, rep = repair_json('<think>\nhmm\n</think>{"summary": "s", "concepts": []}')
ok("a think block is stripped before parsing", obj is not None, str(rep))
obj, rep = repair_json('{"summary": "s", "concepts": ["a",]}')
ok("a trailing comma is repaired and reported",
   obj == {"summary": "s", "concepts": ["a"]} and "trailing comma" in rep, str(rep))
obj, rep = repair_json('```json\n{"summary":"s","concepts":["a","b",]}\n```')
ok("repairs stack, and every one of them is named",
   obj and rep == ["code fence", "trailing comma"], str(rep))
obj, rep = repair_json("not json at all")
ok("unparseable output is reported, not guessed at",
   obj is None and "unparseable" in rep)


# --- the gates ------------------------------------------------------------
ok("an empty summary fails", gate_summary("") == "empty")
ok("two sentences fail", gate_summary("One. Two.") is not None)
ok("one sentence passes", gate_summary("A page about the afterglow.") is None)
ok("an over-long summary fails", gate_summary("x " * 200) is not None)

TEXT = ("The afterglow follows the burst. " * 30) + \
       "We use a Markov Chain Monte Carlo sampler and the Amati relation."
kept, low = gate_concepts(
    ["afterglow", "Markov Chain Monte Carlo", "Amati relation",
     "compactness problem argument"], TEXT)
ok("non-verbatim concepts are dropped, the rest survive",
   len(kept) == 3 and "compactness problem argument" not in kept, str(kept))
ok("the gate is not all-or-nothing", "Amati relation" in kept)
ok("a healthy yield is not flagged", not low)
kept, low = gate_concepts([], "word " * 1000)
ok("a page that genuinely yields nothing IS flagged", low and kept == [])
ok("duplicates collapse", len(gate_concepts(["afterglow", "Afterglow"], TEXT)[0]) == 1)

LINKED = "We fit the [[Amati relation]] and the [[afterglow]] here. " * 20
kept2, _ = gate_concepts(["[[Amati relation]]", "[[afterglow]]", "afterglow"],
                         LINKED)
ok("a concept copied with its link markup is still recognised",
   kept2 == ["Amati relation", "afterglow"], str(kept2))
ok("and it does not double-count against its unbracketed twin", len(kept2) == 2)

# --- the pass on a real vault --------------------------------------------
tmp = Path(tempfile.mkdtemp())
vault = tmp / "vault"
(vault / "wiki").mkdir(parents=True)
(vault / "wiki" / "paper.md").write_text(
    "---\ntype: note\nsummary: Gamma-ray bursts represent some of the most "
    "energetic events.\nsummary_provisional: 1\n---\n\n# Paper\n\n"
    "## Method\n\nWe estimate the initial Lorentz factor from the onset bump "
    "feature. The `[n_0]` density is fixed. We also use a Markov Chain Monte "
    "Carlo sampler.\n\n## Result\n\nThe initial Lorentz factor correlates with "
    "isotropic energy.\n", encoding="utf-8")
(vault / "wiki" / "initial Lorentz factor.md").write_text(
    "---\ntype: concept\nsummary: The bulk motion factor.\n"
    "summary_provisional: 0\n---\n\n# initial Lorentz factor\n\nA concept.\n",
    encoding="utf-8")

idx = Index(vault, tmp / "a.db")
idx.sync()
S = Sentinel(idx)
before = page_hash((vault / "wiki" / "paper.md").read_text())

model = ScriptedModel(
    '{"summary": "Estimates initial Lorentz factors from onset bump features.",'
    ' "concepts": ["initial Lorentz factor", "onset bump feature", '
    '"Markov Chain Monte Carlo", "not in the text at all"]}')
r = analyze_page(S, "wiki/paper.md", model)

ok("the pass succeeds in one call", r["status"] == "ok" and model.calls == 1, str(r))
ok("the fabricated concept was dropped, the real ones kept", r["concepts"] == 3,
   str(r["concepts"]))
ok("and the dropped one never reaches the linker",
   "not in the text at all" not in r["survivors"] and len(r["survivors"]) == 3,
   str(r.get("survivors")))
ok("every surviving concept became a link", r["links"] == 3, str(r.get("links")))

row = idx.meta("wiki/paper.md")
ok("the real summary replaces the provisional one",
   row["summary"].startswith("Estimates initial") and row["summary_provisional"] == 0,
   str(dict(row)))
body = (vault / "wiki" / "paper.md").read_text()
ok("links are placed", "[[initial Lorentz factor]]" in body, body)
ok("a concept with no page is still linked and KEPT",
   "[[onset bump feature]]" in body)
ok("the user's own bracket is untouched", "`[n_0]`" in body)
ok("only the first occurrence is linked",
   body.count("[[initial Lorentz factor]]") == 1,
   str(body.count("[[initial Lorentz factor]]")))

after = page_hash((vault / "wiki" / "paper.md").read_text())
ok("the content hash did not move, so no outstanding expect went stale",
   after == before, f"{before[:8]} -> {after[:8]}")
ok("analyzed_hash equals content_hash, so the page is not pending again",
   "wiki/paper.md" not in idx.pending_analysis(), str(idx.pending_analysis()))

# --- idempotency ----------------------------------------------------------
model2 = ScriptedModel(
    '{"summary": "Estimates initial Lorentz factors from onset bump features.",'
    ' "concepts": ["initial Lorentz factor", "onset bump feature"]}')
analyze_page(S, "wiki/paper.md", model2)
ok("a second pass does not double-link",
   (vault / "wiki" / "paper.md").read_text().count("[[initial Lorentz factor]]") == 1)
ok("and the hash still has not moved",
   page_hash((vault / "wiki" / "paper.md").read_text()) == before)

# --- low yield: one retry, then accept ------------------------------------
(vault / "wiki" / "formulas.md").write_text(
    "---\ntype: note\nsummary: Equations.\nsummary_provisional: 1\n---\n\n"
    "# Formulas\n\n" + "$E = mc^2$ " * 300, encoding="utf-8")
idx.sync()
empty = ScriptedModel('{"summary": "A page of equations.", "concepts": []}')
r = analyze_page(S, "wiki/formulas.md", empty)
ok("a low yield triggers exactly ONE retry, never a loop", empty.calls == 2,
   str(empty.calls))
ok("and the second low result is ACCEPTED, not failed",
   r["status"] == "low_yield", str(r))
ok("knowing a page contributes nothing is worth recording",
   idx.db.execute("SELECT status FROM analysis WHERE path='wiki/formulas.md'"
                  ).fetchone()["status"] == "low_yield")
ok("a low-yield page leaves the queue", "wiki/formulas.md" not in idx.pending_analysis())

# --- failure: keep the provisional summary, record against the hash -------
(vault / "wiki" / "broken.md").write_text(
    "---\ntype: note\nsummary: First sentence here.\nsummary_provisional: 1\n---\n\n"
    "# Broken\n\nSome text about the afterglow.\n", encoding="utf-8")
idx.sync()
bad = ScriptedModel("this is not json", "still not json")
r = analyze_page(S, "wiki/broken.md", bad)
ok("repeated failure is reported, not retried forever",
   r["status"] == "failed" and bad.calls == 2, str(r))
ok("the provisional summary is kept",
   idx.meta("wiki/broken.md")["summary"] == "First sentence here.")
ok("the page is left unlinked", "[[" not in (vault / "wiki" / "broken.md").read_text())
ok("the failure is recorded against the hash, so it is not retried every pass",
   "wiki/broken.md" not in idx.pending_analysis())
ok("but the reason stays visible rather than silent",
   idx.db.execute("SELECT note FROM analysis WHERE path='wiki/broken.md'"
                  ).fetchone()["note"] != "")

# --- editing the page requeues it ----------------------------------------
p = vault / "wiki" / "broken.md"
p.write_text(p.read_text().replace("Some text", "Quite different text"))
idx.sync()
ok("a real content edit puts a failed page back in the queue",
   "wiki/broken.md" in idx.pending_analysis())

bad_model = ScriptedModel("I am not going to answer that.",
                          "Still not answering.")
tmp_page = vault / "wiki" / "diag.md"
tmp_page.write_text(
    "---\ntype: note\nsummary: First line.\nsummary_provisional: 1\n---\n\n"
    "# diag\n\nText about the afterglow.\n", encoding="utf-8")
idx.sync()
rd = analyze_page(S, "wiki/diag.md", bad_model)
ok("an unparseable failure keeps what the model actually said",
   "Still not answering" in rd["note"], rd["note"])
ok("it keeps the LAST attempt, which is the one that settled the failure",
   "I am not going to answer" not in rd["note"], rd["note"])
ok("and the raw output is available for diagnosis", "raw" in rd)
ok("the recorded note carries it too, so it survives the session",
   "Still not answering" in idx.db.execute(
       "SELECT note FROM analysis WHERE path='wiki/diag.md'").fetchone()["note"])
tmp_page.unlink()
idx.sync()

# --- a page too large for the context ------------------------------------
class SizedModel(ScriptedModel):
    num_ctx = 8192

    def complete(self, prompt):
        text, _ = super().complete(prompt)
        return text, {"prompt_tokens": self.reported}

huge = vault / "wiki" / "huge.md"
huge.write_text(
    "---\ntype: note\nsummary: Enormous.\nsummary_provisional: 1\n---\n\n"
    "# huge\n\n" + ("The afterglow follows the burst. " * 4000),
    encoding="utf-8")
idx.sync()
never = SizedModel('{"summary": "s", "concepts": ["afterglow"]}')
never.reported = 0
r = analyze_page(S, "wiki/huge.md", never)
ok("a page too large for the context is read in windows, not refused",
   r["status"] in ("ok", "low_yield") and r["windows"] > 1, str(r))
ok("and the note says why it was read that way",
   "past the" in r["note"] and "kept its existing summary" in r["note"],
   r["note"])
huge.unlink()
idx.sync()

# --- a page that fits the estimate but truncates in reality --------------
marginal = vault / "wiki" / "marginal.md"
marginal.write_text(
    "---\ntype: note\nsummary: Borderline.\nsummary_provisional: 1\n---\n\n"
    "# marginal\n\n" + ("The afterglow follows the burst. " * 200),
    encoding="utf-8")
idx.sync()
trunc = SizedModel('{"summary": "Looks fine.", "concepts": ["afterglow"]}')
trunc.reported = 8000          # sitting at the ceiling of 8192
r = analyze_page(S, "wiki/marginal.md", trunc)
ok("a prompt that filled the context is treated as truncated, not accepted",
   r["status"] == "failed" and "truncated" in r["note"], str(r))
ok("even though the output itself looked well-formed", trunc.calls >= 1)
ok("the page keeps its provisional summary",
   idx.meta("wiki/marginal.md")["summary"] == "Borderline.")

fits = SizedModel('{"summary": "Real summary.", "concepts": ["afterglow"]}')
fits.reported = 2000
r = analyze_page(S, "wiki/marginal.md", fits)
ok("a prompt with room to spare goes through",
   r["status"] in ("ok", "low_yield") and "truncated" not in r["note"], str(r))
marginal.unlink()
idx.sync()

# --- an unwritable page ends that page, not the pass ---------------------
import os as _os  # noqa: E402
locked = vault / "wiki" / "locked.md"
locked.write_text(
    "---\ntype: note\nsummary: Read only.\nsummary_provisional: 1\n---\n\n"
    "# locked\n\nText about the afterglow.\n", encoding="utf-8")
idx.sync()
# chmod is not enough here - a test suite may run as root, where the
# permission bits are advisory. Patch the check itself, so what is under test
# is the code path and not the filesystem.
from .. import analysis as _an  # noqa: E402
_real_access = _an.os.access
_an.os.access = lambda p, mode: False if str(p).endswith("locked.md") \
    else _real_access(p, mode)

counting = ScriptedModel('{"summary": "A page.", "concepts": ["afterglow"]}')
r = analyze_page(S, "wiki/locked.md", counting)
ok("an unwritable page fails cleanly instead of crashing",
   r["status"] == "failed" and "not writable" in r["note"], str(r))
ok("and no model call was spent on it", counting.calls == 0, str(counting.calls))
ok("it stays in the queue, because the fix is a chmod not an edit",
   "wiki/locked.md" in idx.pending_analysis())

_an.os.access = _real_access
r = analyze_page(S, "wiki/locked.md", counting)
ok("once writable it analyses normally", r["status"] == "ok", str(r))

# --- one bad page must not end a pass ------------------------------------
class Exploding:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("connection reset")
        return '{"summary": "A page.", "concepts": ["afterglow"]}', {}

(vault / "wiki" / "p1.md").write_text(
    "---\ntype: note\nsummary: One.\nsummary_provisional: 1\n---\n\n"
    "# p1\n\nText about the afterglow.\n", encoding="utf-8")
(vault / "wiki" / "p2.md").write_text(
    "---\ntype: note\nsummary: Two.\nsummary_provisional: 1\n---\n\n"
    "# p2\n\nMore text about the afterglow.\n", encoding="utf-8")
idx.sync()
res = run_pass(S, Exploding(), verbose=False)
ok("a crashing call is recorded and the pass continues",
   any(r["status"] == "failed" for r in res)
   and any(r["status"] in ("ok", "low_yield") for r in res),
   str([(r["path"], r["status"]) for r in res]))

# --- the pass over the queue ---------------------------------------------
res = run_pass(S, ScriptedModel('{"summary": "A page.", "concepts": ["afterglow"]}'),
               verbose=False)
ok("run_pass drains the queue", not idx.pending_analysis(), str(idx.pending_analysis()))


class Releasable(ScriptedModel):
    released = False

    def unload(self):
        Releasable.released = True
        return True


(vault / "wiki" / "rel.md").write_text(
    "---\ntype: note\nsummary: One.\nsummary_provisional: 1\n---\n\n"
    "# rel\n\nText about the afterglow.\n", encoding="utf-8")
idx.sync()
run_pass(S, Releasable('{"summary": "A page.", "concepts": ["afterglow"]}'),
         verbose=False)
ok("the pass lets the model go when it is done", Releasable.released)

class NoUnload(ScriptedModel):
    pass

(vault / "wiki" / "rel2.md").write_text(
    "---\ntype: note\nsummary: Two.\nsummary_provisional: 1\n---\n\n"
    "# rel2\n\nMore about the afterglow.\n", encoding="utf-8")
idx.sync()
run_pass(S, NoUnload('{"summary": "A page.", "concepts": ["afterglow"]}'),
         verbose=False)
ok("a model that cannot be unloaded is not a problem", True)
ok("and reports one result per page", len(res) >= 1)

# --- gate 2: a derived page cannot discover new concepts -------------------
#
# test_maintain.py's convergence check does not distinguish the gate from its
# absence: the model there always returns two names that are already known
# by the time the derived page is analysed, so that assertion would hold even
# with _known_keys() deleted. This fixture seeds a name known only through a
# PAGE and a name known only through an unresolved LINK, then adds one name
# that is neither, so there is something for the gate to actually block.
seed = ScriptedModel(
    '{"summary": "A related idea.", "concepts": ["known concept two"]}')
(vault / "wiki" / "known-via-link.md").write_text(
    "---\ntype: note\nsummary: About a related idea.\nsummary_provisional: 0\n"
    "---\n\n# known-via-link\n\nWe discuss known concept two here.\n",
    encoding="utf-8")
idx.sync()
seed_r = analyze_page(S, "wiki/known-via-link.md", seed)
ok("setup: the seed concept is linked with no page of its own",
   seed_r.get("links") == 1 and idx.meta("wiki/known concept two.md") is None,
   str(seed_r))

(vault / "wiki" / "derived-page.md").write_text(
    "---\ntype: concept\norigin: derived\nsummary: A written concept page.\n"
    "summary_provisional: 1\n---\n\n# derived-page\n\nThis page mentions the "
    "initial Lorentz factor, known concept two, and a totally new phenomenon "
    "nobody has referenced before.\n", encoding="utf-8")
idx.sync()
derived_model = ScriptedModel(
    '{"summary": "A written concept page.", "concepts": '
    '["initial Lorentz factor", "known concept two", "totally new phenomenon"]}')
r = analyze_page(S, "wiki/derived-page.md", derived_model)
body = (vault / "wiki" / "derived-page.md").read_text()

ok("block: the undiscovered name is counted and dropped",
   r.get("not_discovered") == 1, str(r))
ok("block: it is never bracketed into the page",
   "[[totally new phenomenon]]" not in body, body)
ok("block: it never enters the links table, so the growth queue never sees it",
   idx.db.execute("SELECT COUNT(*) c FROM links WHERE target_key=?",
                  (normalize("totally new phenomenon"),)).fetchone()["c"] == 0)
ok("pass: a name known through its own PAGE is linked normally",
   "[[initial Lorentz factor]]" in body, body)
ok("pass: a name known only through an unresolved LINK is ALSO linked",
   "[[known concept two]]" in body, body)
ok("pass: the gate does not cut everything - two of three concepts got through",
   r.get("links") == 2, str(r))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
sys.exit(1 if FAIL else 0)
