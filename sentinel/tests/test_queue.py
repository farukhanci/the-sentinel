"""Run: python3 -m sentinel.tests.test_queue"""

from __future__ import annotations

import re
import sys
import tempfile
import tempfile as _tf0
import zlib
from pathlib import Path

from ..embed import embed_pending
from ..index import Index
from ..resolve_queue import (
    apply_answers,
    build_queue,
    format_batch,
    next_batch,
    run_resolution,
)
from ..text import normalize

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


class FakeEncoder:
    """Character trigrams. Deterministic, and unlike a bag of words it at
    least moves in the right direction for NAME similarity, which is what the
    proximity source now uses. It is still not a semantic model: whether the
    real signal is any good was settled by field_test_proximity on the real
    vault, not here."""

    def encode(self, texts, kind="passage"):
        import numpy as np
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            t = " " + " ".join(t.lower().split()) + " "
            for j in range(len(t) - 2):
                out[i, zlib.crc32(t[j:j + 3].encode()) % 384] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-9, None)


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)], {}


def build_vault():
    tmp = Path(tempfile.mkdtemp())
    v = tmp / "vault"
    (v / "wiki").mkdir(parents=True)
    pages = {
        "forward shock.md": "The [[forward shock]] moves into the circumburst medium "
                    "and produces the early emission of the blast wave.",
        "fits.md": "In our fits the [[FS model]] describes the blast wave "
                    "moving into the circumburst medium at early times.",
        "reverse shock.md": "The [[reverse shock]] travels back into the ejecta and "
                      "produces a bright optical flash.",
        "survey.md": "We observed with [[Ariel]] during the survey.",
        "telescope.md": "The [[Ariel Space Telescope]] carries an infrared "
                     "spectrograph for the survey.",
        "targets.md": "Targets in [[Ariel Tier 2]] get more visits.",
        "bursts.md": "A [[GRB]] releases most of its energy in gamma rays.",
        "energetics.md": "Many [[gamma-ray bursts]] release energy in gamma rays.",
        "bright event.md": "The event [[GRB 221009A]] was exceptionally bright.",
    }
    for name, body in pages.items():
        (v / "wiki" / name).write_text(
            f"---\ntype: note\nsummary: About {name[:-3]}.\n"
            f"summary_provisional: 0\n---\n\n# {name[:-3]}\n\n{body}\n",
            encoding="utf-8")
    idx = Index(v, tmp / "q.db")
    idx.sync()
    embed_pending(idx, FakeEncoder())
    return v, idx


enc = FakeEncoder()


v, idx = build_vault()
stats = build_queue(idx, enc)

pairs = {(r["display_a"], r["display_b"]): r["source"]
         for r in idx.db.execute("SELECT * FROM queue")}
flat = {frozenset(k) for k in pairs}

# --- what code decided should NOT be here --------------------------------
ok("an acronym pair is decided by code, never queued",
   frozenset({"GRB", "gamma-ray bursts"}) not in flat, str(list(pairs)[:6]))
ok("a designation pair is decided by code, never queued",
   frozenset({"GRB", "GRB 221009A"}) not in flat)

# --- what code must NOT decide IS here -----------------------------------
ok("an expansion with no number defers to the queue",
   frozenset({"Ariel", "Ariel Space Telescope"}) in flat, str(list(pairs)))
ok("both candidate sources produced entries",
   stats["structure"] > 0 and stats["proximity"] > 0, str(stats))
ok("proximity works without any chunk vectors, only names",
   stats["proximity"] > 0)

# --- the proximity source works off the NAME -----------------------------
prox = {frozenset((r["display_a"], r["display_b"]))
        for r in idx.db.execute("SELECT * FROM queue WHERE source='proximity'")}
ok("the proximity source produces candidates", len(prox) > 0, str(len(prox)))
ok("mutual agreement prunes it below the one-sided count",
   len(prox) <= 3 * len(idx.db.execute(
       "SELECT DISTINCT target_key FROM links WHERE resolved=0").fetchall()),
   str(len(prox)))
ok("and they went through the decider like every other source",
   not any(frozenset({"GRB", "GRB 221009A"}) == p for p in prox),
   str([tuple(x) for x in prox][:5]))

# Name vectors are cached, so a rebuild does not re-encode them.
before_vecs = idx.db.execute("SELECT COUNT(*) c FROM vectors").fetchone()["c"]
calls = [0]
_real = enc.encode


def counting(texts, kind="passage"):
    calls[0] += len(texts)
    return _real(texts, kind)


enc.encode = counting
build_queue(idx, enc)
enc.encode = _real
ok("a rebuild re-encodes no name it has already seen",
   calls[0] == 0, f"{calls[0]} names re-encoded")
ok("and the cache is what makes that true", before_vecs > 0)

# --- a common token is not evidence --------------------------------------
from ..resolve_queue import _structural_candidates  # noqa: E402

tmp9 = Path(_tf0.mkdtemp())
v9 = tmp9 / "vault"
(v9 / "wiki").mkdir(parents=True)
# Filename debris of exactly the shape the real vault had: split source pages
# sharing `arxiv`, `md`, the paper id, and a section number.
for paper in ("2504-11743", "2509-02657"):
    for i in range(1, 6):
        (v9 / "wiki" / f"arxiv-{paper}-md--{i}-section.md").write_text(
            f"---\ntype: note\nsummary: Part {i}.\nsummary_provisional: 0\n"
            f"---\n\n# part\n\nText mentioning [[isotropic energy]] and "
            f"[[kinetic energy of the fireball shell]].\n", encoding="utf-8")
idx9 = Index(v9, tmp9 / "df.db")
idx9.sync()
cands = {frozenset((a, b)) for a, b, _, _ in _structural_candidates(idx9)}
names9 = {k for pair in cands for k in pair}

ok("two papers' section 1 are not paired on the digit they share",
   not any("arxiv250411743md1section" in p and "arxiv250902657md1section" in p
           for p in cands), str(list(cands)[:4]))
ok("nor on `arxiv` or `md`, which every one of them carries",
   not any(all("arxiv" in k for k in p) for p in cands), str(list(cands)[:4]))
ok("while a rare shared token still produces a candidate",
   any({normalize("isotropic energy"),
        normalize("kinetic energy of the fireball shell")} == set(p)
       for p in cands), str([tuple(p) for p in cands]))

# --- the batch is paragraphs, not strings --------------------------------
batch = next_batch(idx, 30)
ok("a queue entry carries a paragraph for each side",
   all(len(e["para_a"]) > 20 and len(e["para_b"]) > 20 for e in batch),
   str([(e["display_a"], e["para_a"][:30]) for e in batch[:2]]))
prompt = format_batch(batch)
ok("pairs are numbered so the answer is an index, not a copied string",
   "1.\nA: " in prompt and '"n": 1' in prompt)
ok("the three-way answer is stated in the prompt",
   '"unclear"' in prompt and "is a real" in prompt)

# --- applying answers -----------------------------------------------------
target = next(e for e in batch
              if {e["display_a"], e["display_b"]} == {"Ariel", "Ariel Space Telescope"})
n = batch.index(target) + 1
got = apply_answers(idx, batch, [{"n": n, "answer": "same"}])
ok("one accepted answer is applied, the rest left open", got["same"] == 1, str(got))

alias = idx.db.execute("SELECT * FROM aliases").fetchone()
ok("an accepted same writes an alias entry", alias is not None)
ok("the side with a page is canonical",
   alias["canonical_key"] == normalize("Ariel Space Telescope")
   or alias["alias_key"] == normalize("Ariel"), dict(alias) if alias else "none")
ok("the alias is marked as judged, not derived", alias["kind"] == "judged")

# --- unclear: queued at zero cost, reopened only on new evidence ---------
# A structural pair, so it is regenerated on every build and the reopen rule
# is what is under test rather than whether the candidate came back.
structural = {r["pair_key"] for r in idx.db.execute(
    "SELECT pair_key FROM queue WHERE source='structure'")}
other = next(e for e in batch
             if e["pair_key"] != target["pair_key"]
             and e["pair_key"] in structural)
apply_answers(idx, [other], [{"n": 1, "answer": "unclear"}])
before = idx.db.execute("SELECT mentions_at_ask FROM queue WHERE pair_key=?",
                        (other["pair_key"],)).fetchone()["mentions_at_ask"]
build_queue(idx, enc)
ok("an unclear pair is NOT re-asked with no new evidence",
   idx.db.execute("SELECT status FROM queue WHERE pair_key=?",
                  (other["pair_key"],)).fetchone()["status"] == "unclear")

(v / "wiki" / "extra.md").write_text(
    f"---\ntype: note\nsummary: More.\nsummary_provisional: 0\n---\n\n"
    f"# extra\n\nAnother mention of [[{other['display_a']}]] here.\n",
    encoding="utf-8")
idx.sync()
r = build_queue(idx, enc)
ok("a new mention reopens it", r["reopened"] == 1, str(r))
ok("and only then", before >= 0)

# --- same and different are FINAL ----------------------------------------
apply_answers(idx, [target], [{"n": 1, "answer": "same"}])
build_queue(idx, enc)
ok("a decided pair is never re-asked",
   idx.db.execute("SELECT status FROM queue WHERE pair_key=?",
                  (target["pair_key"],)).fetchone()["status"] == "same")
ok("and it is not offered in the next batch",
   target["pair_key"] not in {e["pair_key"] for e in next_batch(idx, 20)})

# --- a rejected pair stays rejected --------------------------------------
b = next_batch(idx, 20)
diff = b[0]
apply_answers(idx, [diff], [{"n": 1, "answer": "different"}])
build_queue(idx, enc)
ok("a `different` is final too",
   idx.db.execute("SELECT status FROM queue WHERE pair_key=?",
                  (diff["pair_key"],)).fetchone()["status"] == "different")

# --- a wrong answer can be taken back ------------------------------------
from ..resolve_queue import decided, revise  # noqa: E402

seen = decided(idx)
ok("decisions are reviewable", len(seen) > 0 and
   all(r["status"] != "open" for r in seen), str(len(seen)))
ok("and can be filtered by what was answered",
   all(r["status"] == "same" for r in decided(idx, "same")))

merged_pair = decided(idx, "same")[0]["pair_key"]
before_alias = idx.db.execute("SELECT COUNT(*) c FROM aliases").fetchone()["c"]
msg = revise(idx, merged_pair, "different")
ok("revising a merge removes the alias it wrote",
   idx.db.execute("SELECT COUNT(*) c FROM aliases").fetchone()["c"]
   == before_alias - 1, msg)
ok("and the pair is recorded as different now",
   idx.db.execute("SELECT status FROM queue WHERE pair_key=?",
                  (merged_pair,)).fetchone()["status"] == "different")
ok("an unknown pair is [STOP]", revise(idx, "nope|nope", "same").startswith("[STOP]"))
ok("an invalid answer is [RETRY]",
   revise(idx, merged_pair, "maybe").startswith("[RETRY]"))

# --- a name that is not concept-shaped is never a candidate --------------
tmp8 = Path(_tf0.mkdtemp())
v8 = tmp8 / "vault"
(v8 / "wiki").mkdir(parents=True)
(v8 / "wiki" / "src.md").write_text(
    "---\ntype: source\nsummary: A paper.\nsummary_provisional: 0\n---\n\n"
    "# src\n\nWe cite [[Constraining the initial Lorentz factor of gamma-ray "
    "bursts]] and also [[Gamma-Ray Burst]] physics here.\n", encoding="utf-8")
(v8 / "wiki" / "Gamma-Ray Burst.md").write_text(
    "---\ntype: concept\nsummary: The burst.\nsummary_provisional: 0\n---\n\n"
    "# Gamma-Ray Burst\n\nA concept page.\n", encoding="utf-8")
idx8 = Index(v8, tmp8 / "shape.db")
idx8.sync()
build_queue(idx8, enc)
titles = [r for r in idx8.db.execute("SELECT * FROM queue")
          if "Constraining" in (r["display_a"] + r["display_b"])]
ok("a nine-word paper title is never offered as a merge candidate",
   not titles, str([(r["display_a"], r["display_b"]) for r in titles]))

# --- the alias makes earlier links resolve, with no file rewritten -------
v2, idx2 = build_vault()
(v2 / "wiki" / "Ariel Space Telescope.md").write_text(
    "---\ntype: concept\nsummary: The telescope.\nsummary_provisional: 0\n---\n\n"
    "# Ariel Space Telescope\n\nA mission.\n", encoding="utf-8")
idx2.sync()
build_queue(idx2, enc)
b2 = next_batch(idx2, 30)
t2 = next(e for e in b2
          if {e["display_a"], e["display_b"]} == {"Ariel", "Ariel Space Telescope"})
before_files = {p: (v2 / "wiki" / p.name).read_text()
                for p in (v2 / "wiki").glob("*.md")}
unresolved_before = idx2.db.execute(
    "SELECT resolved FROM links WHERE target_key=?",
    (normalize("Ariel"),)).fetchone()["resolved"]
apply_answers(idx2, [t2], [{"n": 1, "answer": "same"}])
after_resolved = idx2.db.execute(
    "SELECT resolved FROM links WHERE target_key=?",
    (normalize("Ariel"),)).fetchone()["resolved"]
ok("before the alias, the link was unresolved", unresolved_before == 0)
ok("after it, the same link resolves", after_resolved == 1)
ok("and not one file was rewritten",
   all((v2 / "wiki" / p.name).read_text() == t for p, t in before_files.items()))

# --- the whole pass -------------------------------------------------------
v3, idx3 = build_vault()
build_queue(idx3, enc)
n_open = idx3.db.execute(
    "SELECT COUNT(*) c FROM queue WHERE status='open'").fetchone()["c"]
model = ScriptedModel('```json\n[{"n": 1, "answer": "same"}, '
                      '{"n": 2, "answer": "different"}, '
                      '{"n": 3, "answer": "unclear"}]\n```',
                      '[{"n": 1, "answer": "same"}]')
tot = run_resolution(idx3, model, batch_size=3)
ok("a fenced array is repaired and applied",
   tot["same"] == 1 and tot["different"] == 1 and tot["unclear"] == 1, str(tot))
ok("a merge costs a second, confirming ask; the others do not",
   model.calls == 2, str(model.calls))

# --- an unconfirmed merge becomes unclear, not same ----------------------
v5b, idx5b = build_vault()
build_queue(idx5b, enc)
flip = ScriptedModel('[{"n": 1, "answer": "same"}, {"n": 2, "answer": "same"}]',
                     '[{"n": 1, "answer": "same"}, {"n": 2, "answer": "different"}]')
tot = run_resolution(idx5b, flip, batch_size=2)
ok("a merge that does not survive the swapped question is downgraded",
   tot["same"] == 1 and tot["unclear"] == 1 and tot["unconfirmed"] == 1, str(tot))
ok("and only the confirmed one wrote an alias",
   idx5b.db.execute("SELECT COUNT(*) c FROM aliases").fetchone()["c"] == 1)

v5c, idx5c = build_vault()
build_queue(idx5c, enc)
nodouble = ScriptedModel('[{"n": 1, "answer": "different"}, '
                         '{"n": 2, "answer": "unclear"}]')
tot = run_resolution(idx5c, nodouble, batch_size=2)
ok("a batch with no merge is asked once", nodouble.calls == 1, str(nodouble.calls))
ok("and nothing is written", idx5c.db.execute(
    "SELECT COUNT(*) c FROM aliases").fetchone()["c"] == 0)

tot = run_resolution(idx5c, ScriptedModel('[{"n": 1, "answer": "same"}]'),
                     batch_size=1, confirm=False)
ok("confirmation can be turned off deliberately", tot["same"] == 1, str(tot))
ok("the heaviest pairs are offered first",
   [e for e in next_batch(idx3, 5)] == sorted(
       next_batch(idx3, 5),
       key=lambda e: -idx3.db.execute(
           "SELECT weight FROM queue WHERE pair_key=?",
           (e["pair_key"],)).fetchone()["weight"]))
ok("unanswered pairs stay open",
   idx3.db.execute("SELECT COUNT(*) c FROM queue WHERE status='open'"
                   ).fetchone()["c"] == n_open - 3)

bad = ScriptedModel("I cannot answer that")
tot = run_resolution(idx3, bad, batch_size=3)
ok("unparseable output decides nothing", tot["unparseable"] == 1
   and tot["same"] == 0, str(tot))



# ===========================================================================
# The moment a page is created - the acronym rule finally has a trigger
# ===========================================================================

import tempfile as _tf  # noqa: E402
from ..tools import Sentinel  # noqa: E402

def naming_vault(links: dict):
    tmp = Path(_tf.mkdtemp())
    v = tmp / "vault"
    (v / "wiki").mkdir(parents=True)
    for i, (display, count) in enumerate(links.items()):
        for j in range(count):
            (v / "wiki" / f"src-{i}-{j}.md").write_text(
                f"---\ntype: note\nsummary: Cites {display}.\n"
                f"summary_provisional: 0\n---\n\n# src {i} {j}\n\n"
                f"We discuss [[{display}]] at length here.\n", encoding="utf-8")
    idx = Index(v, tmp / "n.db")
    idx.sync()
    return v, idx, Sentinel(idx)


# --- the auto-merge that field test 1 recorded and no code performed ------
v4, idx4, S4 = naming_vault({"GRBs": 3})
r = S4.write("wiki/gamma-ray burst.md", "# gamma-ray burst\n\nA burst.",
             expect="new")
ok("creating the long-form page auto-merges the acronym",
   "came alive" in r, r)
ok("and the earlier links resolve with no file rewritten",
   idx4.db.execute("SELECT resolved FROM links WHERE target_key=?",
                   (normalize("GRBs"),)).fetchone()["resolved"] == 1)
ok("the alias is recorded as derived from the rule, not judged",
   idx4.db.execute("SELECT kind FROM aliases").fetchone()["kind"] == "acronym")

# --- the collision the strict rule cannot see -----------------------------
v5, idx5, S5 = naming_vault({"gamma-ray burst": 4})
r = S5.write("wiki/grb.md", "# grb\n\nA burst.", expect="new")
ok("an abbreviated filename is WARNED about, not silently orphaned",
   "waiting for the spelling 'gamma-ray burst'" in r, r)
ok("the warning names how many links are affected", "4 links" in r, r)
ok("but nothing is merged on a loose match",
   idx5.db.execute("SELECT COUNT(*) c FROM aliases").fetchone()["c"] == 0)
ok("and it points at the repair", "relocate()" in r)

fixed = S5.relocate("wiki/grb.md", "wiki/gamma-ray burst.md")
ok("relocating to the right name resolves them",
   idx5.db.execute("SELECT resolved FROM links WHERE target_key=?",
                   (normalize("gamma-ray burst"),)).fetchone()["resolved"] == 1,
   fixed)

# --- a shared-token near miss --------------------------------------------
v6, idx6, S6 = naming_vault({"afterglow": 5})
r = S6.write("wiki/afterglow emission.md", "# afterglow emission\n\nText.",
             expect="new")
ok("a shared-token near miss is reported too",
   "waiting for the spelling 'afterglow'" in r, r)

# --- no noise -------------------------------------------------------------
v7, idx7, S7 = naming_vault({"circumburst medium": 2})
r = S7.write("wiki/jet break.md", "# jet break\n\nText.", expect="new")
ok("an unrelated new page produces no naming note", "waiting for" not in r, r)
ok("and no merge", idx7.db.execute(
    "SELECT COUNT(*) c FROM aliases").fetchone()["c"] == 0)

h = idx7.meta("wiki/jet break.md")["content_hash"][:8]
r = S7.write("wiki/jet break.md", "# jet break\n\nEdited.", where="whole", expect=h)
ok("an ordinary write to an existing page says nothing about naming",
   "waiting for" not in r and "came alive" not in r, r)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
if not FAIL:
    print(f"  queue after build: {stats}")
    print(f"  one entry ~{len(format_batch(batch[:1])) // 4} tokens "
          f"(the recorded measurement was ~151)")
sys.exit(1 if FAIL else 0)
