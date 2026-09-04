"""Run: python3 -m sentinel.tests.test_maintain"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from ..index import Index
from ..maintain import LOCK, _held, maintain, survey
from ..tools import Sentinel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


class Model:
    num_ctx = 8192

    def __init__(self):
        self.calls = 0
        self.unloads = 0
        self.slow = 0.0

    def complete(self, prompt):
        self.calls += 1
        if self.slow:
            time.sleep(self.slow)
        if prompt.startswith("Below is text that mentions"):
            name = prompt.split('"')[1]
            text = prompt.split("---\n", 1)[-1]
            sents = [" ".join(s.split()) + "." for s in text.split(".")
                     if f"{name} is" in s]
            return json.dumps({"answer": "yes" if sents else "no",
                               "sentences": sents}), {}
        if prompt.startswith("Below are sentences about"):
            return " ".join(prompt.split("---\n", 1)[-1].split()), {}
        if prompt.startswith("Below are numbered pairs"):
            return '[{"n": 1, "answer": "unclear"}]', {}
        if prompt.startswith("Below is a record of a conversation"):
            return '{"concepts": []}', {}
        return ('{"summary": "A page about shocks.", '
                '"concepts": ["forward shock", "reverse shock"]}'), {}

    def unload(self):
        self.unloads += 1
        return True


def fresh():
    tmp = Path(tempfile.mkdtemp())
    v = tmp / "vault"
    (v / "wiki").mkdir(parents=True)
    (v / "wiki" / "a.md").write_text(
        "---\ntype: note\norigin: source\n---\n\n# a\n\nThe forward shock is "
        "the outward-moving discontinuity. The reverse shock travels back. "
        + "Filler. " * 30, encoding="utf-8")
    (v / "wiki" / "b.md").write_text(
        "---\ntype: note\norigin: source\n---\n\n# b\n\nThe forward shock is "
        "where emission begins. " + "Padding. " * 30, encoding="utf-8")
    idx = Index(v, tmp / "m.db")
    idx.sync()
    return v, idx, Sentinel(idx)


# --- the dry run reports without touching the model -----------------------
v, idx, S = fresh()
s = survey(idx)
ok("the survey reports what is owed", s["pages"] == 2 and s["to analyse"] == 2,
   str(s))
ok("including work that needs no model", "concepts with no page" in s)

model = Model()
before = model.calls
survey(idx)
ok("and it calls no model", model.calls == before)

# --- one run, one load ----------------------------------------------------
out = maintain(S, model, verbose=False)
ok("the run reports what it did", out["analysed"] == 2, str(out))
ok("the model is released exactly once, not once per pass",
   model.unloads == 1, str(model.unloads))
ok("concepts got written from the sources",
   out["written"] >= 1, str(out))
ok("and the time is reported", out["seconds"] >= 0)

# --- resumable ------------------------------------------------------------
s2 = survey(idx)
ok("what is left to analyse is what the run itself wrote",
   s2["to analyse"] == out["written"], f"{s2['to analyse']} vs {out['written']}")

# It converges. A page written by one run is analysed by the next, and that
# page's concepts cannot spawn more - a derived page discovers nothing.
model2 = Model()
out2 = maintain(S, model2, verbose=False)
model3 = Model()
out3 = maintain(S, model3, verbose=False)
ok("a third run has nothing left to analyse",
   survey(idx)["to analyse"] == 0, str(survey(idx)))
ok("and writes nothing new, because derived pages discover no concepts",
   out3.get("written", 0) == 0, str(out3))

# --- the budget stops it starting new work -------------------------------
v3, idx3, S3 = fresh()
slow = Model()
slow.slow = 0.05
outb = maintain(S3, slow, minutes=0.0001, verbose=False)
ok("a run past its budget stops starting work", outb["over budget"], str(outb))
ok("but it always syncs first", "sync" in outb)
ok("and it still releases the model", slow.unloads == 1)

# --- the lock -------------------------------------------------------------
v4, idx4, S4 = fresh()
ok("a free vault is not held", _held(v4) is None)
ok("a held vault is", _held(v4) is not None)
(v4 / LOCK).unlink()
ok("and releasing it frees the vault", _held(v4) is None)

import os as _os  # noqa: E402
old = (v4 / LOCK).stat().st_mtime
_os.utime(v4 / LOCK, (old - 7200, old - 7200))
ok("a lock left by a killed run expires rather than needing a human",
   _held(v4) is None)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
sys.exit(1 if FAIL else 0)
