"""Run: python3 -m sentinel.tests.test_transcript"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ..analysis import analyze_page, extract_concepts, windows
from ..index import Index
from ..search import search
from ..tools import Sentinel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


class WindowModel:
    """Returns the capitalised words it sees, so what comes back depends on
    which window it was given."""

    # Chosen so the window arithmetic gives ~300 characters: the prompt
    # overhead and output reserve come off the context first.
    num_ctx = 875

    def __init__(self):
        self.seen: list[str] = []

    def complete(self, prompt):
        import json as _j
        import re as _re
        self.seen.append(prompt)
        body = prompt.split("---\n", 1)[-1]
        found = sorted(set(_re.findall(r"\b[A-Z][a-z]+ [a-z]+\b", body)))
        return _j.dumps({"concepts": found}), {}


# --- windows --------------------------------------------------------------
w = windows("x" * 1000, 300)
ok("a long text is cut into windows", len(w) == 4, str([len(x) for x in w]))
ok("they overlap, so a name split by a boundary survives",
   sum(len(x) for x in w) > 1000)
ok("a short text stays one window", windows("short", 300) == ["short"])
ok("overlap is a tenth, not a crawl", len(windows("y" * 10000, 1000)) < 20,
   str(len(windows("y" * 10000, 1000))))

# --- a transcript is analysed differently ---------------------------------
tmp = Path(tempfile.mkdtemp())
vault = tmp / "vault"
(vault / "wiki").mkdir(parents=True)
(vault / "conversations").mkdir()

spoken = ("We discussed the Amati relation at length. " * 8 +
          "Then the Lorentz factor came up. " * 8 +
          "Finally the Sedov length was raised. " * 8)
(vault / "conversations" / "2026-09-01-1.md").write_text(
    "---\ntype: transcript\nsummary: A conversation.\nsummary_provisional: 0\n"
    "origin: conversation\n---\n\n# 2026-09-01-1\n\n" + spoken, encoding="utf-8")
(vault / "wiki" / "findings.md").write_text(
    "---\ntype: note\nsummary: What we settled.\nsummary_provisional: 0\n"
    "origin: conversation\n---\n\n# findings\n\nThe Amati relation held.\n\n"
    "Kayıt: [[2026-09-01-1]]\n", encoding="utf-8")

idx = Index(vault, tmp / "t.db")
idx.sync()
S = Sentinel(idx)

model = WindowModel()
r = analyze_page(S, "conversations/2026-09-01-1.md", model)
ok("a long transcript is read window by window, not refused",
   r["status"] in ("ok", "low_yield") and r["windows"] > 1, str(r))
ok("one call per window", r["calls"] == r["windows"], str(r))
ok("no summary was written for it",
   idx.meta("conversations/2026-09-01-1.md")["summary"] == "A conversation.")
body = (vault / "conversations" / "2026-09-01-1.md").read_text()
ok("concepts are linked where they were said", "[[Amati relation]]" in body, body[:200])
ok("a concept from a later window is found too", "[[Sedov length]]" in body)
ok("first occurrence only", body.count("[[Amati relation]]") == 1)

# --- a second model gives the windowed path its summary ------------------
class Summariser:
    """A large-context model that reads whole documents."""

    num_ctx = 120000

    def __init__(self):
        self.seen = []

    def complete(self, prompt):
        import json as _j
        text = prompt.split("---\n", 1)[-1]
        self.seen.append(text)
        tail = text.strip().split(".")[-2].strip() if "." in text else "x"
        return _j.dumps({"summary": f"A conversation ending on {tail[:80]}."}), {}


(vault / "conversations" / "2026-09-05-1.md").write_text(
    "---\ntype: transcript\norigin: conversation\nsummary: hi Hello!\n"
    "summary_provisional: 1\n---\n\n# 2026-09-05-1\n\n## You\n\nhi\n\n"
    "## The Sentinel\n\nHello!\n\n## You\n\n" + spoken, encoding="utf-8")
idx.sync()
sm = Summariser()
r = analyze_page(S, "conversations/2026-09-05-1.md", WindowModel(),
                 summary_model=sm)
ok("the windowed path now writes a summary", r.get("summarised") == "ok", str(r))
ok("the placeholder is gone",
   idx.meta("conversations/2026-09-05-1.md")["summary"] != "hi Hello!",
   str(idx.meta("conversations/2026-09-05-1.md")["summary"]))
ok("and it is no longer provisional",
   idx.meta("conversations/2026-09-05-1.md")["summary_provisional"] == 0)
ok("the summary model saw the WHOLE conversation, not a window",
   len(sm.seen) == 1 and "Sedov length" in sm.seen[0], str(len(sm.seen)))
ok("concepts still come from the windows, not from that call",
   r["windows"] > 1 and r["calls"] == r["windows"], str(r))
ok("the conversation text is untouched",
   "## The Sentinel" in
   (vault / "conversations" / "2026-09-05-1.md").read_text())

# without a summary model the old behaviour stands
(vault / "conversations" / "2026-09-05-2.md").write_text(
    "---\ntype: transcript\norigin: conversation\nsummary: placeholder.\n"
    "summary_provisional: 1\n---\n\n# 2026-09-05-2\n\n" + spoken,
    encoding="utf-8")
idx.sync()
r = analyze_page(S, "conversations/2026-09-05-2.md", WindowModel())
ok("no second model means no summary, as before",
   "summarised" not in r
   and idx.meta("conversations/2026-09-05-2.md")["summary"] == "placeholder.")

# a page longer than the summary model's context is cut, and says so
class Tiny(Summariser):
    num_ctx = 900


(vault / "conversations" / "2026-09-05-3.md").write_text(
    "---\ntype: transcript\norigin: conversation\nsummary: p.\n"
    "summary_provisional: 1\n---\n\n# 2026-09-05-3\n\n" + spoken * 3,
    encoding="utf-8")
idx.sync()
r = analyze_page(S, "conversations/2026-09-05-3.md", WindowModel(),
                 summary_model=Tiny())
ok("too long for the summary model is summarised as far as it fits",
   r.get("summarised") == "ok", str(r))
ok("and the result says it was cut", "longer than" in r["note"], r["note"])

# --- the ladder ------------------------------------------------------------
ok("rung 1 does not return the transcript",
   [h.path for h in search(idx, "Amati relation", depth="summary")]
   == ["wiki/findings.md"],
   str([h.path for h in search(idx, "Amati relation", depth="summary")]))
ok("rung 2 does",
   "conversations/2026-09-01-1.md" in
   [h.path for h in search(idx, "Amati relation", depth="body")])
ok("something said but not written up is unreachable at rung 1",
   not search(idx, "Sedov length", depth="summary"),
   str([h.path for h in search(idx, "Sedov length", depth="summary")]))
hits = [h.path for h in search(idx, "Sedov length", depth="body")]
ok("and reachable at rung 2",
   "conversations/2026-09-01-1.md" in hits, str(hits))
ok("rung 2 returns transcripts and only transcripts here",
   all(h.startswith("conversations/") for h in hits), str(hits))

# --- the summary of a conversation places no links -----------------------
from ..concepts import material  # noqa: E402
from ..text import normalize as _n  # noqa: E402


class Fixed:
    num_ctx = 8192

    def complete(self, prompt):
        import json as _j
        return _j.dumps({"summary": "A page.",
                         "concepts": ["Amati relation"]}), {}


(vault / "conversations" / "dup-1.md").write_text(
    "---\ntype: transcript\norigin: conversation\nsummary: Talk.\n"
    "summary_provisional: 0\n---\n\n# dup-1\n\n"
    "The Amati relation is a correlation we discussed.\n", encoding="utf-8")
(vault / "wiki" / "dup-summary.md").write_text(
    "---\ntype: note\norigin: conversation\nsummary: What we settled.\n"
    "summary_provisional: 1\n---\n\n# dup-summary\n\n"
    "The Amati relation is a correlation we discussed.\n\n"
    "Kayıt: [[dup-1]]\n", encoding="utf-8")
idx.sync()
rt = analyze_page(S, "conversations/dup-1.md", Fixed())
rs = analyze_page(S, "wiki/dup-summary.md", Fixed())

ok("the transcript links its concepts", rt["links"] >= 1, str(rt))
ok("the summary does not", rs.get("links", 0) == 0, str(rs))
ok("and it says why", "linked at the transcript" in rs.get("links_skipped", ""),
   str(rs.get("links_skipped")))
ok("the summary still gets a summary of its own",
   idx.meta("wiki/dup-summary.md")["summary"] == "A page.")
ok("it is still connected to the record it came from",
   _n("dup-1") in [r["target_key"] for r in idx.links_out("wiki/dup-summary.md")],
   str([r["target_key"] for r in idx.links_out("wiki/dup-summary.md")]))

srcs = [r["source"] for r in idx.db.execute(
    "SELECT source FROM links WHERE target_key=?", (_n("Amati relation"),))]
ok("one conversation counts as ONE source in the growth queue",
   srcs.count("wiki/dup-summary.md") == 0 and
   "conversations/dup-1.md" in srcs, str(srcs))
mats = [p for p, _ in material(idx, _n("Amati relation"), "Amati relation")]
ok("and the definition gate reads it once, not twice",
   "wiki/dup-summary.md" not in mats, str(mats))

# --- the link between them -------------------------------------------------
out = S.graph("findings")
ok("the page points at the record it came from", "2026-09-01-1" in out, out)
ok("and the record is a real page, not an unresolved name",
   "no page yet" not in out.split("2026-09-01-1")[0].split("links to")[-1], out)

# --- a page that grew past the context is read, not refused --------------
(vault / "wiki" / "grown.md").write_text(
    "---\ntype: note\norigin: conversation\nsummary: An older summary.\n"
    "summary_provisional: 0\n---\n\n# grown\n\n" + spoken, encoding="utf-8")
idx.sync()
r = analyze_page(S, "wiki/grown.md", WindowModel())
ok("an oversized page is read in windows rather than failed",
   r["status"] in ("ok", "low_yield") and r["windows"] > 1, str(r))
ok("and it says why it was read that way",
   "past the" in r["note"] and "windows" in r["note"], r["note"])
ok("its existing summary is kept, since a windowed read cannot make one",
   idx.meta("wiki/grown.md")["summary"] == "An older summary.")
ok("but its concepts are still linked",
   "[[Amati relation]]" in (vault / "wiki" / "grown.md").read_text())

# --- every window failing is a failure, one failing is not ----------------
class HalfBad(WindowModel):
    def complete(self, prompt):
        self.seen.append(prompt)
        if len(self.seen) == 1:
            return "not json", {}
        return super().complete(prompt)[0], {}


(vault / "conversations" / "2026-09-01-2.md").write_text(
    "---\ntype: transcript\nsummary: Another.\nsummary_provisional: 0\n---\n\n"
    "# 2026-09-01-2\n\n" + spoken, encoding="utf-8")
idx.sync()
r = analyze_page(S, "conversations/2026-09-01-2.md", HalfBad())
ok("one unreadable window does not lose the others",
   r["status"] in ("ok", "low_yield") and r["failed_windows"] == 1, str(r))

class AllBad(WindowModel):
    def complete(self, prompt):
        self.seen.append(prompt)
        return "not json", {}

(vault / "conversations" / "2026-09-01-3.md").write_text(
    "---\ntype: transcript\nsummary: Third.\nsummary_provisional: 0\n---\n\n"
    "# 2026-09-01-3\n\n" + spoken, encoding="utf-8")
idx.sync()
r = analyze_page(S, "conversations/2026-09-01-3.md", AllBad())
ok("every window failing IS a failure", r["status"] == "failed", str(r))
ok("and it says so", "window" in r["note"], r["note"])

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
sys.exit(1 if FAIL else 0)
