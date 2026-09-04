"""Run: python3 -m sentinel.tests.test_harness

The model is scripted. What is under test is the LOOP: dispatch, the guard,
what happens to a malformed call, and whether the budget is countable. None of
that needs a real model, and all of it is what a real model breaks.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ..harness import Harness, ollama_tools
from ..index import Index
from ..schema import REQUIRED, TOOLS
from ..tools import Sentinel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


class ScriptedModel:
    """Each response is either a list of tool calls or a final string."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0
        self.seen_messages = []

    def chat(self, messages, tools):
        self.seen_messages = list(messages)
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        meta = {"prompt_tokens": 100 * self.calls, "output_tokens": 20}
        if isinstance(item, str):
            return {"role": "assistant", "content": item}, meta
        return {"role": "assistant", "content": "", "tool_calls": item}, meta


def fresh():
    tmp = Path(tempfile.mkdtemp())
    v = tmp / "vault"
    (v / "wiki").mkdir(parents=True)
    (v / "wiki" / "afterglow.md").write_text(
        "---\ntype: concept\nsummary: Late-time emission.\n"
        "summary_provisional: 0\n---\n\n# afterglow\n\n## Origin\n\n"
        "It follows the burst.\n\n## Decay\n\nIt fades.\n", encoding="utf-8")
    idx = Index(v, tmp / "h.db")
    idx.sync()
    return v, idx, Sentinel(idx)


# --- the schemas Ollama receives -----------------------------------------
ot = ollama_tools()
ok("all seven tools are offered", len(ot) == 7, str(len(ot)))
ok("each carries its docstring",
   all(len(t["function"]["description"]) > 120 for t in ot))
ok("required fields are marked required",
   all(t["function"]["parameters"]["required"] == REQUIRED[t["function"]["name"]]
       for t in ot))
ok("enums survive into the schema",
   "meta" in json.dumps(ot) and "all_tags" in json.dumps(ot))

# --- dispatch -------------------------------------------------------------
v, idx, S = fresh()
h = Harness(S, ScriptedModel([call("read", path="afterglow")], "It is a concept."))
t = h.ask("what is the afterglow page")
ok("a tool call is executed and its result fed back",
   t.steps and t.steps[0].result.startswith("[OK] wiki/afterglow.md"), str(t.steps))
ok("the final answer comes back", t.answer == "It is a concept.")
ok("the tool result is in the message history",
   any(m.get("role") == "tool" for m in h.messages))
ok("prompt tokens are counted", t.prompt_tokens > 0)

# --- malformed calls become statuses, never exceptions -------------------
h = Harness(S, ScriptedModel([call("reed", path="x")], "ok"))
t = h.ask("typo")
ok("an unknown tool is [RETRY] listing the real ones",
   t.steps[0].result.startswith("[RETRY] no tool named") and "read" in t.steps[0].result)

h = Harness(S, ScriptedModel([call("read", pathh="afterglow")], "ok"))
t = h.ask("wrong parameter")
ok("an invented parameter is [RETRY] naming the real ones",
   t.steps[0].result.startswith("[RETRY] read has no parameter pathh"),
   t.steps[0].result)

h = Harness(S, ScriptedModel([call("read")], "ok"))
t = h.ask("missing required")
ok("a missing required field is [RETRY]",
   t.steps[0].result == "[RETRY] read needs path", t.steps[0].result)

h = Harness(S, ScriptedModel([call("read", path="nope")], "ok"))
t = h.ask("no such page")
ok("an impossible request is [STOP], not [RETRY]",
   t.steps[0].result.startswith("[STOP]"), t.steps[0].result)

# --- the identical-call guard --------------------------------------------
h = Harness(S, ScriptedModel(
    [call("read", path="afterglow")],
    [call("read", path="afterglow")],
    "done"))
t = h.ask("read it twice")
ok("the second identical call is refused structurally",
   t.steps[1].result.startswith("[STOP] you already called read"),
   t.steps[1].result)
ok("and it points at the result it already has",
   "the result is above" in t.steps[1].result)

h = Harness(S, ScriptedModel(
    [call("read", path="afterglow"), call("read", path="afterglow", depth="outline")],
    "done"))
t = h.ask("different arguments")
ok("different arguments are not the same call",
   not t.steps[1].result.startswith("[STOP] you already"), t.steps[1].result)

# --- a write resets the guard, because now something HAS changed ---------
v, idx, S = fresh()
h = Harness(S, ScriptedModel(
    [call("listing", by="recent")],
    [call("write", path="wiki/new.md", content="# new\n\nText.", expect="new")],
    [call("listing", by="recent")],
    "done"))
t = h.ask("list, write, list again")
ok("the write succeeded", t.steps[1].result.startswith("[DONE]"), t.steps[1].result)
ok("the same listing is allowed again after the vault changed",
   t.steps[2].result.startswith("[OK]"), t.steps[2].result)
ok("and it sees the new page", "wiki/new.md" in t.steps[2].result)

# --- the step limit -------------------------------------------------------
loop = [[call("read", path="afterglow", from_part=i)] for i in range(1, 12)]
h = Harness(S, ScriptedModel(*loop), max_steps=4)
t = h.ask("go forever")
ok("the loop is bounded", len(t.steps) == 4, str(len(t.steps)))
ok("and says so rather than pretending it finished",
   any("tool budget" in t.answer for _ in [0]) and t.notes, t.answer)

# --- the context is the model's, not a constant --------------------------
class Small:
    num_ctx = 900

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        meta = {"prompt_tokens": 880, "output_tokens": 10}
        if isinstance(item, str):
            return {"role": "assistant", "content": item}, meta
        return {"role": "assistant", "content": "", "tool_calls": item}, meta


v, idx, S = fresh()
h = Harness(S, Small("done"))
ok("the harness takes the model's real context", h.context == 900, str(h.context))
t = h.ask("x")
ok("a near-full context is reported", any("close to full" in n for n in t.notes),
   str(t.notes))

# --- old tool results are dropped, and the drop is announced -------------
h = Harness(S, Small([call("read", path="afterglow", depth="full")],
                     [call("read", path="afterglow", depth="outline")],
                     "done"), max_steps=6)
h.messages += [{"role": "tool", "name": "read", "content": "x" * 4000}
               for _ in range(3)]
t = h.ask("fill it up")
ok("older tool results are dropped rather than the context overflowing",
   any("dropped" in n for n in t.notes), str(t.notes))
ok("and the model is told, so it can fetch them again",
   any("were dropped" in str(m.get("content", "")) for m in h.messages),
   str([m.get("role") for m in h.messages]))
ok("the system prompt survives pruning", h.messages[0]["role"] == "system")
ok("the size estimate includes the tool schemas, which are in every prompt",
   h._overhead() > 1000, str(h._overhead()))
notes = [m for m in h.messages
         if "were dropped" in str(m.get("content", ""))]
ok("one drop note, not one per dropped result", len(notes) <= 1, str(len(notes)))

# --- the loop forces an answer before the budget runs out ----------------
loop = [[call("read", path="afterglow", depth="part", target=f"S{i}")]
        for i in range(20)]
h = Harness(S, ScriptedModel(*loop), max_steps=5)
t = h.ask("read everything")
ok("the model is told to conclude before the steps run out",
   any("used most of your tool budget" in str(m.get("content", ""))
       for m in h.messages), str(len(h.messages)))

# --- an empty answer is never handed back silently -----------------------
h = Harness(S, ScriptedModel(""))
t = h.ask("say nothing")
ok("silence is replaced by a statement of what went wrong",
   t.answer and "did not produce an answer" in t.answer, repr(t.answer))

# --- reset ----------------------------------------------------------------
h = Harness(S, ScriptedModel([call("read", path="afterglow")], "a"))
h.ask("x")
before = len(h.messages)
h.reset()
ok("reset clears the conversation but keeps the system prompt",
   len(h.messages) == 1 and h.messages[0]["role"] == "system" and before > 1)
h2 = Harness(S, ScriptedModel([call("read", path="afterglow")], "a",
                              [call("read", path="afterglow")], "a"))
h2.ask("x")
h2.reset()
t = h2.ask("x again")
ok("and clears the guard with it",
   not t.steps[0].result.startswith("[STOP] you already"), t.steps[0].result)

# --- the fixed overhead, measured on the real schemas --------------------
tools_json = json.dumps(ollama_tools())
from ..harness import SYSTEM  # noqa: E402
tool_tokens = len(tools_json) // 4
sys_tokens = len(SYSTEM) // 4
# The pre-build estimate was 1592 and the earlier system spent 8188. The gap
# to today's figure is the rules that had to be written INTO the descriptions
# rather than the system prompt, each one after a measured failure - where
# pages go, what not to invent, when to record. Rules in the schema were
# measured to be followed where the same rules in the prompt were not, so the
# tokens buy behaviour rather than prose.
#
# The ceiling is what matters: at 12288 this is 18% of context, against 81%
# for the system it replaced.
ok("the seven tools stay well under the budget they replaced",
   tool_tokens < 2600, f"{tool_tokens} tokens")
ok("the system prompt is a fraction of the old one", sys_tokens < 900,
   f"{sys_tokens} tokens")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
if not FAIL:
    print(f"  tool descriptions ~{tool_tokens} tokens")
    print(f"  system prompt     ~{sys_tokens} tokens  (the old one was 5548)")
    print(f"  fixed overhead    ~{tool_tokens + sys_tokens} of 17000, "
          f"{round((tool_tokens + sys_tokens) / 17000 * 100)}%")
    print(f"  working budget    ~{17000 - tool_tokens - sys_tokens} "
          f"against the earlier 3264")
sys.exit(1 if FAIL else 0)
