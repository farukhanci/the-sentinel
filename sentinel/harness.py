"""The Sentinel - the harness.

The layer that hands the seven primitives to the orchestrator model and runs
the loop. Until this exists there is a library and no system: nothing can call
a tool.

Written as its own loop rather than inside Open WebUI on purpose. Three things
the design requires live here and cannot live in someone else's loop: the
identical-call guard has to be structural rather than advised, the status
markers have to drive control flow, and the token budget has to be countable
against the 17k the whole design is sized around.

Observed failures this exists to make impossible: the model invented a heading
that did not exist, re-called a tool whose result it already held, and argued
with itself about whether the work was finished.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .schema import REQUIRED, TOOLS, schemas
from .tools import Sentinel

MAX_STEPS = 8
CONTEXT_TOKENS = 17000       # the design figure; the real one comes from the model
CHARS_PER_TOKEN = 3.5        # for pruning decisions only, never for a hard limit
PRUNE_AT = 0.70              # of context, before the model is called
FORCE_ANSWER_AT = 2          # steps left when the model is told to conclude

SYSTEM = """You are the control layer of an Obsidian vault. The vault is the
object; you work on it through seven tools.

Finding things:
- `search` when you do not know the page name. It returns one line per hit
  saying what that page is. THAT IS USUALLY ENOUGH - stop there unless the
  summaries showed a page is relevant but did not contain the answer.
- `listing` to enumerate by an exact attribute: a tag, a folder, recency.
- `graph` for link structure, or with no subject for the shape of the vault.
- `read` once you know which page. Start at `outline`, then take one section.
  Never read a whole page unless you have a reason.

Writing:
- `write` needs the `expect` value from your last read of that page, or "new".
- Pages go in `wiki/`. `graph` lists concepts with NO page; `read` refuses
  them, so do not try.
- Links are not your job. A dedicated pass places them later; a page without
  them is not broken.
- `relocate` renames and redirects links. `remove` deletes.

Keeping what is settled:
- WRITE WHEN THE USER ASKS YOU TO. "kaydet", "save this", "write that down" -
  that is the trigger. Do not decide on your own that something is worth
  keeping, and do not offer to.
- When they ask, call `write` immediately. One call. No search first, no
  question about which page, no description of what you are about to write.
- Write what the conversation settled, not what you know about the subject.
  A conclusion you drew from what the vault told you belongs; the rest of
  your knowledge of the topic does not.
- Write what the conversation established, in the user's own terms, and
  nothing you are filling in from your own knowledge. This is the USER'S
  thinking, not a source, and the vault marks it as such.
- If a page already holds the subject, write to that one - it grows rather
  than being replaced.
- The conversation itself is recorded alongside, by the system. You do not
  have to do it and you cannot edit it.

When the vault has nothing:
- Read the summaries that came back. If they are about other subjects, the
  vault does not contain what you asked for. SAY SO. Rewording the query
  returns the same pages, and the tools will tell you when you are repeating
  yourself.
- Do not fill the gap from your own knowledge and present it as the vault's;
  if an answer is yours rather than the vault's, mark it as yours.

When a step is finished:
- SEARCHING IS FINISHED as soon as the results come back about other
  subjects. That is the answer: the vault does not have it. Searching again
  with different words cannot change it.
- FINDING NOTHING IS FINISHED when you call `write`. A statement the user made
  and no page to hold it is not a dead end, it is the next call.
- WRITING IS FINISHED when `write` returns [DONE]. Say what you wrote in one
  line and stop.
- A TURN IS FINISHED when you have either answered from what the tools
  returned or written what the user told you. Do not offer to do more.

Every result begins with a status:
  [OK] worked · [DONE] finished · [more] there is a next call, it is named
  [RETRY] the call was malformed - fix it and try again
  [STOP] the request cannot succeed - do NOT retry it, say so instead
  [note] something abnormal, worth mentioning to the user

Answer the user in the language they used."""


def ollama_tools() -> list[dict]:
    """The same schemas, in the shape Ollama's /api/chat wants."""
    out = []
    for s in schemas(Sentinel):
        out.append({"type": "function", "function": {
            "name": s["name"],
            "description": s["description"],
            "parameters": {
                "type": "object",
                "properties": s["input_schema"]["properties"],
                "required": s["input_schema"]["required"],
            }}})
    return out


@dataclass
class Step:
    tool: str
    args: dict
    result: str


@dataclass
class Turn:
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    notes: list[str] = field(default_factory=list)


class Harness:
    def __init__(self, sentinel: Sentinel, model, max_steps: int = MAX_STEPS,
                 context_tokens: int | None = None):
        self.s = sentinel
        self.model = model
        self.max_steps = max_steps
        # The REAL context, not the design figure. Running an 8192 model
        # against a hardcoded 17000 meant the budget check never fired: the
        # conversation reached 8138 of 8192, one answer was cut off
        # mid-sentence and the next came back empty, with nothing said about
        # it. Ollama truncates silently - the same failure the analysis pass
        # has a gate for, and this had none.
        self.context = context_tokens or getattr(model, "num_ctx", None) \
            or CONTEXT_TOKENS
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM}]
        self._generation = 0      # bumped by anything that changes the vault
        self._seen: set = set()
        self._fixed: int | None = None

    # -- dispatch ---------------------------------------------------------

    def _call(self, name: str, args: dict) -> str:
        """Every failure here is a STATUS, never an exception. A traceback
        reaching the model is a result it cannot act on."""
        if name not in TOOLS:
            return (f"[RETRY] no tool named '{name}'. The tools are: "
                    + ", ".join(TOOLS))
        allowed = set(TOOLS[name])
        unknown = set(args) - allowed
        if unknown:
            return (f"[RETRY] {name} has no parameter "
                    f"{', '.join(sorted(unknown))}. It takes: "
                    + ", ".join(sorted(allowed)))
        missing = [r for r in REQUIRED[name] if not args.get(r)]
        if missing:
            return f"[RETRY] {name} needs {', '.join(missing)}"

        key = (self._generation, name, json.dumps(args, sort_keys=True))
        if key in self._seen:
            # Structural, not advised. The guard is keyed on a vault
            # GENERATION rather than a fixed whitelist of idempotent tools: a
            # repeat is only meaningless while nothing has changed, and the
            # moment a write lands the same call becomes a legitimate one.
            return (f"[STOP] you already called {name} with these arguments "
                    f"and the result is above. Use it, or do something else.")
        self._seen.add(key)

        try:
            out = getattr(self.s, name)(**args)
        except TypeError as e:
            return f"[RETRY] {name}: {e}"
        except Exception as e:
            return f"[STOP] {name} failed: {type(e).__name__}: {e}"

        if name in ("write", "relocate", "remove") and out.startswith("[DONE]"):
            self._generation += 1
        return out

    # -- keeping the conversation inside the context ----------------------

    def _overhead(self) -> int:
        """The tool schemas and system prompt, which are in EVERY prompt.

        Measured: ~1743 tokens of tool descriptions plus ~350 of system
        prompt. Pruning to 70% of the context while ignoring 2093 tokens of
        fixed cost meant pruning against the wrong number - the conversation
        kept arriving at 8180 of 8192 no matter how much was dropped.
        """
        if self._fixed is None:
            self._fixed = int(len(json.dumps(ollama_tools())) / CHARS_PER_TOKEN)
        return self._fixed

    def _size(self) -> int:
        return self._overhead() + int(
            sum(len(str(m.get("content", ""))) for m in self.messages)
            / CHARS_PER_TOKEN)

    def _prune(self) -> int:
        """Drop the OLDEST tool results first. Returns how many went.

        Tool results are the bulk and the most recoverable: the model can call
        the tool again, and the guard now allows it because the generation
        moves on. What is never dropped is the system prompt or anything the
        user or the model actually said - losing those loses the thread, while
        losing an old search result loses a lookup.
        """
        limit = int(self.context * PRUNE_AT)
        dropped = 0
        while self._size() > limit:
            idx = next((i for i, m in enumerate(self.messages)
                        if m.get("role") == "tool"), None)
            if idx is None:
                break
            self.messages.pop(idx)
            dropped += 1
        # One note per turn, not one per drop. Four identical lines in a row
        # is noise where a number was wanted.
        self.messages = [m for m in self.messages
                         if not str(m.get("content", "")).startswith(
                             "[note] ") or "were dropped" not in
                         str(m.get("content", ""))]
        if dropped:
            # A dropped result is a fact about the conversation, not a silent
            # edit. Say it where the model can act on it.
            self._seen.clear()
            self.messages.insert(1, {
                "role": "user",
                "content": f"[note] {dropped} older tool results were dropped "
                           f"to stay inside the context. Call again if you "
                           f"need them."})
        return dropped

    # -- the loop ---------------------------------------------------------

    def ask(self, user: str) -> Turn:
        self.messages.append({"role": "user", "content": user})
        turn = Turn()

        for step in range(self.max_steps):
            gone = self._prune()
            if gone:
                turn.notes.append(f"dropped {gone} older tool results to stay "
                                  f"inside {self.context} tokens")

            if step == self.max_steps - FORCE_ANSWER_AT and not turn.answer:
                # STRUCTURE, not wording. The system prompt already says to
                # stop at the summaries; measured, the model read eight
                # sections one after another and never answered. A rule the
                # loop enforces is a different thing from a rule the prompt
                # requests.
                self.messages.append({
                    "role": "user",
                    "content": "[note] You have used most of your tool budget "
                               "for this turn. Answer now from what the tools "
                               "have already returned, and say plainly what "
                               "you could not determine."})

            reply, meta = self.model.chat(self.messages, ollama_tools())
            turn.prompt_tokens = max(turn.prompt_tokens,
                                     meta.get("prompt_tokens", 0))
            turn.output_tokens += meta.get("output_tokens", 0)

            calls = reply.get("tool_calls") or []
            self.messages.append({k: v for k, v in reply.items() if v})

            if not calls:
                turn.answer = reply.get("content", "").strip()
                break

            for c in calls:
                fn = c.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                result = self._call(name, args)
                turn.steps.append(Step(name, args, result))
                self.messages.append(
                    {"role": "tool", "name": name, "content": result})
        else:
            # Not an error the model should keep working around - say it.
            turn.notes.append(
                f"stopped after {self.max_steps} tool calls without an answer")
            turn.answer = turn.answer or (
                "I used my tool budget for this turn without reaching an "
                "answer. Ask me again more narrowly.")

        used = turn.prompt_tokens
        if used > self.context * 0.9:
            turn.notes.append(
                f"context {used} of {self.context} - close to full")
        if not turn.answer:
            # An empty answer is a failure with a cause, and the cause is
            # knowable. Never hand back silence.
            turn.answer = (
                "I did not produce an answer this turn"
                + (f" - the context was nearly full ({used} of {self.context})."
                   if used > self.context * 0.9 else ".")
                + " Try /reset, or ask again more narrowly.")
        return turn

    def reset(self) -> None:
        self.messages = self.messages[:1]
        self._seen.clear()
