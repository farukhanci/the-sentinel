"""Run: python3 -m sentinel.tests.test_models

`OpenAICompatModel` had ZERO coverage before this file: it was written at the
first commit, never touched since, and reachable only through
`field_test_timing.py --webui`, which needs a live endpoint and is not part of
this suite. Connecting it to the maintenance pass without a fixture that runs
in a second would have been the third way of finding out it does not work.

The HTTP layer is faked, so what is under test is the CLIENT - which fields it
sends, what it does with the three answers it can get back, and what it
reports when the endpoint cannot do what was asked of it.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.request

from ..models import ModelHTTPError, OpenAICompatModel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


URL = "http://endpoint/v1"


class _Body:
    def __init__(self, data: bytes):
        self.data = data

    def read(self) -> bytes:
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Fake:
    """Stands in for urllib.request.urlopen and keeps what was sent."""

    def __init__(self, *queued):
        self.queue = list(queued)
        self.sent: list[dict] = []

    def __call__(self, req, timeout=None):
        self.sent.append({
            "url": req.full_url,
            "body": json.loads(req.data.decode()),
            # urllib title-cases header names, so compare lowered.
            "headers": {k.lower(): v for k, v in req.headers.items()},
        })
        nxt = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(nxt, Exception):
            raise nxt
        return _Body(json.dumps(nxt).encode())


def reply(content: str = '{"summary": "s", "concepts": []}',
          usage: dict | None = {"prompt_tokens": 120, "completion_tokens": 30},
          **message) -> dict:
    msg = {"role": "assistant", "content": content}
    msg.update(message)
    out: dict = {"choices": [{"message": msg}]}
    if usage is not None:
        out["usage"] = usage
    return out


def http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(f"{URL}/chat/completions", code, "err", {},
                                  io.BytesIO(body.encode()))


def run(model: OpenAICompatModel, fake: Fake, prompt: str = "p"):
    """One call with urlopen replaced, restored whatever happens."""
    real = urllib.request.urlopen
    urllib.request.urlopen = fake
    try:
        return model.complete(prompt)
    finally:
        urllib.request.urlopen = real


# --- the shape of the answer ----------------------------------------------
f = Fake(reply(content="the text"))
text, t = run(OpenAICompatModel("m", URL), f)
ok("complete returns the message content", text == "the text", repr(text))
ok("and a timings dict beside it", isinstance(t, dict), repr(t))
ok("wall time is measured here, since the endpoint does not report it",
   "wall" in t and t["wall"] >= 0, repr(t))
ok("token counts come from usage",
   (t["prompt_tokens"], t["output_tokens"]) == (120, 30), repr(t))

# THE CONTRACT THE PROGRESS LINE DEPENDS ON. `analysis.py` branches on whether
# `load` is present, because a 0.0 it cannot distinguish from "not reported"
# printed `load 0.0s prefill 0.0s gen 0.0s` under a two-minute call.
for absent in ("load", "prefill", "generate", "unaccounted", "tok_per_s"):
    ok(f"{absent!r} is ABSENT, not zero - the endpoint does not account for it",
       absent not in t, repr(t))

# --- num_ctx, which is why five call sites were guessing 8192 -------------
ok("num_ctx defaults to the same 8192 the call sites assumed",
   OpenAICompatModel("m", URL).num_ctx == 8192)
big = OpenAICompatModel("m", URL, num_ctx=128000)
ok("a real context window is carried on the object",
   getattr(big, "num_ctx", 8192) == 128000)
f = Fake(reply())
run(big, f)
ok("the request body carries no num_ctx or options block",
   "num_ctx" not in f.sent[0]["body"] and "options" not in f.sent[0]["body"],
   str(f.sent[0]["body"]))

# --- the key, and where it is not ----------------------------------------
f = Fake(reply())
run(OpenAICompatModel("m", URL, api_key="secret"), f)
ok("an api key becomes a bearer header",
   f.sent[0]["headers"].get("authorization") == "Bearer secret",
   str(f.sent[0]["headers"]))
f = Fake(reply())
run(OpenAICompatModel("m", URL), f)
ok("and with no key there is no authorization header at all",
   "authorization" not in f.sent[0]["headers"], str(f.sent[0]["headers"]))
ok("the url is the endpoint's chat completions path",
   f.sent[0]["url"] == f"{URL}/chat/completions", f.sent[0]["url"])

# --- think: the measured decision, carried as far as it can be -----------
f = Fake(reply())
run(OpenAICompatModel("m", URL), f)
ok("think is sent FALSE by default, as the analysis pass requires",
   f.sent[0]["body"].get("think") is False, str(f.sent[0]["body"]))
f = Fake(reply())
run(OpenAICompatModel("m", URL, think=None), f)
ok("think=None sends nothing and leaves the endpoint alone",
   "think" not in f.sent[0]["body"], str(f.sent[0]["body"]))

# A STRICT ENDPOINT REJECTS THE FIELD. That is the unambiguous case.
m = OpenAICompatModel("m", URL)
f = Fake(http_error(400, '{"error": "unknown field: think"}'), reply("ok"))
text, t = run(m, f)
ok("a 400 on `think` is retried without it rather than failing the page",
   text == "ok" and len(f.sent) == 2, str(f.sent))
ok("the retry drops the field", "think" not in f.sent[1]["body"],
   str(f.sent[1]["body"]))
ok("and the fallback is REPORTED, not swallowed",
   "rejected `think`" in t.get("note", ""), repr(t.get("note")))
f2 = Fake(reply("second"))
text2, t2 = run(m, f2)
ok("after a rejection the field is not sent again - one round trip, not one "
   "per page", "think" not in f2.sent[0]["body"], str(f2.sent[0]["body"]))
ok("and the warning is not repeated on every page after it",
   "note" not in t2, repr(t2))
ok("but it stays on the client for whoever summarises the run",
   "rejected `think`" in m.think_note, m.think_note)

# THE QUIETER FAILURE: 200 OK, field ignored, reasoning anyway.
m = OpenAICompatModel("m", URL)
_, t = run(m, Fake(reply(reasoning_content="let me think about this")))
ok("reasoning returned despite think=false is caught",
   "NOT off" in t.get("note", ""), repr(t.get("note")))
m = OpenAICompatModel("m", URL)
_, t = run(m, Fake(reply(reasoning="thinking")))
ok("the other field name for it is caught too",
   "NOT off" in t.get("note", ""), repr(t.get("note")))
m = OpenAICompatModel("m", URL)
_, t = run(m, Fake(reply(content="<think>hmm</think>{}")))
ok("and thinking that arrives inline in the content",
   "NOT off" in t.get("note", ""), repr(t.get("note")))
m = OpenAICompatModel("m", URL)
_, t = run(m, Fake(reply()))
ok("a clean answer raises nothing", "note" not in t and not m.think_note,
   repr(t))
m = OpenAICompatModel("m", URL, think=None)
_, t = run(m, Fake(reply(reasoning_content="deliberate")))
ok("thinking is not flagged when it was never asked to be off",
   "note" not in t, repr(t))

# --- failures a person can act on ----------------------------------------
try:
    run(OpenAICompatModel("m", URL), Fake(http_error(500, "model not loaded")))
    caught = None
except ModelHTTPError as e:
    caught = e
ok("an HTTP failure carries the server's own words", caught is not None
   and "model not loaded" in str(caught), str(caught))
ok("and the status, so a caller can tell a rejected field from a dead server",
   getattr(caught, "status", None) == 500, str(getattr(caught, "status", None)))

try:
    run(OpenAICompatModel("m", URL),
        Fake(urllib.error.URLError("connection refused")))
    unreachable = None
except ModelHTTPError as e:
    unreachable = e
ok("an unreachable endpoint says so instead of raising a socket error",
   unreachable is not None and "cannot reach" in str(unreachable),
   str(unreachable))

# --- usage: the boundary worth knowing about ------------------------------
#
# `analysis.py` detects a truncated prompt by comparing the endpoint's
# prompt_tokens against the context. Some OpenAI-compatible proxies return no
# usage block at all, and then that number is 0 - which the gate reads as "no
# measurement" and lets through. The client cannot invent it; what it can do
# is not pretend, and what this test does is pin the limit so it is known
# rather than discovered on a page that was silently half-read.
_, t = run(OpenAICompatModel("m", URL), Fake(reply(usage=None)))
ok("a missing usage block yields zero tokens, not a crash",
   t["prompt_tokens"] == 0 and t["output_tokens"] == 0, repr(t))
_, t = run(OpenAICompatModel("m", URL), Fake(reply(usage={})))
ok("an empty usage block behaves the same way", t["prompt_tokens"] == 0,
   repr(t))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f_ in FAIL:
    print("  FAIL  " + f_)
sys.exit(1 if FAIL else 0)
