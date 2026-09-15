"""The Sentinel - model clients for the analysis pass.

Field test 2 measured ~253 seconds per page and the CAUSE IS UNKNOWN. Three
hypotheses were tested and all died: not prefill (188 / 1875 / 7144 tokens
took 102 / 196 / 149 s, not even monotonic), not reasoning (the think block
was empty every time), not model loading (three identical small calls took
24 / 63 / 72 s, the FIRST being fastest - a load cost front-loads, it does not
grow). GPU draw and VRAM were normal and the user's own rate on this model is
~35 tok/s, roughly 35x what those runs imply.

Two untested suspects remain: the Open WebUI pipeline making extra calls per
request, and KV pressure across successive calls. Both are testable by going
around Open WebUI, which is what OllamaModel does.

DO NOT use the 253 figure to justify a smaller analysis model. An earlier
draft did; it is withdrawn. With the cause unknown, model size is not
established as the variable.

The lesson for every timing run: report LOAD, PREFILL and GENERATION
SEPARATELY. Collapsing them into one total produced three wrong diagnoses in a
row. Ollama returns all three natively, which is why it is the direct client
here.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

NS = 1e9


class OllamaModel:
    """Direct to the backend. Returns the durations the server itself reports,
    which is the only way to separate load from prefill from generation."""

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 num_ctx: int = 8192, num_gpu: int = 256, timeout: int = 900,
                 think: bool = False, num_predict: int = 1024):
        # num_gpu=256 means "put every layer on the GPU", which overrides
        # Ollama's own fitting decision. Measured on the real card: left to
        # itself at ctx 4096 it reported 24%/76% CPU/GPU, while forced at ctx
        # 8192 it reports 100% GPU at 5541 of 6141 MiB. So the model DOES fit
        # and CPU offload is NOT the cause of the 253 s - a fourth hypothesis
        # dead alongside prefill, reasoning and model loading.
        #
        # 600 MiB of headroom is not much. Raising num_ctx will cross the line.
        # THINKING OFF, and generation bounded.
        #
        # This is where the 253 seconds went. Measured on a real page: load
        # 6.5 s, prefill 0.59 s, unaccounted 0.06 s - and GENERATION 184 s for
        # 5620 output tokens at 30.5 tok/s. The model was running at its
        # normal speed; it was simply producing five thousand tokens of
        # reasoning for a two-field JSON object.
        #
        # That kills the last two hypotheses at once. It is not the Open WebUI
        # pipeline and it is not KV pressure: the cost is in the model's own
        # output, and it is visible only because load, prefill and generation
        # are reported separately.
        #
        # The analysis pass is a TRANSPORT task - copy a sentence, copy the
        # concept names that are already in the text. There is nothing to
        # reason about, so `think` is off by default, and `num_predict` bounds
        # what a runaway can cost even if it is.
        self.model, self.host, self.num_ctx, self.num_gpu, self.timeout = \
            model, host, num_ctx, num_gpu, timeout
        self.think, self.num_predict = think, num_predict

    def complete(self, prompt: str) -> tuple[str, dict]:
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "think": self.think,
            "options": {"temperature": 0, "num_ctx": self.num_ctx,
                        "num_gpu": self.num_gpu,
                        "num_predict": self.num_predict},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Some builds reject `think` on a model that has no thinking mode.
            # Retry without it rather than failing the whole pass.
            if e.code != 400:
                raise
            body = json.loads(body)
            body.pop("think", None)
            req = urllib.request.Request(
                f"{self.host}/api/generate", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
        wall = time.perf_counter() - t0
        prefill = d.get("prompt_eval_duration", 0) / NS
        gen = d.get("eval_duration", 0) / NS
        load = d.get("load_duration", 0) / NS
        return d.get("response", ""), {
            "wall": wall, "load": load, "prefill": prefill, "generate": gen,
            # Everything the server did not account for. If this is most of
            # the wall time, the cost is not in the model at all.
            "unaccounted": wall - load - prefill - gen,
            "prompt_tokens": d.get("prompt_eval_count", 0),
            "output_tokens": d.get("eval_count", 0),
            "tok_per_s": d.get("eval_count", 0) / gen if gen else 0,
        }


    def unload(self) -> bool:
        """Drop the model from memory. Called at the end of a batch pass.

        Observed on the real machine: after a long run of requests the system
        sat at 11 GB of 14 with no swap, and unloading and reloading the model
        returned it to normal. The weights are 5.7 GB on the GPU; what
        accumulates is on the host side, in the runner, across requests.

        A batch pass is exactly the shape that accumulates it - hundreds of
        calls back to back - and it is also the one place where dropping the
        model afterwards costs nothing, because the conversation model has to
        be loaded next anyway.
        """
        try:
            body = json.dumps({"model": self.model, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{self.host}/api/generate", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60):
                return True
        except Exception:
            return False


class ModelHTTPError(RuntimeError):
    """An HTTP failure with the server's own explanation attached.

    A bare `HTTPError: 500` is undiagnosable, and the reason is always in the
    body the caller never sees - a model id the endpoint does not have, a
    context length it will not take, a key that expired. `chat.py` learned
    this against Ollama and says so in `_post`; the cost is the same here, and
    higher, because a remote endpoint is the one place the failure cannot be
    reproduced by hand in a second.

    `status` is kept so a caller can tell a rejected FIELD (400) from a
    rejected REQUEST, which is what the `think` retry below turns on.
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()[:600]
        except Exception:
            detail = "(no body)"
        raise ModelHTTPError(e.code, f"{e.code} from {url}: {detail}") from None
    except urllib.error.URLError as e:
        raise ModelHTTPError(0, f"cannot reach {url}: {e.reason}") from None


class OpenAICompatModel:
    """Through Open WebUI or any OpenAI-shaped endpoint. Wall time only -
    which is precisely the comparison worth making.

    WHAT THIS CANNOT REPORT, and it is the reason the direct client exists:
    the OpenAI shape returns no load / prefill / generation breakdown, only a
    total. Collapsing those three produced three wrong diagnoses in a row (see
    the module docstring), so this returns `wall` and does NOT invent the rest
    - the keys are absent rather than zero, and every reader must branch on
    their presence instead of reading a plausible-looking 0.0.
    """

    def __init__(self, model: str, base_url: str, api_key: str = "",
                 timeout: int = 900, num_ctx: int = 8192,
                 think: bool | None = False):
        # `num_ctx` IS NOT SENT ANYWHERE. It is here because five call sites -
        # analysis.py three times, concepts.py, harness.py - ask the model
        # object how much context it has with `getattr(model, "num_ctx", 8192)`
        # and decide two things with the answer: whether a page is read whole
        # or in windows, and whether a returned prompt_tokens means the page
        # was truncated. Without the attribute every one of them silently
        # assumed 8192 for an endpoint that may have sixteen times that, so a
        # page that fitted easily was chopped into windows and lost its
        # summary for no reason. Set it to the context the endpoint really has.
        #
        # `think` defaults to FALSE, matching OllamaModel, because the
        # analysis pass is a transport task and reasoning there cost 5620
        # tokens for a two-field JSON object. There is no field in the OpenAI
        # shape that turns reasoning off, so this is a best effort: the field
        # is sent the way Ollama's own /v1 shim and several self-hosted
        # servers accept it, and when the endpoint will not have it, or takes
        # it and reasons anyway, that is REPORTED rather than assumed. See
        # `_warn`. Pass None to send nothing and leave the endpoint alone.
        self.model, self.base_url, self.api_key, self.timeout = \
            model, base_url.rstrip("/"), api_key, timeout
        self.num_ctx, self.think = num_ctx, think
        # Sticky, both of them. A pass is hundreds of calls against ONE
        # endpoint: a field it rejected on page 1 will be rejected on page 50,
        # so stop paying the round trip, and say it once rather than fifty
        # times.
        self.think_sent = think is not None
        self.think_note = ""

    def _warn(self, text: str) -> str:
        """Record an endpoint-level warning, and return it ONCE.

        It is a fact about the endpoint, not about the page, so repeating it
        per page would bury the pass output it is meant to stand out in. The
        first result carries it; `self.think_note` keeps it for whoever
        summarises the run.
        """
        if self.think_note:
            return ""
        self.think_note = text
        return text

    def complete(self, prompt: str) -> tuple[str, dict]:
        payload = {
            "model": self.model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.think_sent:
            payload["think"] = self.think
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"

        note = ""
        t0 = time.perf_counter()
        try:
            d = _post(url, payload, headers, self.timeout)
        except ModelHTTPError as e:
            # A strict endpoint rejects an unknown field outright. That is the
            # ONE case where the answer is unambiguous, so take it: drop the
            # field, stop sending it, and say that thinking is now whatever
            # the endpoint decides.
            if e.status != 400 or not self.think_sent:
                raise
            payload.pop("think", None)
            self.think_sent = False
            note = self._warn(
                "the endpoint rejected `think`, so thinking is at the "
                "model's own default and the analysis pass is not turning it "
                "off here")
            d = _post(url, payload, headers, self.timeout)
        wall = time.perf_counter() - t0

        msg = d["choices"][0]["message"]
        text = msg.get("content") or ""
        if self.think is False and not note:
            # THE QUIETER FAILURE, and the one worth catching: a server that
            # does not know the field accepts the request and ignores it. 200
            # OK proves nothing, so the only honest check is the output -
            # reasoning that came back anyway, in either of the two shapes it
            # arrives in.
            if msg.get("reasoning_content") or msg.get("reasoning") \
                    or "<think>" in text:
                note = self._warn(
                    "the endpoint returned reasoning although `think` was "
                    "false - thinking is NOT off here, and the analysis pass "
                    "is paying for it")

        usage = d.get("usage") or {}
        timings = {
            "wall": wall,
            # ABSENT, not zero, when the endpoint does not account for them.
            # A reader that cannot tell 0.0 from "not reported" prints
            # `load 0.0s prefill 0.0s gen 0.0s` under a call that took two
            # minutes, which is worse than printing nothing.
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
        if note:
            timings["note"] = note
        return text, timings
