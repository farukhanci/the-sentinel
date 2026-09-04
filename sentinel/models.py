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


class OpenAICompatModel:
    """Through Open WebUI or any OpenAI-shaped endpoint. Wall time only -
    which is precisely the comparison worth making."""

    def __init__(self, model: str, base_url: str, api_key: str = "",
                 timeout: int = 900):
        self.model, self.base_url, self.api_key, self.timeout = \
            model, base_url.rstrip("/"), api_key, timeout

    def complete(self, prompt: str) -> tuple[str, dict]:
        body = json.dumps({
            "model": self.model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers)
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        wall = time.perf_counter() - t0
        usage = d.get("usage", {})
        return d["choices"][0]["message"]["content"], {
            "wall": wall,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
