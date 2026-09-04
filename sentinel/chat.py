"""The Sentinel - talk to the vault.

    python3 -m sentinel.chat --vault ~/obsidian/Obsidian-1 --model qwen3.5-9b-jinja:latest

Goes straight to Ollama, not through Open WebUI. The loop, the guard and the
budget accounting have to be visible; a pipeline that hides them is exactly
what made 253 seconds impossible to diagnose three times over.

Commands: /reset clears the conversation, /budget prints the fixed overhead,
/steps shows the tool calls from the last turn, /quit exits.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .embed import E5Encoder
from .harness import SYSTEM, Harness, ollama_tools
from .index import Index, default_db
from .tools import Sentinel


def _post(url: str, payload: dict, timeout: int) -> dict:
    """POST and, on failure, SHOW WHAT THE SERVER SAID.

    A bare `HTTPError: 500` is undiagnosable. Ollama puts the reason in the
    response body - unsupported tool calling, a context size the model cannot
    take, a missing model - and swallowing it turns a one-line fix into an
    afternoon.
    """
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode()[:600]
        except Exception:
            body = "(no body)"
        raise RuntimeError(f"ollama {e.code} from {url}: {body}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"cannot reach ollama at {url}: {e.reason}. Is it running?"
        ) from None


class OllamaChat:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 num_ctx: int = 17000, num_gpu: int = 256, timeout: int = 900,
                 think: bool | None = None):
        # `think` is None by default, meaning: leave it to the model's own
        # setting. NOT forced off as it is in the analysis pass.
        #
        # The two jobs are different. Analysis is transport - copy a sentence,
        # copy names already in the text - and reasoning there bought nothing
        # while costing 5620 tokens. Choosing a tool, deciding which rung of
        # the ladder to start on and deciding when the answer is enough are
        # judgements, and a wrong choice degrades everything downstream.
        #
        # So it stays a flag rather than a decision, and `--think false` makes
        # the comparison measurable on the same question.
        self.model, self.host = model, host
        self.num_ctx, self.num_gpu, self.timeout = num_ctx, num_gpu, timeout
        self.think = think

    def chat(self, messages: list[dict], tools: list[dict]):
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": 0.3, "num_ctx": self.num_ctx,
                        "num_gpu": self.num_gpu},
        }
        if tools:
            payload["tools"] = tools
        if self.think is not None:
            payload["think"] = self.think
        t0 = time.perf_counter()
        d = _post(f"{self.host}/api/chat", payload, self.timeout)
        return d.get("message", {}), {
            "wall": time.perf_counter() - t0,
            "prompt_tokens": d.get("prompt_eval_count", 0),
            "output_tokens": d.get("eval_count", 0),
            "load": d.get("load_duration", 0) / 1e9,
            "prefill": d.get("prompt_eval_duration", 0) / 1e9,
            "generate": d.get("eval_duration", 0) / 1e9,
        }


def show_steps(turn) -> None:
    for i, s in enumerate(turn.steps if turn else [], 1):
        args = ", ".join(f"{k}={v!r}" for k, v in s.args.items())
        head = s.result.splitlines()[0]
        print(f"  {i}. {s.tool}({args})")
        print(f"     -> {head}   [{len(s.result)} chars]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="A folder to leave out of the index, relative to the "
                         "vault root. Repeat for several. Dot-folders are "
                         "always skipped.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--embed-model", default=None,
                    help="Path to multilingual-e5-small. Without it search "
                         "runs on the literal half alone, which still works.")
    ap.add_argument("--db", default=None)
    ap.add_argument("--num-ctx", type=int, default=17000,
                    help="The design is sized around 17k, but the KV cache "
                         "for that much context is what pushes a 9B model off "
                         "a 6 GB card. Measured: 8192 fits at 100%% GPU with "
                         "5541 of 6141 MiB used. Check `ollama ps` after the "
                         "first message - anything other than 100%% GPU means "
                         "this number is too high for the model.")
    ap.add_argument("--num-gpu", type=int, default=256)
    ap.add_argument("--think", choices=["on", "off"], default=None,
                    help="Leave unset to use the model's own default. Set it "
                         "to compare the same question both ways: what to "
                         "watch is the tool sequence and whether it stops at "
                         "the right rung, not only the wall clock.")
    args = ap.parse_args()

    vault = Path(args.vault).expanduser()
    idx = Index(vault, args.db or default_db(vault),
                exclude=args.exclude)
    if idx.hash_version_changed is not None:
        # Visible AND handled. The alternative, measured, was seventeen pages
        # quietly back in the queue and a hand-written SQL statement to undo
        # it. Announcing without acting would only move the chore.
        n = idx.carry_analysis_forward()
        print(f"[note] content-hash definition changed "
              f"(v{idx.hash_version_changed} -> current); {n} analysed pages "
              f"restored")
    stats = idx.sync()
    print(f"vault: {stats['seen']} pages")
    if stats["unreadable"]:
        print(f"[note] {len(stats['unreadable'])} unreadable files")

    encoder = None
    if args.embed_model:
        from .embed import embed_pending
        encoder = E5Encoder(args.embed_model)
        n = embed_pending(idx, encoder)
        if n:
            print(f"embedded {n} new chunks")

    pending = len(idx.pending_analysis())
    if pending:
        print(f"[note] {pending} pages awaiting analysis - summaries are "
              f"provisional and no links have been placed")

    h = Harness(Sentinel(idx, encoder),
                OllamaChat(args.model, args.host, num_ctx=args.num_ctx,
                           num_gpu=args.num_gpu,
                           think=None if args.think is None
                           else args.think == "on"),
                context_tokens=args.num_ctx)
    print(f"context: {args.num_ctx}")
    print("\n/reset  /budget  /steps  /quit\n")
    last = None

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        # A pasted line often carries the prompt with it. Only strip it when
        # what follows is a command, so a message that genuinely starts with
        # '>' still works.
        if line.startswith(">") and line.lstrip("> ").startswith("/"):
            line = line.lstrip("> ")
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            h.reset()
            print("conversation cleared")
            continue
        if line == "/budget":
            tools_json = json.dumps(ollama_tools())
            print(f"  tool descriptions ~{len(tools_json) // 4} tokens")
            print(f"  system prompt     ~{len(SYSTEM) // 4} tokens")
            print(f"  fixed overhead    ~{(len(tools_json) + len(SYSTEM)) // 4}"
                  f" of 17000")
            continue
        if line == "/steps":
            if not last or not last.steps:
                print("  no tool calls last turn")
            show_steps(last)
            continue

        t0 = time.perf_counter()
        try:
            last = h.ask(line)
        except RuntimeError as e:
            # The model is a dependency like any other. A failure to reach it
            # ends the turn, not the session.
            print(f"\n[STOP] {e}\n")
            continue
        print(f"\n{last.answer}\n")
        print(f"[{len(last.steps)} tool calls, {last.prompt_tokens} prompt "
              f"tokens of {args.num_ctx}, {time.perf_counter() - t0:.1f}s]")
        for n in last.notes:
            print(f"[note] {n}")
        if last.notes and last.steps:
            # When the loop gives up, what it DID is the diagnosis. Requiring
            # a second command to see it loses the turn that failed.
            print("  what it did:")
            show_steps(last)


if __name__ == "__main__":
    main()
