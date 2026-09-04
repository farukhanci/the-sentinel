"""Field test 5 - where do the 253 seconds go?

    python3 -m sentinel.tests.field_test_timing --model qwen3:4b \\
        --page ~/obsidian/wiki/arxiv-2504.11743.md \\
        --webui http://localhost:3000/api

Runs four experiments, each aimed at one surviving hypothesis. Every number is
reported as load / prefill / generation separately, never as a total - that
collapse produced three wrong diagnoses in a row.

1. INPUT SIZE. The same page truncated to four lengths. If the cost is
   prefill, the curve is monotonic. Last time it was not.
2. SUCCESSIVE CALLS. The identical short call, five times in a row. A load
   cost front-loads; KV pressure GROWS. Last time three calls took 24/63/72s,
   which is the growth shape, and that is the live suspect.
3. KEEP-ALIVE. The same call with the model held resident. Isolates reload.
4. OPEN WEBUI VS DIRECT. The same prompt through the pipeline and straight to
   the backend. The pipeline making extra calls per request is the other live
   suspect, and it is invisible from inside Open WebUI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis import PROMPT
from ..models import OllamaModel, OpenAICompatModel
from ..text import split_frontmatter


def line(label: str, t: dict) -> str:
    if "load" in t:
        return (f"  {label:<22} wall {t['wall']:6.1f}s = load {t['load']:5.1f}"
                f" + prefill {t['prefill']:5.1f} + gen {t['generate']:6.1f}"
                f" + unaccounted {t['unaccounted']:5.1f}"
                f"   [{t['prompt_tokens']:>5} in, {t['output_tokens']:>4} out,"
                f" {t['tok_per_s']:.1f} tok/s]")
    return (f"  {label:<22} wall {t['wall']:6.1f}s"
            f"   [{t['prompt_tokens']:>5} in, {t['output_tokens']:>4} out]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--webui", default=None,
                    help="OpenAI-compatible base URL, e.g. "
                         "http://localhost:3000/api")
    ap.add_argument("--key", default="")
    ap.add_argument("--webui-model", default=None,
                    help="The model id as Open WebUI names it, if it differs "
                         "from the Ollama tag - e.g. a workspace model built "
                         "on top of it. Run this twice: once with the SAME "
                         "base model on both sides, which isolates the "
                         "pipeline, and once with the workspace model, which "
                         "isolates its own system prompt and tools.")
    args = ap.parse_args()

    body = split_frontmatter(Path(args.page).read_text(encoding="utf-8"))[1]
    direct = OllamaModel(args.model, args.host)
    print(f"page: {len(body)} chars, ~{len(body.split())} words")
    print("BEFORE READING ANY NUMBER BELOW, run `ollama ps` in another shell\n"
          "while this is working. If it does not say 100% GPU, the model is\n"
          "partly on the CPU and that alone explains a 35x slowdown - no other\n"
          "hypothesis needs testing until that is ruled out.\n")

    print("1. input size - is it prefill?")
    for frac in (0.05, 0.25, 0.5, 1.0):
        cut = body[:max(int(len(body) * frac), 200)]
        _, t = direct.complete(PROMPT.format(text=cut))
        print(line(f"{int(frac * 100)}% of the page", t))
    print("   monotonic in input size means prefill; flat or jagged means not.\n")

    print("2. five identical calls - does the cost GROW?")
    short = PROMPT.format(text="The afterglow follows the burst.")
    for i in range(5):
        _, t = direct.complete(short)
        print(line(f"call {i + 1}", t))
    print("   a load cost front-loads. Growth across calls is KV pressure.\n")

    print("3. the same call, model already resident")
    _, t = direct.complete(short)
    print(line("warm", t))
    print("   compare load against call 1 above.\n")

    if args.webui:
        print("4. Open WebUI vs direct - same prompt, same model")
        pipe = OpenAICompatModel(args.webui_model or args.model,
                                 args.webui, args.key)
        print(f"   direct: {args.model}   webui: {args.webui_model or args.model}")
        _, t = direct.complete(short)
        print(line("direct to backend", t))
        _, t = pipe.complete(short)
        print(line("through Open WebUI", t))
        print("   a large gap on an identical prompt means the cost is in the\n"
              "   pipeline, not the model - and that is testable no other way.")
    else:
        print("4. skipped - pass --webui to compare against the pipeline")


if __name__ == "__main__":
    main()
