# The Sentinel

An Obsidian vault a model can read and write through seven primitives.

The Sentinel puts a local model in front of a markdown vault: it searches by
meaning and by wording at once, follows the link structure, opens pages to a
bounded depth, and writes back. A nightly pass summarises new pages, extracts
the concepts they refer to, and embeds them, so the search substrate stays
current without anyone typing a command.

Everything runs locally — a 9B orchestrator on a 6 GB card, embeddings on the
CPU. No API keys.

## Why it is built this way

Two questions shaped everything. What does a small model on a small card
actually need in order to be right rather than to look right — and how do you
stop a system that writes into its own source of truth from slowly filling it
with its own inventions.

### Fidelity is enforced in code, not asked for in the prompt

A prompt that says "only use what the sources say" is a request. These are
gates: a model that ignores them produces nothing rather than something wrong.

**A concept that does not appear verbatim in the text is dropped before the
linker ever sees it.** Not scored down — dropped.

**A derived page cannot discover new concepts.** The system's own output is
never a source for the next generation. This is the loop the model-collapse
literature describes, and closing it structurally is cheaper than detecting it
later.

**A concept the sources never define gets no page.** Mentioned a hundred times
is still not defined once. On the real vault, 121 unresolved names had enough
surrounding text but only about a third carried a sentence saying what the
thing *is*, and reading those by hand cut it further. `wireless power
transfer` sits in eleven thousand characters and is never once defined.

**The graph is fed by what the user saved, not by everything discussed.** A
conversation is not evidence.

The restraint is the design. The closest thing to this run at scale is Cebuano
Wikipedia, where one bot wrote 99% of the articles. The facts were correct —
they came from databases — and the community still proposed closing the
project, because the volume of low-information stubs was itself the problem.

### What a small model needs

**Seven tools, not thirty-two.** Tool count is an accuracy problem before it
is a token problem. Accuracy degrades past 15–20 tools in rotation, and on the
Berkeley Function Calling Leaderboard it fell from 43% to 2% as the tool count
went from 4 to 51. The failure is not "I don't know which tool" — the model
picks a plausible wrong one, or fills arguments borrowed from a different
tool's schema. So there are seven: `read`, `search`, `listing`, `graph`,
`write`, `relocate`, `remove`. The schema cost fell with the count: 8188
tokens for the old 32-tool system, 2214 for these seven.

Where a parameter exists it is depth of one intent, which is the only kind of
mode parameter that survives contact with a small model. A combined
`find(query, kind)` was tried and split back apart: semantic retrieval and
deterministic enumeration are two intents, and they are `search` and
`listing`.

**A ladder, cheapest rung first.** An exact page lookup is free, so it goes
first. Then summary-level search across both systems, fused — the normal
stopping place. Body-level verbatim chunks only when the summaries showed a
page was relevant but did not hold the answer. Opening the file is the last
rung, and even then through bounded depths rather than a dump.

**Never show a similarity score.** Every score measured on this model fell
between 0.740 and 0.861, wrong answers included, with winning margins of 0.015
to 0.064. Printing `(similarity 0.82)` reads as confidence when the wrong
answer scored 0.79. Only rank carries information — which is also why the two
retrieval systems are fused with RRF, which ranks rather than scores.

**One output policy for all seven primitives.** An earlier version had five
read tools and four different policies: one refused over 8000 characters, one
truncated silently at 3000, one capped at 10 hits without saying so, one
stated its cap, one dumped without limit. The rule now is never refuse and
never truncate silently. Silent truncation is the recurring defect class,
because a cut answer looks exactly like a complete one.

### Things that turned out to be cheap

**Unresolved links are stored rather than discarded.** The unresolved list
*is* the growth queue — the same rows that say "this link points nowhere" say
"these are the concepts the vault keeps reaching for". And when the page is
finally written, every link written earlier comes alive in one SQL update: 228
links, 0.55 ms, zero files rewritten.

**One hash, three uses.** Hashing the content rather than the markup means
inserting a link does not move the hash. So the write guard does not go stale,
the page does not requeue its own pass, and a real edit is still caught. Three
behaviours, one field.

**The index is derived, never authoritative.** Three of the four read
primitives were walking the vault on every call; everything they need now sits
in one SQLite file alongside the embeddings. A full rebuild by scanning the
vault must always be possible, and where the two disagree the vault wins.
Maintaining it is nearly free: the embedding sync already walks the vault
comparing mtimes, so pulling out each page's wikilinks and summary in the same
pass costs a few lines rather than a second sync.

### Where the model is checked against itself

**Name resolution errs in one direction.** Three mechanisms decide; everything
else defers. The property that matters is zero wrong merges, because a wrong
merge destroys information irreversibly while a missed merge costs one
unresolved link and is repaired by a later alias entry. What code must not
decide: `gamma-ray burst` / `short gamma-ray burst` is a subclass and `Ariel` /
`Ariel Space Telescope` is an expansion, and the two have an identical shape.
Both defer.

**Merges are verified by reversal.** The model made two wrong merges in twenty
answers, and both contradicted its own other answers — so the pair is swapped
and the question asked again. On the real vault this catches several more
wrong merges on every pass.

**Summary and extraction are two different models, because they are two
different shapes of problem.** A summary is holistic and cannot be windowed; a
4B model at 120k context reads the whole document in one call. Extraction is
local and windows fine; the 9B walks the concepts window by window.

## Two ways in

**Open WebUI** is the everyday one. `openwebui_tool.py` loads into Open WebUI
and hands the seven primitives to whatever model you are talking to. It is
also the only path that can see `__messages__` and `__metadata__`, which is
what the raw capture mechanism is built on: the code copies the whole
conversation and finds the file again by `chat_id`.

**`sentinel.chat`** goes straight to Ollama and runs the harness: the tool
loop, the status markers that drive control flow, and token accounting. Use
it when something needs diagnosing — `/steps`, `/budget`, `/reset` show you
what Open WebUI hides.

An HTTP tool server used to be a third path. It was removed: it could not see
`__messages__`, so capture could not work through it, and its reason for
existing did not survive contact with the mount setup below.

## What it does not do

**It does not verify claims.** Search finds pages; it does not judge them.
Summaries are extractive but unchecked. What the vault says is what you get.
The gates above stop the system inventing *new* material — they say nothing
about whether what you saved was right.

**Maintenance does not run during a conversation.** The analysis model and the
conversation model cannot share a 6 GB card, so a pass evicts whatever is
loaded. It runs overnight, on a timer, and surfaces itself through the health
line rather than interrupting. It is not slow — about 12 seconds a page; seven
pages took 139 seconds and two took 30 on the last passes. It was slow once:
with the thinking block on, the model produced 5620 tokens for a two-field
JSON object, and turning it off took generation from 184 seconds to 2.5.

**`write`, `relocate` and `remove` are real, and there is no
authentication.** See the security note below.

## Requirements

- Python 3.11 or later
- [Ollama](https://ollama.com) with a tool-calling model
- An Obsidian vault (any directory of markdown files with frontmatter)
- An ONNX embedding model on disk — multilingual-e5-small, 384-dim
- A GPU the orchestrator fits on; embeddings run on the CPU

Embeddings run on the CPU on purpose: search must never compete with the
conversation model for VRAM. multilingual-e5-small was chosen over
bge-small-en-v1.5 because that one is English-only and would have failed
silently on a bilingual vault. Same 384 dimensions, so the swap cost nothing
structurally — but changing it again invalidates every stored vector and
forces a full re-embed.

Context size is a per-model measurement, not a constant. `CONTEXT_TOKENS` in
`harness.py` is 17000, the figure the design was sized around; in practice
17000 gave CUDA OOM, and the two models in use run at 12288 and 35000.
Measure yours.

## Models

Three roles, three models, because they are three different jobs. These are
what the numbers in this file were measured on — substitute your own, but the
context sizes are per-model measurements rather than settings you can copy.

| Role | Model | Context | Notes |
| --- | --- | --- | --- |
| Conversation | `hf.co/AtomicChat/Ornith-1.5-9B-GGUF:IQ4_XS` | 35000 | temperature 0.6, top_p 0.95, top_k 20, min_p 0, max_tokens 3000, num_gpu 256 |
| Concept extraction | `qwen3.5-9b-jinja` | 8192 | local work, windowed |
| Summaries | `qwen3.5-4b-xl` | 120000 | whole document in one call |
| Embeddings | multilingual-e5-small (ONNX) | — | CPU, 384-dim |

The analysis pass runs with thinking off. With it on, the model produced 5620
tokens for a two-field JSON object; turning it off took generation from 184
seconds to 2.5.

## Setup

```bash
git clone https://github.com/farukhanci/the-sentinel
cd the-sentinel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull a model, put the embedding model somewhere on disk, and talk to the vault:

```bash
python3 -m sentinel.chat \
    --vault ~/obsidian/YourVault \
    --model your-model:latest
```

## Open WebUI

Open WebUI runs in Docker here, so it needs two bind mounts: this repository,
and the vault. Nothing is installed inside the container — the tool file puts
the repository on `sys.path` and imports from there.

```bash
docker run -d --name open-webui \
  -v ~/the-sentinel:/sentinel \
  -v ~/obsidian/YourVault:/vault \
  ... your usual flags ...
```

Then paste `openwebui_tool.py` into Open WebUI under Workspace → Tools, and
enable it for the model you talk to. Everything else is configured through
the valves on that tool: the vault path as the container sees it, folders to
exclude, and the path to the embedding model. Leave the index path empty
unless you have a reason — empty means the tool derives it the same way every
other entry point does, and a second copy of that path is how the index once
split in two.

Then give the model the system prompt in `openwebui_system_prompt.txt`. It is
not optional decoration: it is where the ladder is explained, where writing is
tied to the user asking for it rather than the model deciding, and where the
status markers are defined. Without it the model has seven tools and no idea
when to stop.

Reloading matters. Open WebUI caches tool modules, so an edit on the host is
silently ignored and the old code keeps answering. The tool file drops
`sentinel.*` out of the module cache as it loads, which makes pasting it again
a real reload.

**`write`, `relocate` and `remove` are real, and nothing authenticates
them.** Anyone who can talk to the model can delete a note.

## Nightly maintenance

One ordered run, on a timer. The order is load-bearing: sync the index first
so everything after it sees the vault as it is, then analysis to produce
summaries and concepts, then embeddings over the chunks analysis just made.

```bash
python3 -m sentinel.timer --install \
    --vault ~/obsidian/YourVault --model your-model:latest \
    --at 23:00 --minutes 30
```

Install it from inside the venv — the timer writes `sys.executable` into the
unit, so it points at the right Python.

A user timer rather than a system one: the vault is in a home directory and
Ollama runs as the user, so a system unit would need permissions it has no
business having. It is persistent, because the machine is a laptop — a timer
that only fires at 23:00 fires never on a machine that is asleep at 23:00.

`deploy/` holds the units this installs, if you would rather place them by
hand.

## Flaws found by using it

None of these were caught by the tests. They are here because a system that
only lists its strengths is not telling you much.

- Two components were writing to different databases. Both worked, separately,
  and the disagreement stayed invisible until something read what the other
  had written. One function decides the path now.
- Open WebUI caches tool modules, so a change on the host was silently
  ignored and the old code kept answering. The tool file now clears its own
  modules from the cache as it loads.
- The transcript froze on its first write and stayed frozen.

## Tests

```bash
for t in run_tests test_index test_search test_tools test_write test_analysis \
         test_stale test_queue test_repair test_harness test_transcript \
         test_concepts test_maintain test_timer; do
  echo -n "$t: "; python3 -m sentinel.tests.$t 2>&1 | grep passed
done
```

595 checks over fourteen modules. The `field_test_*` modules are separate:
they run against a real vault and a real model rather than fixtures, and they
take minutes.

## The Searcher

[The Searcher](https://github.com/farukhanci/the-searcher) is a separate
service that researches a question on the web and writes verified passages
into the vault's `sources/` directory. Neither repository imports the other.

What joins them is the system prompt, which describes `research` alongside
the seven vault primitives and says when each applies: the vault first, the
web only when the user asks for it, and an empty vault is not itself a reason
to go looking. Run the Sentinel without the Searcher and everything works
except that one tool.

## License

MIT — see `LICENSE`.

Models are separate and carry their own terms.
