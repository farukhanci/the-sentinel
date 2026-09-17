# The Sentinel

An Obsidian vault a model can read and write through seven primitives.

The Sentinel puts a local model in front of a markdown vault: it searches by
meaning and by wording at once, follows the link structure, opens pages to a
bounded depth, and writes back. A nightly pass summarises new pages, extracts
the concepts they refer to, and embeds them, so the search substrate stays
current without anyone typing a command.

Everything runs locally by default — a 9B orchestrator on a 6 GB card,
embeddings on the CPU, no account anywhere. Any role can be pointed at a
cloud model instead; the gates are in the code, so they hold either way.

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
conversation is not evidence. Deciding unprompted was tried across five
phrasings of the instruction: each time the model restated the rule correctly
in its own reasoning and then ended the turn with an offer instead of a call.
Waiting to be asked costs nothing, because the alternative was nothing
happening.

**And the record the fourth gate rests on cannot be edited by the model.**
`write` refuses a transcript outright — not a retry, a stop, because no
rewording of the request would make it succeed. Silencing concept extraction
over conversations means nothing if the model can rewrite the conversation
first, and it tried: caught once planning to "reconstruct the transcript with
additions".

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

**A tool result is read as an answer, not as an observation.** The model
treats what a tool returns as the thing that finishes the turn, and that
assumption has to be designed around rather than argued with. `[STOP] you
already searched for this` was read as "no results" and the model searched
three more times in the same turn, ending on a 900-character query. A `graph`
listing of concepts with no page yet was read as a list of things to open: it
guessed at paths and spent eight of twenty tool calls on pages that by
definition did not exist. So every result opens with a status — `[OK]`,
`[OK → next]`, `[DONE]`, `[RETRY]`, `[STOP]`, `[more]`, `[note]` — and each
one answers a different control-flow question. `[RETRY]` and `[STOP]` are
separate because collapsing them means either retrying the impossible or
giving up on the fixable. `[more]` exists because a truncated answer is
indistinguishable from a complete one.

**Rules in the tool schema hold; rules in the system prompt do not.** The same
instruction was written five ways into the system prompt. Each time the model
restated it correctly in its own reasoning and then did the opposite. Moved
into the tool's own description — one sentence — it held immediately. The
likely reason is that tool-calling training treats the schema as how the tool
works and the prompt as text that happened to arrive. The cost is real and was
measured: the descriptions grew from an estimated 1592 tokens to 2214, and
every token of that difference is a rule that had to be moved after a failure.

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

Summary and concept list, on the other hand, are merged into one call — and
that merge is safe only because both are extractive over a text that already
exists. An earlier merge was not: writing the prose *and* bracketing it put a
generative job and an extractive one in the same call, and the second was done
unreliably. The Discussion section came back with zero usable markers, because
the model had wrapped them all in backticks.

## Four folders

Each one marks a different stage of where something came from, and each stage
is treated differently.

| | summary search | places links | writing |
| --- | --- | --- | --- |
| `notes/` — what you chose to keep | yes | **yes** | free; a path with no folder lands here |
| `wiki/` — what the sources support | yes | yes | creating refused, editing free |
| `conversations/` — what was said | no | no | **refused outright** |
| `sources/` — what was read | no | no | free |

Staying out of summary search is not exclusion: those pages are still found at
the body rung. Nothing said is lost, it just is not promoted unasked.

Placing links is what feeds the graph, and only the first two do it. The
direct consequence is that the growth queue counts pages *you* kept, not a
paper mentioning a term twenty times.

That arrangement is the second one. At first the transcript was the authority:
if a summary invented something, it would not appear in the raw record, so it
could not become a concept. The gate worked and the side effect was measured —
because the transcript placed links, the model's own turns fed the graph. One
pass produced `Obsidian vault` and `Sentinel`, from the assistant introducing
itself, and `capacitance of the sphere` and `breakdown field of air`, from
physics nobody had asked about. None became pages; the definition gate held.
They filled the queue instead. Reversed — transcripts silent, `notes/` the
authority — the same pass on the same vault produced `growth queue`,
`maintenance pass` and `index database`, all from the one page that had been
deliberately kept. The noise class disappeared entirely.

### Tracing a claim back

Every link in the chain is written by code, not by the model.

A concept page lists in its frontmatter which files it was written from. Each
of those is a Searcher record under `sources/`, holding the original question,
the passages taken, and the URL each came from — including the pages the
reader could not use, listed separately.

A page written from a conversation carries a `Kayıt:` line to its transcript
under `conversations/`, where the exchange sits verbatim. A page added to from
several conversations carries one line per conversation, so it shows all of
its own history rather than half of it.

## Two ways in

**[Open WebUI](https://docs.openwebui.com)** is the everyday one.
`openwebui_tool.py` loads into it and hands the seven primitives to whatever
model you are talking to. It is
also the only path that can see `__messages__` and `__metadata__`, which is
what the raw capture mechanism is built on: the code copies the whole
conversation and finds the file again by `chat_id`.

**`sentinel.chat`** goes straight to Ollama and runs the harness: the tool
loop, the status markers that drive control flow, and token accounting. Use
it when something needs diagnosing — `/steps`, `/budget`, `/reset` show you
what Open WebUI hides. This path is Ollama-only; a cloud model reaches the
same seven primitives through Open WebUI instead.

An HTTP tool server used to be a third path. It was removed: it could not see
`__messages__`, so capture could not work through it, and its reason for
existing did not survive contact with the mount setup below.

## What it does not do

**It does not verify claims.** Search finds pages; it does not judge them.
Summaries are extractive but unchecked. What the vault says is what you get.
The gates above stop the system inventing *new* material — they say nothing
about whether what you saved was right.

**Maintenance does not run during a conversation.** The analysis model and the
conversation model do not fit on a card this size at once, so a pass evicts
whatever is loaded. It runs overnight, on a timer, and surfaces itself through the health
line rather than interrupting. It is not slow — about 12 seconds a page; seven
pages took 139 seconds and two took 30 on the last passes. It was slow once:
with the thinking block on, the model produced 5620 tokens for a two-field
JSON object, and turning it off took generation from 184 seconds to 2.5.

**`write`, `relocate` and `remove` are real, and there is no
authentication.** See the security note below. The index does refuse one
thing on its own: if every page disappears at once it declines to delete
anything, because that is an unmounted drive rather than an edit, and a real
emptying is recovered with an explicit rebuild.

**The vault this runs on is small.** 46 files. Every gate above is
covered by a test that fails when the gate is removed, so the mechanism is not
in doubt — but keeping a knowledge base from filling with low-information
pages is a claim about scale, and this has not been run at scale. The numbers
quoted in this file come from that vault, not from a large one.

## How it is meant to be used

This describes the design, not experience of it. The system is days old and
the vault holds 46 files: 33 under `sources/`, 7 under `conversations/`, one
note, and `wiki/` empty — the maintenance pass has not yet written a concept
page.

**Keeping something.** The trigger is the user asking: "kaydet", "save this",
"note this". The model does not decide on its own and does not offer.

What happens then is code, not the model. The conversation is copied whole
into `conversations/`, found again by its chat id. The page is written into
`notes/`, named by subject. A `Kayıt:` line pointing at the transcript is
added to the page body.

A second save in the same chat adds to the same page rather than replacing it,
and adds no second `Kayıt:` line — that conversation is already recorded. A
save from a *different* chat does add one, so a page shows all of its own
history rather than half of it. The transcript is refreshed on every save, so
it always holds the whole exchange rather than freezing at the first one.

**Where a concept page comes from.** Two ways, and the second exists because
of a restriction.

The maintenance pass writes one when the sources support it: paragraphs are
collected from the pages that link to the concept, each is put to the model as
*does this say what the thing is*, and two defining sentences are enough. One
is not, and none means the concept stays in the queue as something to go and
read about.

Or the user opens the page. Clicking a faded link in Obsidian creates a file
with that name, and from that point the model can fill it in conversation —
because creating a page under `wiki/` from a chat is refused, while editing an
existing one is not. That folder means "concepts the sources define", and a
page put there from a conversation does not carry that claim. The maintenance
pass then leaves it alone: its first check is whether a page already resolves
from the name.

**The nightly pass.** It runs at 23:00 and is read with
`journalctl --user -u sentinel-maintenance`. One run looked like this:

```
analysis
  [1/2] conversations/2026-09-03-1.md: ok 4 concepts, 9.4s
  [2/2] wiki/settled decisions.md: ok 8 concepts, load 0.3s prefill 0.3s gen 3.5s
concept pages
  analysis pass: skipped - mentioned but never defined in the sources
  chat interface: skipped - mentioned but never defined in the sources
  analysed 2   embedded 7   written 0   seconds 37.4
  resolved {'same': 0, 'different': 3, 'unclear': 4, 'unconfirmed': 1}
```

`unconfirmed` counts merge decisions that did not hold when the pair was
swapped and the question asked again. This run had one. Two pages took 37
seconds, seven took 139, peak memory 1.5 GB.

**The health line.** Appended to tool output, and only when there is something
to say: pages awaiting analysis, failed analyses, concepts four or more pages
now reference but nothing has defined, and concept pages past twelve sections.
The last two have never fired here — the vault has not reached that scale.

It never acts. Whether a bloated page should be split is a judgement about the
subject, and it is left to the person.

## Requirements

- Python 3.11 or later, with `venv` and `pip`
- [Ollama](https://ollama.com) with a tool-calling model
- A vault — any directory of markdown files with frontmatter. Nothing here
  needs [Obsidian](https://obsidian.md) running, but one thing needs it
  installed; see step 4 below
- multilingual-e5-small as ONNX, on disk
- A GPU with 4 GB or more; embeddings run on the CPU, and the measured numbers are under Models below

On a bare Debian or Ubuntu, the system packages come first:

```bash
sudo apt install git python3 python3-venv python3-pip docker.io
sudo usermod -aG docker $USER   # then log out and back in
```

The Docker part is only needed for Open WebUI. Without the group change every
`docker` command fails on a permission error at the socket.

On Windows, run it under WSL2 and follow the Linux instructions as written —
the systemd units, the timer and the Docker paths all work there unchanged.
Two things to get right: keep the vault inside the WSL filesystem rather than
under `/mnt/c`, where crossing the boundary costs more than it sounds like it
should, and check that systemd is enabled (`systemctl --user status` answers
if it is; otherwise `systemd=true` under `[boot]` in `/etc/wsl.conf`).

The package itself has no POSIX dependency and the stored paths are
normalised, so it should also run on Windows directly — but nothing here has
been run that way, and the units and Docker flags in this file assume Linux.

Embeddings run on the CPU on purpose: search must never compete with the
conversation model for VRAM. multilingual-e5-small was chosen over
bge-small-en-v1.5 because that one is English-only and would have failed
silently on a bilingual vault. Same 384 dimensions, so the swap cost nothing
structurally — but changing it again invalidates every stored vector and
forces a full re-embed.

Context size is a per-model measurement, not a constant. `CONTEXT_TOKENS` in
`harness.py` is 17000, the figure the design was sized around; in practice
17000 gave CUDA OOM on one model and another runs at 35000. What each takes
is measured under Models below; the method for finding your own is in step 5.

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

### What they actually take

Measured on an RTX 4050 Mobile with the desktop on the integrated GPU, so the
card held nothing but the model — `num_gpu` high enough that `ollama ps`
reported the whole thing on the GPU, and read after a first request so the KV
cache was populated.

| Model | Context | VRAM |
| --- | --- | --- |
| `qwen3.5-4b-xl` | 35000 | 3.7 GB |
| `qwen3.5-4b-xl` | 120000 | 5.6 GB |
| `hf.co/AtomicChat/Ornith-1.5-9B-GGUF:IQ4_XS` | 35000 | 5.6 GB |

The context window matters about as much as the model does: the 4B at 120k
costs what the 9B costs at 35k. That is why the table above is a record of
measurements rather than a recommendation.

**On a 4 GB card** the 4B model at 35000 fits with room to spare, and it can
take all three roles. The conversation gets worse — a 4B answering is not a
9B answering — but nothing about the design changes, because the gates are in
the code rather than in the model's judgement.

**On a bigger card, spend it on the conversation and nowhere else.** That is
the one role where a better model produces a better result. On 24 GB a Q4
build of something in the 27B class fits with context to spare — Qwen3.8-27B's
Q4_K_M is around 16 GB of weights, which is why 24 GB rather than 16 GB is the
honest floor for it.

Upgrading extraction or summaries is close to pointless. Both are transport
jobs: copy the concepts as they appear in the text, say what this page is.
Both run at `temperature: 0` for that reason, with no sampling parameters at
all — the same page should produce the same concepts. A larger model does that
work more slowly and no more correctly. The 9B already scores its own output
past every gate; the gates are what decide, not the model's judgement.

More context is not automatically better either. The conversation model in use
holds up to about 32k and degrades past it, so the headroom goes into fitting
the model comfortably rather than into a larger window. Measure where yours
starts to drift.

What a bigger card would genuinely unlock is the one constraint this design
works around: the analysis and conversation models cannot be resident at once,
which is why maintenance runs at night. With room for both, maintenance could
run when the vault is idle rather than on a timer. Nothing here does that —
the pass is written to be scheduled, and making it opportunistic is a change
to the maintenance loop, not a setting.

### None of these has to be local

The roles are independent, and nothing in the design assumes a small model —
the gates are in the code, so they hold whatever is answering.

**Conversation** is Open WebUI's to decide. It connects to OpenAI-shaped
providers and to Anthropic directly, under Admin Settings → Connections, and
whatever model you pick there gets the seven primitives. The tool runs on the
host and reaches the vault from there, so it does not care who called it.

**Extraction and summaries** take their own flags, and each role is chosen
separately — a large-context summary model in the cloud and extraction on the
card is a reasonable split:

```bash
python3 -m sentinel.maintain --vault ~/obsidian/YourVault \
    --model qwen3.5-9b-jinja --num-ctx 8192 \
    --summary-provider openai \
    --summary-base-url https://api.openai.com/v1 \
    --summary-api-key-env OPENAI_API_KEY \
    --summary-model gpt-4o-mini --summary-num-ctx 120000
```

The key is read from the environment by name, never passed on the command
line where it would land in shell history and in `ps`. Giving no provider
flags leaves everything exactly as it was.

Two things are worse over an API, and both are reported rather than hidden.
The OpenAI shape returns one total instead of load, prefill and generation
separately — the breakdown that made three wrong diagnoses obvious — so the
progress line falls back to wall clock. And turning thinking off has no
standard field: it is sent the way several servers accept it, and if the
endpoint refuses it or reasons anyway, the run says so once rather than
quietly paying for it.

One limit cannot be fixed from here. The truncation gate works by comparing
what the server says it evaluated against the context window; an endpoint that
returns no `usage` block reports zero, and the gate cannot fire. A page whose
tail was never read would then be analysed and marked done. There is a test
standing on that limit so it stays visible.

## Setup

### 1. The package

```bash
git clone https://github.com/farukhanci/the-sentinel
cd the-sentinel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama and the models

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama pull hf.co/AtomicChat/Ornith-1.5-9B-GGUF:IQ4_XS   # conversation
ollama pull qwen3.5-9b-jinja                             # concept extraction
ollama pull qwen3.5-4b-xl                                # summaries
```

Substitute your own — any tool-calling model works. Ollama listens on
`http://localhost:11434`, which is where this expects to find it.

### 3. The embedding model

Two files are needed: the ONNX graph and the tokenizer. Downloading the whole
repository would also pull PyTorch weights that go unused. `huggingface_hub`
came in with the requirements above, so `hf` is already on the path.

```bash
hf download intfloat/multilingual-e5-small \
    onnx/model.onnx tokenizer.json \
    --local-dir models/multilingual-e5-small
```

Inside the repository on purpose. Open WebUI runs in a container and the
repository is mounted into it, so a model that sits here is visible there
without a second mount to remember. `models/` is gitignored.

The loader accepts `<dir>/onnx/model.onnx` or `<dir>/model.onnx`, with
`tokenizer.json` beside it either way, and says which one is missing if one is.

### 4. The vault

Any directory will do, and the folders the system uses appear the first time
something is written into them:

```bash
mkdir -p ~/obsidian/YourVault
```

That is enough for everything except one thing, and the exception is worth
setting up now. A concept the sources define gets written by the maintenance
pass on its own. A concept you want a page for *before* that happens is
created by hand — and the way you do it is to click the faded link in
[Obsidian](https://obsidian.md) and let it make the file, because the model
is refused when it tries to create a page under `wiki/` itself.

So: install Obsidian, open that directory with **Open folder as vault**, and
under Settings → Files and links set **Default location for new notes** to a
folder — `wiki/`, if you want faded links to land where concept pages live.
Obsidian puts new files wherever that setting says, and a link clicked into
existence follows it.

### 5. Talk to the vault

```bash
python3 -m sentinel.chat \
    --vault ~/obsidian/YourVault \
    --model hf.co/AtomicChat/Ornith-1.5-9B-GGUF:IQ4_XS \
    --num-ctx 35000 \
    --embed-model models/multilingual-e5-small
```

35000 is what that model takes on this card, not a setting to copy. Find your
own: keep `num_gpu` high enough to put every layer on the card, then lower
`--num-ctx` until `ollama ps` reports 100% GPU instead of a CPU/GPU split.
The split is the thing to avoid, and that readout is the only reliable way
to see it.

Leave `--embed-model` out and it still runs — on the literal half of search
alone, quietly, with no error. That is half the substrate missing and nothing
says so, which is why it is in every command in this file.

**Run the maintenance pass once before you rely on search.** Summaries and
concepts come from the analysis pass, and the summary rung — the one search
normally stops at — has nothing to stop at until that pass has run. On an
existing vault this takes a while, so start it and leave it:

```bash
python3 -m sentinel.maintain \
    --vault ~/obsidian/YourVault \
    --model qwen3.5-9b-jinja --num-ctx 8192 \
    --summary-model qwen3.5-4b-xl --summary-num-ctx 120000 \
    --embed-model models/multilingual-e5-small \
    --minutes 60
```

`--summary-model` is what gives summaries the whole document at once, which
is the split the Models table above describes. The two models are loaded one
after the other, not per page.

`--dry-run` first counts what the vault owes without touching the model, which
is worth knowing before a first run of unknown length. The folders the system
uses — `notes/`, `wiki/`, `conversations/`, `sources/` — are created when
something is first written to them; an empty vault needs no preparation.

## Open WebUI

Open WebUI runs in Docker here, so it needs two bind mounts: this repository,
and the vault. Nothing is installed inside the container — the tool file puts
the repository on `sys.path` and imports from there.

```bash
docker run -d --name open-webui -p 3000:8080 \
  --user $(id -u):$(id -g) \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  -v ~/the-sentinel:/sentinel:ro \
  -v ~/obsidian/YourVault:/vault \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  ghcr.io/open-webui/open-webui:main
```

Open WebUI is then at `http://localhost:3000`. Ollama runs on the host, and
`127.0.0.1` inside a container is the container — hence `--add-host` and the
base URL together. It can also be set afterwards under Settings →
Connections, but a container that comes up already knowing where Ollama is
saves finding out the hard way that the model list is empty.

`--user` is not optional in practice: a container running as root writes root-owned files into the vault,
and nothing on the host can edit them afterwards. A bind mount cannot be added
to a running container, so an existing one has to be recreated — the data
survives in the named volume.

Then paste `openwebui_tool.py` into Open WebUI under Workspace → Tools → +,
and enable it for the model you talk to.

Everything else is configured through that tool's valves — the gear icon next
to it in the tools list. The paths there are as the **container** sees them,
not the host: with the mounts above the vault is `/vault`, and the embedding
model is `/sentinel/models/multilingual-e5-small`, which is why step 3 put it
inside the repository. Leave the index path empty unless you have a reason —
empty means the tool derives it the same way every other entry point does,
and a second copy of that path is how the index once split in two.

Then make a model. Workspace → Models → + takes a base model and wraps it
with everything this needs in one place, which is where the rest of the setup
lives:

**The system prompt** — the contents of `openwebui_system_prompt.txt`. Not
optional decoration: it is where the ladder is explained, where writing is
tied to the user asking for it rather than the model deciding, and where the
status markers are defined. Without it the model has seven tools and no idea
when to stop.

**The tool**, ticked on for this model.

**The sampling and runtime parameters**, under Advanced Params on the same
screen. The values in the Models table above go here, and one of them is not
cosmetic: `num_gpu`. Leave it unset and Ollama uses its own estimate of what
fits, which leaves part of the model on the CPU — the system works and is
slow, with nothing anywhere saying why. Set it high enough to mean every
layer, and set `num_ctx` to whatever you measured. `ollama ps` while a
conversation is running tells you which of the two happened.

This is the step to get right. The first person other than me to install this
did everything else correctly, left the parameters at their defaults, and got
a system that ran at a fraction of the speed with no error to explain it.

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
    --summary-model your-summary-model --summary-num-ctx 120000 \
    --embed-model models/multilingual-e5-small \
    --at 23:00 --minutes 30
```

Install it from inside the venv — the timer writes `sys.executable` into the
unit, so it points at the right Python. Check that it took:

```bash
systemctl --user list-timers sentinel-maintenance.timer
```

And `loginctl enable-linger $USER`, or the user session — and every timer in
it — ends when you log out. On a system without polkit that needs `sudo`.

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

One conclusion in this file was withdrawn rather than fixed. The 253 seconds
above were once used to argue for a smaller analysis model; with the cause
still unknown, model size was never established as the variable, so the
argument went. The lesson that replaced it is duller and more useful: report
load, prefill and generation separately. Collapsing them into one total
produced three wrong diagnoses in a row.

## Tests

```bash
for t in run_tests test_index test_search test_tools test_write test_analysis \
         test_stale test_queue test_repair test_harness test_transcript \
         test_concepts test_maintain test_timer test_models; do
  echo -n "$t: "; python3 -m sentinel.tests.$t 2>&1 | grep passed
done
```

659 checks over fifteen modules. The `field_test_*` modules are separate:
they run against a real vault and a real model rather than fixtures, and they
take minutes.

## The Searcher

[The Searcher](https://github.com/farukhanci/the-searcher) is a separate
service that researches a question on the web and writes verified passages
into the vault's `sources/` directory. Neither repository imports the other.

What joins them is the system prompt, which describes `research` alongside
the seven vault primitives and says when each applies: the vault first, the
web only when the user asks for it, and an empty vault is not itself a reason
to go looking.

They are installed separately: the Searcher runs as its own service and is
registered in Open WebUI under Settings → Tools as an external tool server.
Run the Sentinel without it and everything works except that one tool — the
prompt will still describe `research`, which is worth trimming if you are not
going to install it.

## License

MIT — see `LICENSE`.

Models are separate and carry their own terms.
