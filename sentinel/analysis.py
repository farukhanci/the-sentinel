"""The Sentinel - the analysis pass.

ONE model, ONE clean context, ONE read, TWO structured outputs. Not "the
summary system" and "the linking system" - one pass.

The merge is safe here where an earlier one was not: that pair was GENERATIVE
plus extractive (write the prose AND bracket it), and a model doing two jobs
does the second one unreliably - measured, the Discussion section produced
zero usable markers because the model wrapped them in backticks. Summary and
concept list are BOTH extractive over an existing text.

The quantity gate survives the merge because the output has two named fields,
which can be gated INDEPENDENTLY. Gain: prefill halves, ~3000 -> ~1500 tokens,
and two VRAM swaps become one.

This pass NEVER runs during a conversation. Measured at ~253 seconds per page;
fifty pending pages would be three and a half hours. It surfaces itself
through the health line and runs at a natural pause.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

from pathlib import Path

from .index import parse_page
from .output import cut_point
from .text import (
    concept_present,
    shape_ok,
    link_body,
    normalize,
    page_hash,
    split_frontmatter,
    strip_links,
)
from .tools import render_page

# The freeze candidate. Passed on first use: clean JSON, no fence, no preamble,
# summary gate passed, 14 of 15 concepts verbatim.
PROMPT = """Read the text below and return a JSON object with exactly two fields.

"summary": ONE sentence saying what this page is about. Not a list, not two
sentences, under 300 characters. State the subject directly - do not begin
with "This page", "This document" or "This article".
"concepts": the important concept names that appear in the text, copied as they
appear. Names of things, ideas, methods, objects. Not measurements, not units,
not section labels, not whole phrases. Some names in the text are wrapped in
[[double brackets]] - give the name only, without the brackets.

Return only the JSON object. No explanation, no markdown fences, no preamble.

---
{text}
"""

CONCEPT_FLOOR = 0.25 / 100   # surviving concepts per word. 1 per 400 words.
SUMMARY_CAP = 300

# Only a coarse pre-filter, so it is GENEROUS on purpose: its job is to catch
# the page that obviously cannot fit without spending a model call, not to
# decide the marginal case. The marginal case is decided by measurement after
# the call - see the truncation check below.
CHARS_PER_TOKEN = 4
PROMPT_OVERHEAD = 200        # the instructions above the text
OUTPUT_RESERVE = 600         # room for the JSON to come back


# ---------------------------------------------------------------------------
# Output repair
# ---------------------------------------------------------------------------


def repair_json(raw: str) -> tuple[dict | None, list[str]]:
    """Returns (parsed, repairs applied).

    The repairs are REPORTED rather than swallowed, because which ones were
    needed is the signal about prompt quality. A fence or a sentence of
    preamble around the object is the single most common corruption.
    """
    repairs: list[str] = []
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|```$", "", text).strip()
        repairs.append("code fence")
    if not text.startswith("{") and "{" in text:
        text = text[text.index("{"):]
        repairs.append("preamble")
    if not text.endswith("}") and "}" in text:
        text = text[:text.rindex("}") + 1]
        repairs.append("trailing text")
    try:
        return json.loads(text), repairs
    except Exception:
        pass
    # A trailing comma before a closing bracket is the commonest malformed
    # JSON a model produces, and it is unambiguous to fix - unlike, say,
    # single quotes, where repairing means guessing at the author's intent.
    # Every repair here has to be one where there is only one thing it could
    # have meant.
    patched = re.sub(r",(\s*[}\]])", r"\1", text)
    if patched != text:
        try:
            return json.loads(patched), repairs + ["trailing comma"]
        except Exception:
            pass
    return None, repairs + ["unparseable"]


# ---------------------------------------------------------------------------
# The gates - two fields, two INDEPENDENT gates
# ---------------------------------------------------------------------------


def gate_summary(summary: str) -> str | None:
    """None if it passes, else the reason."""
    s = (summary or "").strip()
    if not s:
        return "empty"
    if len(s) > SUMMARY_CAP:
        return f"{len(s)} chars, over {SUMMARY_CAP}"
    if len(re.findall(r"[.!?](?:\s|$)", s)) > 1:
        return "more than one sentence"
    return None


def gate_concepts(concepts: list[str], body: str) -> tuple[list[str], bool]:
    """(surviving concepts, whether the yield is below the floor).

    NOT all-or-nothing. Field test 2 had 14 of 15 verbatim; one entry drifted
    (`compactness problem argument` against the text's `compactness problem`),
    and an all-or-nothing gate would have discarded all 14 and spent four
    minutes retrying to save a single unusable one. A non-verbatim concept
    cannot be linked anyway, so dropping it costs nothing.

    The floor is 1 per 400 words, calibrated on real sections. An earlier
    figure of 1 per 100 words was four times too strict - it would have
    flagged every section including the three that worked.
    """
    stripped = strip_links(body)
    survivors, seen = [], set()
    for c in concepts or []:
        # Strip link markup off the concept itself. Pages written by the
        # earlier pipeline already carry `[[...]]`, and "copied as they
        # appear" then yields `[[GRB]]` - measured on the real vault. The
        # shape filter would reject it and the presence check would not find
        # it, so the concept is simply lost. The same rule as everywhere else
        # applies: the markup is not the content.
        c = strip_links(c.strip()).strip()
        key = normalize(c)
        if key and key not in seen and concept_present(stripped, c):
            seen.add(key)
            survivors.append(c)
    words = max(len(stripped.split()), 1)
    return survivors, len(survivors) / words < CONCEPT_FLOOR


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


SUMMARY_PROMPT = """Read the text below and return a JSON object with one field.

"summary": ONE sentence saying what this text is about, covering what it
concludes and not only what it opens with. Not a list, not two sentences, under
250 characters. Write it in the language the text is written in.

Return only the JSON object. No explanation, no markdown fences, no preamble.

---
{text}
"""


def summarise_page(sentinel, path: str, model) -> dict:
    """The summary, from the WHOLE page, in its own call.

    Split out from concept extraction because the two want opposite things. A
    summary is holistic - it cannot be produced from a window, which is why a
    transcript went without one and sat in every listing as `who are you?`.
    Extraction is local: a name is in a paragraph or it is not.

    So they get different models. A small model with a very large context
    (measured: Qwen3.5-4B, 120 000 tokens in 5.6 GB) reads a whole document at
    once; the larger model keeps the extraction work, where its accuracy on
    what counts as a concept was measured and is worth its context limit.

    The risk this has to clear is context rot - the U-curve, where a model
    attends to the start and end of a long input and loses the middle, ≥30%
    across 18 models. Tested on a real 40 000-character paper by asking about
    a section near the END: the answer was accurate and detailed, and the
    summary reflected material from that section. It reads the whole document.
    """
    idx = sentinel.index
    f = idx.vault / path
    if not os.access(f, os.W_OK):
        return {"path": path, "status": "failed", "summary": None,
                "note": "file is not writable", "environmental": True}

    fm, body = split_frontmatter(f.read_text(encoding="utf-8"))
    ctx = getattr(model, "num_ctx", 8192)
    room = (ctx - PROMPT_OVERHEAD - OUTPUT_RESERVE) * CHARS_PER_TOKEN

    text, truncated = body, False
    if len(body) > room:
        # As much as fits, cut at a boundary. A partial summary of a very long
        # page is worth more than the first sentence of it, and the summary's
        # job is to say whether the page is worth opening.
        cut, _ = cut_point(body[:room], room)
        text, truncated = body[:cut], True

    raw, timings = model.complete(SUMMARY_PROMPT.format(text=text))
    obj, repairs = repair_json(raw)
    result = {"path": path, "status": "ok", "note": "", "repairs": repairs,
              "truncated": truncated, "seconds": timings.get("total", 0.0)}
    if obj is None:
        result.update(status="failed", summary=None,
                      note="unparseable summary response")
        return result

    summary = str(obj.get("summary") or "").strip()
    bad = gate_summary(summary)
    if bad:
        result.update(status="failed", summary=None, note=bad)
        return result

    fm["summary"] = summary
    fm["summary_provisional"] = "0"
    try:
        f.write_text(render_page(fm, body), encoding="utf-8")
    except OSError as e:
        result.update(status="failed", summary=None,
                      note=f"could not write: {e.strerror}", environmental=True)
        return result

    idx._replace_page(parse_page(path, f.read_text(encoding="utf-8")),
                      f.stat().st_mtime)
    idx.db.commit()
    result["summary"] = summary
    if truncated:
        result["note"] = (f"page is longer than the summary model's context; "
                          f"summarised the first {len(text)} of {len(body)} "
                          f"characters")
    return result


def analyze_page(sentinel, path: str, model, retry: bool = True,
                 summary_model=None) -> dict:
    """One page: call, gates, one retry, summary, links, index.

    Order matters and step 2 is not optional - the first-occurrence set is
    seeded from the links already in the body, inside link_body. Unseeded, a
    re-run double-links every concept that was already linked.
    """
    idx = sentinel.index
    f = idx.vault / path

    # Check writability BEFORE spending a model call on it. Measured: a
    # read-only page in the vault crashed the pass outright, and it would have
    # crashed after three minutes of generation - the whole cost paid for a
    # write that could never land.
    if not os.access(f, os.W_OK):
        return {"path": path, "repairs": [], "status": "failed",
                "note": "file is not writable", "concepts": 0,
                "seconds": 0.0, "timings": {},
                "environmental": True}

    raw = f.read_text(encoding="utf-8")
    fm, body = split_frontmatter(raw)
    before_hash = page_hash(raw)

    # A TRANSCRIPT is analysed differently, and the difference is the whole
    # point of keeping one.
    #
    # It is the record of what was actually said, so it is the authority for
    # which concepts are real - measured on a captured conversation, the page
    # the model wrote about it had added a computed "~73x faster" and a
    # section of steps nobody took. Those are not in the transcript, so they
    # never become concepts.
    #
    # Concepts come window by window, because a conversation has no length
    # limit and refusing to read a long one would defeat the record. No
    # summary is written: a summary needs the whole text at once, which is
    # exactly what a windowed read cannot give, and the transcript already
    # carries the one `write` placed.
    if fm.get("type") == "transcript":
        return _analyze_windowed(sentinel, path, f, fm, body, before_hash,
                                 model, summary_model)

    # DOES THE PAGE FIT? Nothing else asks this, and the failure without it is
    # silent in the worst way: Ollama truncates an over-long prompt without
    # complaint, the model summarises whatever survived the cut, the gates see
    # a well-formed summary and a plausible concept list, and `analyzed_hash`
    # is written. The page is then marked analysed on the strength of its
    # first few pages.
    #
    # Measured on the real vault: a raw arxiv download of 1,206,117 characters
    # sat in the queue behind an 8192-token context.
    ctx = getattr(model, "num_ctx", 8192)
    room = (ctx - PROMPT_OVERHEAD - OUTPUT_RESERVE) * CHARS_PER_TOKEN
    if len(body) > room:
        # TOO LARGE IS NOT A FAILURE, it is a different way of reading.
        #
        # It used to fail here, and that was wrong for the pages this system
        # produces: a page written from conversation GROWS, section by
        # section, and one day crosses the line. Refusing it would mean the
        # page silently stops being maintained on the day it becomes the most
        # substantial thing in the vault.
        #
        # So it is read window by window instead, exactly as a transcript is.
        # What is lost is the summary - that needs the whole text at once -
        # and the page keeps the one it already has, which was written when it
        # still fitted.
        r = _analyze_windowed(sentinel, path, f, fm, body, before_hash, model,
                              summary_model)
        r["note"] = ((r["note"] + "; ") if r["note"] else "") + (
            f"{len(body)} chars is past the {ctx}-token context, so it was "
            f"read in windows and kept its existing summary")
        return r

    result = {"path": path, "repairs": [], "status": "ok", "note": "",
              "concepts": 0, "seconds": 0.0, "timings": {}}

    attempts = 2 if retry else 1
    for attempt in range(attempts):
        t0 = time.perf_counter()
        text, timings = model.complete(PROMPT.format(text=body))
        result["seconds"] += time.perf_counter() - t0
        result["timings"] = timings
        # DID THE PROMPT ACTUALLY FIT? The pre-filter above is an estimate;
        # this is the measurement. Ollama silently drops what does not fit and
        # reports how much it evaluated, so a prompt_eval_count sitting at the
        # ceiling means the tail of the page was never read - and a summary of
        # a page's first half, gated and accepted, is worse than no summary.
        used = timings.get("prompt_tokens", 0)
        if used and used >= ctx - OUTPUT_RESERVE // 2:
            result.update(
                status="failed",
                note=f"prompt filled the context ({used} of {ctx}) - the page "
                     f"was truncated before the model saw the end of it")
            _record(idx, path, before_hash, "failed", result["note"])
            return result

        obj, repairs = repair_json(text)
        result["repairs"] = repairs
        if obj is None:
            # KEEP WHAT IT SAID. Recording only "unparseable" throws away the
            # one thing that could explain the failure, and a failure nobody
            # can diagnose comes back every time the page is edited. Measured
            # on the real vault: one section of one paper failed twice and
            # left nothing behind to look at.
            snippet = " ".join(text.split())[:300]
            result["status"] = "failed"
            result["note"] = f"unparseable output: {snippet}"
            result["raw"] = text
            continue
        why = gate_summary(obj.get("summary", ""))
        if why:
            result["status"], result["note"] = "failed", f"summary gate: {why}"
            continue
        concepts, low = gate_concepts(obj.get("concepts", []), body)
        if low and attempt + 1 < attempts:
            # ONE retry. A low yield is not always the model's fault: a page
            # that is 5000 characters of formulas genuinely contains no
            # concepts, and no retry can change that. Never loop.
            continue
        result.update(status="low_yield" if low else "ok",
                      note="below the concept floor" if low else "",
                      summary=obj["summary"].strip(), concepts=len(concepts))
        result["survivors"] = concepts
        break
    else:
        if result["status"] != "failed":
            result["status"] = "failed"

    if result["status"] == "failed":
        # Keep the provisional summary, leave the page unlinked, and record
        # the failure AGAINST THE HASH - so the same content is not retried
        # every pass, but the state stays visible rather than silent.
        _record(idx, path, before_hash, result["status"], result["note"],
                result.get("environmental", False))
        return result

    # The GATED list, not the model's raw one. A concept the gate dropped has
    # no business reaching the linker, even though it could not have matched.
    #
    # And on a DERIVED page, only names the graph already knows. A concept
    # page is written from other pages; if the names it happens to contain
    # became new concepts, the system would be discovering concepts in its own
    # output, and the next generation would be written from that. The measured
    # shape of this failure is knowledge collapse: fluency survives while the
    # facts drift, and it accelerates with the proportion of generated text in
    # the corpus.
    #
    # Linking a name the graph already knows is not discovery, it is wiring,
    # and that stays.
    survivors = result["survivors"]

    # THE AUTHORITY IS WHAT THE USER CHOSE TO KEEP, not everything that was
    # said. A note in `notes/` exists because he asked for it; a transcript
    # exists because the code wrote it. So the note carries the links and the
    # transcript carries none.
    #
    # It ran the other way first, and the cost was measured: concepts came off
    # the transcript, which meant the model's own turns fed the graph.
    # `Obsidian vault` and `Sentinel` entered as concepts from the assistant
    # describing its own tools; `capacitance of the sphere` and `breakdown
    # field of air` entered from the model reciting physics nobody had asked
    # about. The definition gate kept every one of them from becoming a page,
    # but they still filled the queue.
    #
    # A transcript still gets a summary and is still searchable at rung 2.
    # Nothing said is lost - it just does not get promoted unasked.
    if sentinel.sources and path.startswith(sentinel.sources + "/"):
        survivors = []
        result["links_skipped"] = "a filed source places no links"
    elif (fm.get("origin") or "") == "derived":
        known = _known_keys(idx)
        dropped = [c for c in survivors if normalize(c) not in known]
        survivors = [c for c in survivors if normalize(c) in known]
        result["not_discovered"] = len(dropped)
    new_body, report = link_body(body, survivors, _resolution_index(idx))
    fm["summary"] = result["summary"]
    fm["summary_provisional"] = "0"
    fm["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        f.write_text(render_page(fm, new_body), encoding="utf-8")
    except OSError as e:
        # An unwritable or vanished page ends THIS page, not the pass.
        result.update(status="failed", note=f"could not write: {e.strerror}",
                      environmental=True)
        return result

    after = page_hash(f.read_text(encoding="utf-8"))
    if after != before_hash:
        # Should be impossible: links are stripped before hashing and
        # frontmatter is excluded. If it ever fires, the hash rule has been
        # broken somewhere and every outstanding `expect` just went stale.
        result["status"] = "failed"
        result["note"] = "content hash moved during analysis"
        _record(idx, path, before_hash, "failed", result["note"])
        return result

    idx._replace_page(parse_page(path, f.read_text(encoding="utf-8")),
                      f.stat().st_mtime)
    idx.db.execute("UPDATE files SET analyzed_hash=? WHERE path=?",
                   (before_hash, path))
    idx.refresh_resolved()
    _record(idx, path, before_hash, result["status"], result["note"])
    result["links"] = len(report.resolved) + len(report.unresolved)
    return result


def _known_keys(idx) -> set:
    """Concept keys the graph already has, from pages and from links placed on
    pages that are NOT derived. A derived page may point at these; it may not
    add to them."""
    keys = {r["page_key"] for r in idx.db.execute(
        "SELECT page_key FROM files")}
    keys |= {r["target_key"] for r in idx.db.execute(
        """SELECT DISTINCT l.target_key FROM links l JOIN files f
             ON f.path = l.source
           WHERE COALESCE(f.origin, '') != 'derived'""")}
    return keys


def _resolution_index(idx) -> dict[str, str]:
    """Built from FILENAMES only. Reading every page's frontmatter for type
    precedence was O(n) requests and bought almost nothing."""
    out = {}
    for r in idx.db.execute("SELECT path, page_key, type FROM files"):
        name = r["path"].rsplit("/", 1)[-1][:-3]
        # `concept` wins precedence: a concept candidate should never resolve
        # onto something else.
        if r["page_key"] not in out or r["type"] == "concept":
            out[r["page_key"]] = name
    for r in idx.db.execute("SELECT alias_key, canonical_key FROM aliases"):
        if r["canonical_key"] in out and r["alias_key"] not in out:
            out[r["alias_key"]] = out[r["canonical_key"]]
    return out


def _record(idx, path: str, content_hash: str, status: str, note: str,
            environmental: bool = False) -> None:
    """`environmental` failures do NOT leave the queue.

    A gate failure is about this content and will not change until the content
    does, so recording it against the hash is right. A permission error is
    about the filesystem: fixing it does not touch the page, so marking the
    hash as attempted would hide the page forever.
    """
    idx.db.execute("""CREATE TABLE IF NOT EXISTS analysis (
                        path TEXT PRIMARY KEY, content_hash TEXT, status TEXT,
                        note TEXT, at TEXT)""")
    idx.db.execute("INSERT OR REPLACE INTO analysis VALUES (?,?,?,?,?)",
                   (path, content_hash, status, note,
                    datetime.now(timezone.utc).isoformat(timespec="seconds")))
    if status == "failed" and not environmental:
        # Mark the attempt so the page leaves the queue, while the analysis
        # table keeps WHY visible.
        idx.db.execute("UPDATE files SET analyzed_hash=? WHERE path=?",
                       (content_hash, path))
    idx.db.commit()


def run_pass(sentinel, model, limit: int | None = None, verbose: bool = True,
             release: bool = True, summary_model=None):
    """Load once, one call per page, unload.

    Batching is at the PASS level, not the call level: combining pages into
    one call saves nothing, because different texts all have to enter the
    context regardless. What is expensive is the VRAM swap. One page per call
    also keeps the gates and the retry per-page.
    """
    pending = sentinel.index.pending_analysis()
    if limit:
        pending = pending[:limit]
    results = []
    for i, path in enumerate(pending, 1):
        try:
            r = analyze_page(sentinel, path, model,
                             summary_model=summary_model)
        except Exception as e:
            # One page must never end the pass. Fifty pages is hours of work;
            # losing it to the forty-ninth is not an acceptable failure mode.
            r = {"path": path, "status": "failed", "concepts": 0,
                 "seconds": 0.0, "timings": {}, "repairs": [],
                 "note": f"{type(e).__name__}: {e}", "environmental": True}
        results.append(r)
        if verbose:
            t = r["timings"]
            detail = (f"load {t.get('load', 0):.1f}s prefill "
                      f"{t.get('prefill', 0):.1f}s gen {t.get('generate', 0):.1f}s"
                      if t else f"{r['seconds']:.1f}s")
            print(f"  [{i}/{len(pending)}] {path}: {r['status']} "
                  f"{r['concepts']} concepts, {detail}"
                  + (f" (repaired: {', '.join(r['repairs'])})" if r["repairs"] else "")
                  + (f" - {r['note']}" if r["note"] else ""))
    if release:
        _release(model, verbose)
        if summary_model is not None:
            _release(summary_model, verbose=False)
    return results


def _release(model, verbose: bool = True) -> None:
    """Let the analysis model go when the pass is done.

    Two reasons, and the second is the one that was measured. It frees the
    card for the conversation model, which cannot share it. And host memory
    accumulates across a long run of requests - the machine sat at 11 GB of 14
    with no swap after one pass, and unloading returned it.
    """
    if hasattr(model, "unload") and model.unload() and verbose:
        print("  model released")


# ---------------------------------------------------------------------------
# Long text: extract concepts window by window
# ---------------------------------------------------------------------------

CONCEPT_PROMPT = """Below is a record of a conversation. Return a JSON object
with one field.

"concepts": the technical names that were talked about, copied exactly as they
appear in the text. Names of things, methods, systems, parameters, components,
measurements by name.

A conversation names things differently from a paper. A name is usually inside
a sentence about doing something - "we turned off X", "Y worked better than
Z", "the Q dropped" - and the thing being named is X, Y, Z, Q. Take those.

Take everything that was named, not only what was explained. A name mentioned
once still counts.

Leave out: greetings, questions, feelings, and the assistant's descriptions of
its own tools. Leave out whole sentences and phrases - a name is a few words.
Some names are wrapped in [[double brackets]]; give the name without them.

Return only the JSON object. No explanation, no markdown fences, no preamble.

---
{text}
"""

OVERLAP = 0.10   # of the window


def windows(text: str, size: int, overlap: float = OVERLAP):
    """Overlapping slices, so a name split by a boundary is whole in the next.

    A tenth is enough and costs almost nothing: what a boundary can cut is a
    concept name, which is a few words. Sliding one character at a time would
    catch the same names and re-read the text a hundred times over.
    """
    if size <= 0 or len(text) <= size:
        return [text]
    step = max(int(size * (1 - overlap)), 1)
    return [text[i:i + size] for i in range(0, len(text), step)
            if text[i:i + size].strip()]


def extract_concepts(model, text: str) -> tuple[list[str], dict]:
    """Concepts from a text of any length. One call per window, no shared
    context between them.

    NO SUMMARY here. A summary needs the whole text at once, and that is
    exactly what cannot be done for something too long to fit - so this
    extracts only what a window can honestly produce. The summary for a
    conversation page comes from the page itself.
    """
    ctx = getattr(model, "num_ctx", 8192)
    size = (ctx - PROMPT_OVERHEAD - OUTPUT_RESERVE) * CHARS_PER_TOKEN
    parts = windows(text, size)
    seen, out = set(), []
    stats = {"windows": len(parts), "calls": 0, "repairs": [], "failed": 0}

    for part in parts:
        raw, _ = model.complete(CONCEPT_PROMPT.format(text=part))
        stats["calls"] += 1
        obj, repairs = repair_json(raw)
        stats["repairs"] += repairs
        if obj is None:
            # One bad window is not a bad pass. The others still carry.
            stats["failed"] += 1
            continue
        for c in obj.get("concepts") or []:
            c = strip_links(str(c).strip()).strip()
            key = normalize(c)
            if key and key not in seen and shape_ok(c):
                seen.add(key)
                out.append(c)
    return out, stats


def _analyze_windowed(sentinel, path, f, fm, body, before_hash,
                      model, summary_model=None) -> dict:
    idx = sentinel.index

    # THE WINDOWED PATH IS THE ONE THAT HAD NO SUMMARY. A summary needs the
    # whole text at once, which is exactly what this path cannot give, so the
    # page kept whatever `write` had put there - for a transcript, the first
    # line of the conversation. Real vaults listed records as `who are you?`.
    #
    # A second model with a much larger context can read the whole thing in
    # one call, so when one is supplied the summary is written first and the
    # windowed extraction below only has to produce concepts.
    summarised = None
    if summary_model is not None:
        summarised = summarise_page(sentinel, path, summary_model)
        if summarised["status"] == "ok":
            fm, body = split_frontmatter(f.read_text(encoding="utf-8"))
    result = {"path": path, "repairs": [], "status": "ok", "note": "",
              "concepts": 0, "seconds": 0.0, "timings": {}}

    t0 = time.perf_counter()
    concepts, stats = extract_concepts(model, body)
    result["seconds"] = time.perf_counter() - t0
    result["repairs"] = stats["repairs"]
    result.update(windows=stats["windows"], calls=stats["calls"],
                  failed_windows=stats["failed"])

    if stats["failed"] == stats["windows"]:
        result.update(status="failed", note="every window failed to parse")
        _record(idx, path, before_hash, "failed", result["note"])
        return result

    survivors, low = gate_concepts(concepts, body)
    result["concepts"] = len(survivors)
    result["survivors"] = survivors
    if low:
        result["note"] = "below the concept floor"
        result["status"] = "low_yield"

    quiet = (fm.get("type") == "transcript"
             or (sentinel.sources
                 and path.startswith(sentinel.sources + "/")))
    if quiet:
        # Read, gated, counted - and not written into the text. Raw material
        # is what was read, not what was decided about it; the graph is built
        # from the pages the user chose to keep. See the authority note in
        # `analyze_page`.
        survivors = []
        result["links_skipped"] = ("a transcript places no links"
                                   if fm.get("type") == "transcript"
                                   else "a filed source places no links")
    new_body, report = link_body(body, survivors, _resolution_index(idx))
    fm["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        f.write_text(render_page(fm, new_body), encoding="utf-8")
    except OSError as e:
        result.update(status="failed", note=f"could not write: {e.strerror}",
                      environmental=True)
        return result

    if page_hash(f.read_text(encoding="utf-8")) != before_hash:
        result.update(status="failed",
                      note="content hash moved during analysis")
        _record(idx, path, before_hash, "failed", result["note"])
        return result

    idx._replace_page(parse_page(path, f.read_text(encoding="utf-8")),
                      f.stat().st_mtime)
    idx.db.execute("UPDATE files SET analyzed_hash=? WHERE path=?",
                   (before_hash, path))
    idx.refresh_resolved()
    _record(idx, path, before_hash, result["status"], result["note"])
    result["links"] = len(report.resolved) + len(report.unresolved)
    if summarised is not None:
        result["summarised"] = summarised["status"]
        if summarised["note"]:
            result["note"] = ((result["note"] + "; ") if result["note"] else "")
            result["note"] += summarised["note"]
    return result
