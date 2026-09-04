"""The Sentinel - writing concept pages, and growing them.

The growth queue says which concepts are referenced and have no page. This
turns the well-supplied ones into pages, and adds to them as new material
arrives.

WHAT MAY BE WRITTEN. Only what the sources define. A concept mentioned in
passing has no page written for it, however often it is mentioned - measured
on the real vault, 121 unresolved names had enough surrounding text but only
about a third carried a sentence that said what the thing IS, and reading
those by hand cut it further. `wireless power transfer` sits in eleven
thousand characters and is never defined once.

That restraint is the design, not a limitation of it. The closest thing to
this system run at scale is Cebuano Wikipedia, where one bot wrote 99% of the
articles: the facts were correct, because they came from databases rather than
from a model, and the outcome was still a project its own community proposed
closing. What went wrong was volume of low-information pages - stubs that
discouraged people from writing, and near-duplicates that a deduplication pass
removes wholesale. Correct and worthless is a reachable state.

So the queue stays a reading list where the sources have nothing to say. That
is a better thing to have than a page.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from .analysis import (
    CHARS_PER_TOKEN,
    _release,
    OUTPUT_RESERVE,
    PROMPT_OVERHEAD,
    _record,
    repair_json,
    windows,
)
from .index import parse_page
from .text import normalize, render_page, shape_ok, split_frontmatter, strip_links

# A page with more sections than this is not wrong, but it has stopped being
# one thing. Reported, never acted on: splitting a page is a judgement about
# what the subject IS, and that is not a decision to automate.
BLOAT_SECTIONS = 12

# Defining sentences a concept needs before a page is written from it.
MIN_SENTENCES = 2

DEFINE_PROMPT = """Below is text that mentions "{name}".

Does this text SAY WHAT "{name}" IS? Not whether it uses the term, or
describes something done with it - whether a reader who did not know the term
would learn from this text what it means.

Return a JSON object with two fields.

"answer": "yes", "no", or "unclear".
"sentences": if yes, the sentences from the text that say what it is, copied
exactly. Otherwise an empty list.

"unclear" is a real answer. Prefer it to a guess.

Return only the JSON object. No explanation, no markdown fences, no preamble.

---
{text}
"""

WRITE_PROMPT = """Below are sentences about "{name}", taken from sources.

Join them into a short passage. Fix the joins so it reads as continuous prose:
reorder, merge, drop a repetition, replace a pronoun with the name. Nothing
else.

DO NOT ADD A SENTENCE. Not an elaboration, not a consequence, not a closing
line. If the material is one sentence long then the passage is one sentence
long, and that is the correct result - it is what the sources say.

Measured on a real run: from one sourced sentence the model produced a second
saying the subject stood out for its "immense power and destructive nature".
No source said that. It was written to fill the shape of a page.

Return only the passage.

---
{material}
"""

SECTION_PROMPT = """Below is new material about "{name}", from sources that
were not used when its page was written.

Write ONE short section to add to that page. Use only what this material says.
Begin with a `## ` heading naming what the section covers.

If the material adds nothing that is not already in the existing page below,
return exactly: SKIP

Return only the section, or SKIP.

--- existing page ---
{existing}

--- new material ---
{material}
"""


# ---------------------------------------------------------------------------
# Candidates and their material
# ---------------------------------------------------------------------------


def candidates(index, limit: int = 20) -> list[dict]:
    """Unresolved, concept-shaped names, most referenced first."""
    out = []
    for r in index.db.execute(
            """SELECT target_key, MIN(display) d, COUNT(*) n FROM links
               WHERE resolved = 0 GROUP BY target_key ORDER BY n DESC"""):
        if shape_ok(r["d"]):
            out.append({"key": r["target_key"], "display": r["d"],
                        "mentions": r["n"]})
        if len(out) >= limit:
            break
    return out


def material(index, key: str, display: str, exclude: set | None = None
             ) -> list[tuple[str, str]]:
    """[(source page, paragraph)] for a concept, from NON-DERIVED pages only.

    A concept page is written from sources. Letting it be written from other
    concept pages is the recursion this system refuses: the corpus would fill
    with text generated from text generated from text, and the measured shape
    of that is facts drifting while the prose stays fluent.
    """
    rows = index.db.execute(
        """SELECT c.path, c.content FROM chunks c
           JOIN files f ON f.path = c.path
           WHERE c.is_summary = 0
             AND COALESCE(f.origin, '') != 'derived'
             AND c.path IN (SELECT source FROM links WHERE target_key = ?)
             AND c.content LIKE ?""", (key, f"%{display}%"))
    return [(r["path"], strip_links(r["content"])) for r in rows
            if not exclude or r["path"] not in exclude]


def _defining(model, name: str, chunks: list[tuple[str, str]]) -> dict:
    """Ask, window by window, whether the material defines the name.

    The question goes to the model rather than to a pattern. A keyword test
    was tried and measured: it counted `GRB 221009A is exceptionally bright`
    as a definition, and it matched one sentence listing four instrument names
    as a definition of all four. It reported 39 writable concepts where
    reading them by hand found closer to a dozen.
    """
    ctx = getattr(model, "num_ctx", 8192)
    size = (ctx - PROMPT_OVERHEAD - OUTPUT_RESERVE) * CHARS_PER_TOKEN
    found, sources, answers = [], set(), []

    for path, text in chunks:
        for part in windows(text, size):
            raw, _ = model.complete(
                DEFINE_PROMPT.format(name=name, text=part))
            obj, _ = repair_json(raw)
            if not obj:
                continue
            ans = str(obj.get("answer", "")).lower()
            answers.append(ans)
            if ans != "yes":
                continue
            for s in obj.get("sentences") or []:
                s = " ".join(str(s).split())
                # It must actually be in the text. The same rule as the
                # concept gate, for the same reason.
                if s and " ".join(part.split()).find(s) >= 0:
                    found.append(s)
                    sources.add(path)
    return {"sentences": found, "sources": sources, "answers": answers}


# ---------------------------------------------------------------------------
# Writing and growing
# ---------------------------------------------------------------------------


def write_concept(sentinel, key: str, display: str, model,
                  folder: str = "wiki") -> dict:
    idx = sentinel.index
    result = {"concept": display, "status": "skipped", "note": "",
              "sources": 0, "sentences": 0}

    if idx.resolve(display):
        result["note"] = "a page already resolves from this name"
        return result

    chunks = material(idx, key, display)
    if not chunks:
        result["note"] = "no material outside derived pages"
        return result

    found = _defining(model, display, chunks)
    result["sentences"] = len(found["sentences"])
    result["sources"] = len(found["sources"])
    if len(found["sentences"]) < MIN_SENTENCES:
        # NOT a failure. The sources use the term without saying enough about
        # it, so there is nothing to write and the queue keeps it as a thing
        # to go and read about.
        #
        # The floor exists because one sentence is not a page and the model
        # will not leave it as one - given a single sourced line it wrote a
        # second, unsourced, to make the page look finished. Refusing thin
        # material is cheaper than policing what gets added to it.
        result["note"] = ("mentioned but never defined in the sources"
                          if not found["sentences"]
                          else f"only {len(found['sentences'])} defining "
                               f"sentence in the sources, too thin for a page")
        return result

    text, _ = model.complete(WRITE_PROMPT.format(
        name=display, material="\n\n".join(found["sentences"])))
    body = text.strip()
    if len(body) < 40:
        result.update(status="failed", note="the model returned almost nothing")
        return result

    path = f"{folder.rstrip('/')}/{display}.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fm = {"type": "concept", "origin": "derived",
          "summary": _first_sentence(body), "summary_provisional": "1",
          "sources": ", ".join(sorted(found["sources"])),
          "created": now, "updated": now}
    f = idx.vault / path
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        result.update(status="skipped", note=f"{path} already exists")
        return result
    f.write_text(render_page(fm, f"# {display}\n\n{body}\n"), encoding="utf-8")

    idx.sync()
    from .resolve_queue import on_page_created
    naming = on_page_created(idx, path)
    result.update(status="written", path=path,
                  merged=[d for d, _ in naming["merged"]],
                  near=[d for d, _ in naming["near"]])
    return result


def grow_concept(sentinel, path: str, model) -> dict:
    """Add a section from sources the page was not written from.

    Adds rather than rewrites. A rewrite would lose whatever the user has
    edited into the page and would move its content hash, putting it back in
    the analysis queue every time it grows.
    """
    idx = sentinel.index
    f = idx.vault / path
    result = {"path": path, "status": "skipped", "note": ""}
    if not f.exists():
        result.update(status="failed", note="no such page")
        return result

    fm, body = split_frontmatter(f.read_text(encoding="utf-8"))
    used = {s.strip() for s in (fm.get("sources") or "").split(",") if s.strip()}
    display = path.rsplit("/", 1)[-1][:-3]
    fresh = material(idx, normalize(display), display, exclude=used)
    if not fresh:
        result["note"] = "no material the page has not already seen"
        return result

    found = _defining(model, display, fresh)
    if not found["sentences"]:
        result["note"] = "the new material never says what it is"
        return result

    text, _ = model.complete(SECTION_PROMPT.format(
        name=display, existing=body[:4000],
        material="\n\n".join(found["sentences"])))
    section = text.strip()
    if section.upper().startswith("SKIP") or len(section) < 40:
        result["note"] = "the new material adds nothing"
        # Still record the sources, or every pass re-reads them.
        fm["sources"] = ", ".join(sorted(used | found["sources"]))
        f.write_text(render_page(fm, body), encoding="utf-8")
        idx.sync()
        return result

    fm["sources"] = ", ".join(sorted(used | found["sources"]))
    fm["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    f.write_text(render_page(fm, body.rstrip() + "\n\n" + section + "\n"),
                 encoding="utf-8")
    idx.sync()
    result.update(status="grown", added=len(section),
                  sources=len(found["sources"]))
    return result


def stale_concepts(index) -> list[str]:
    """Concept pages with material they have not read yet."""
    out = []
    for r in index.db.execute(
            "SELECT path FROM files WHERE type='concept' "
            "AND COALESCE(origin,'') = 'derived'"):
        fm, _ = split_frontmatter(
            (index.vault / r["path"]).read_text(encoding="utf-8"))
        used = {s.strip() for s in (fm.get("sources") or "").split(",")
                if s.strip()}
        name = r["path"].rsplit("/", 1)[-1][:-3]
        if material(index, normalize(name), name, exclude=used):
            out.append(r["path"])
    return out


def bloated(index, limit: int = BLOAT_SECTIONS) -> list[tuple[str, int]]:
    """Pages that have grown into more than one subject.

    A count, and nothing else. What to split off, and whether the page should
    be split at all, is a judgement about the subject - exactly the kind this
    system leaves to the person who owns the vault.
    """
    out = []
    for r in index.db.execute(
            "SELECT path FROM files WHERE type='concept'"):
        body = split_frontmatter(
            (index.vault / r["path"]).read_text(encoding="utf-8"))[1]
        n = len(re.findall(r"^## ", body, re.M))
        if n > limit:
            out.append((r["path"], n))
    return out


def _first_sentence(body: str, cap: int = 300) -> str:
    text = " ".join(re.sub(r"^#{1,6}[^\n]*$", "", body, flags=re.M).split())
    m = re.search(r"(?<=[.!?])\s", text)
    return ((text[:m.start()] if m else text)[:cap]).strip() or "(no text)"


def run_concepts(sentinel, model, limit: int = 5, folder: str = "wiki",
                 verbose: bool = True, release: bool = True) -> list[dict]:
    """One maintenance pass: write what can be written, grow what has moved.

    Ordered by how many pages reference each, so what the vault leans on most
    is written first.
    """
    results = []
    for c in candidates(sentinel.index, limit * 4):
        if len([r for r in results if r["status"] == "written"]) >= limit:
            break
        r = write_concept(sentinel, c["key"], c["display"], model, folder)
        r["mentions"] = c["mentions"]
        results.append(r)
        if verbose:
            print(f"  {c['display']}: {r['status']}"
                  + (f" - {r['note']}" if r["note"] else "")
                  + (f", {r['sentences']} defining sentences"
                     if r["sentences"] else ""))

    for path in stale_concepts(sentinel.index):
        r = grow_concept(sentinel, path, model)
        results.append(r)
        if verbose:
            print(f"  {path}: {r['status']}"
                  + (f" - {r['note']}" if r["note"] else ""))
    if release:
        _release(model, verbose)
    return results
