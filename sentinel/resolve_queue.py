"""The Sentinel - name resolution, the part code must not decide.

Three mechanisms decide and everything else DEFERS. What defers has to go
somewhere, and until now it went nowhere: the design was written but the
schema held only `aliases`, so the pair itself, whether it had been asked, and
what came back were all unrecorded. Without that, a maintenance pass
regenerates the same pairs every run and an `unclear` accumulates nothing.

A QUEUE ENTRY IS TWO PARAGRAPHS, NOT TWO STRINGS. Names do not carry enough -
the measured inversion proves it, `version1`~`version2` scoring 87.5 while
`Ariel`~`Ariel Space Telescope` scores 38.5. The `links` table records the
source page of every mention, so the context is retrievable, and handing over
the paragraphs makes resolution a TRANSPORT task rather than a guess.

MAINTENANCE ONLY. One entry is ~151 tokens in and one out; fifty pairs would
be ~7500 for a single call, so it is batched at ~10 and never runs on a
conversational turn. No primitive calls anything in this file.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .embed import from_blob, to_blob
from .names import DEFER, SAME, initials, resolve_pair
from .text import content_hash, normalize, shape_ok, tokens

# A shared token counts as evidence only if it is rare. The floor keeps the
# rule usable on a small vault; the share keeps it from readmitting common
# words as the vault grows - at 140 names the cap is 3, at 1000 it is 20.
DF_FLOOR = 3
DF_SHARE = 0.02

SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
    pair_key TEXT PRIMARY KEY,        -- the two keys, sorted, joined by |
    key_a TEXT NOT NULL,
    key_b TEXT NOT NULL,
    display_a TEXT,
    display_b TEXT,
    source TEXT NOT NULL,             -- structure | proximity
    status TEXT NOT NULL,             -- open | same | different | unclear
    weight INTEGER DEFAULT 0,         -- mentions across both sides, at insert
    mentions_at_ask INTEGER DEFAULT 0,
    asked_at TEXT,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_queue_status ON queue(status);
"""

PROMPT = """Below are numbered pairs. Each pair shows two names as they appear
in a set of notes, with a paragraph where each one is used.

For every pair, decide whether the two names refer to the SAME thing.

Answer "same" only if they are the same thing under two names.
Answer "different" if they are related but not the same - a subclass, a
specific instance, a different version.
Answer "unclear" if the paragraphs do not settle it. "unclear" is a real
answer and is better than a guess.

Return only a JSON array, one object per pair:
[{"n": 1, "answer": "same"}, {"n": 2, "answer": "unclear"}]

No explanation, no markdown fences, no preamble.

{pairs}
"""


def ensure_schema(index) -> None:
    index.db.executescript(SCHEMA)


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def _candidate_names(index) -> tuple[dict[str, str], set[str]]:
    """Names eligible to be merged, and which of them are unresolved.

    A candidate must be CONCEPT-SHAPED. The shape filter already exists and
    is measured at 18/18 noise rejected and 19/19 concepts kept; running it
    here costs nothing and closes a real hole.

    Measured, first real run of the queue: the model was asked whether
    `Constraining the initial Lorentz factor of gamma-ray bursts` and
    `Gamma-Ray Burst` are the same thing, and answered `same`. That is a
    wrong merge - a paper title against a concept - and wrong merges are the
    one outcome this design says must never happen. It also contradicted
    itself, calling the same title `different` from `initial Lorentz factor`
    one line earlier.

    The model was not the problem. A nine-word title should never have been
    offered as a merge candidate.
    """
    names, seeds = {}, set()
    for r in index.db.execute(
            "SELECT target_key, MIN(display) d, MIN(resolved) res FROM links "
            "GROUP BY target_key"):
        if shape_ok(r["d"]):
            names[r["target_key"]] = r["d"]
            if not r["res"]:
                seeds.add(r["target_key"])
    for r in index.db.execute("SELECT page_key, path FROM files"):
        stem = r["path"].rsplit("/", 1)[-1][:-3]
        if shape_ok(stem):
            names.setdefault(r["page_key"], stem)
    return names, seeds


def _mentions(index, *keys) -> int:
    q = ",".join("?" * len(keys))
    return index.db.execute(
        f"SELECT COUNT(*) c FROM links WHERE target_key IN ({q})", keys
    ).fetchone()["c"]


# ---------------------------------------------------------------------------
# Candidate generation - TWO sources, not one
# ---------------------------------------------------------------------------


def _structural_candidates(index) -> list[tuple[str, str, str, str]]:
    """Names that SHARE STRUCTURE - one contains the other, or they share a
    token - and that the resolution chain then could not decide.

    The two steps are not the same thing and conflating them is a trap worth
    naming. `resolve_pair` is a DECIDER, not a generator: its `defer` means
    "none of the three mechanisms spoke", which is equally true of two names
    with nothing whatever to do with each other. Used as a generator it
    produces the full cross product - measured, 118 pairs from 9 pages,
    including `Ariel Space Telescope` against `forward shock`. At 300 pages
    that is tens of thousands of entries and the queue is unusable.

    Field test 1 already recorded the right shape: of 91 comparisons, ONE was
    queued and 88 were correctly untouched. Untouched is not deferred.

    This test is deliberately narrow, and that narrowness is why there is a
    second source below - `FS model` and `forward shock` are the same thing
    and nothing here can see it: no containment, no shared token, and the
    acronym rule fails because `FS model` is not an acronym, it merely
    contains one.
    """
    names, _ = _candidate_names(index)

    toks = {k: set(tokens(v)) for k, v in names.items()}

    # A token shared by many names says nothing. Measured on the real vault:
    # 140 names produced 28 containment pairs and 204 shared-token pairs, and
    # the tokens driving that were `md`, `arxiv`, `2504`, `11743`, `and`, `of`
    # - filename debris from split source pages, not concepts.
    #
    # Document frequency rather than a stop-word list, because the signal is
    # already in the data and a word list would be language-specific in a
    # bilingual vault. A purely numeric token is dropped outright: two papers'
    # section 1 share the character "1" and nothing else. Digits carry weight
    # in the designation rule, but there they mark a DIFFERENCE.
    df: dict[str, int] = {}
    for ts in toks.values():
        for t in ts:
            df[t] = df.get(t, 0) + 1
    cap = max(DF_FLOOR, int(DF_SHARE * len(names)))

    out, keys = [], sorted(names)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a in b or b in a:
                shares = True                    # containment always counts
            else:
                shares = any(not t.isdigit() and df[t] <= cap
                             for t in toks[a] & toks[b])
            if shares and resolve_pair(names[a], names[b]) == DEFER:
                out.append((a, b, names[a], names[b]))
    return out


def _name_vectors(index, names: dict[str, str], encoder):
    """Embed the NAMES, cached in the vectors table by content hash.

    A name is a few words, so 140 of them cost a fraction of what the chunks
    did, and caching means a rebuild re-encodes only what is new.
    """
    import numpy as np

    todo, out = [], {}
    for key, display in names.items():
        h = content_hash(display)
        row = index.db.execute(
            "SELECT vec FROM vectors WHERE stripped_hash=?", (h,)).fetchone()
        if row:
            v = np.array(from_blob(row["vec"]), dtype=np.float32)
            n = np.linalg.norm(v)
            if n:
                out[key] = v / n
        else:
            todo.append((key, display, h))

    for i in range(0, len(todo), 32):
        batch = todo[i:i + 32]
        vecs = encoder.encode([d for _, d, _ in batch], kind="passage")
        index.db.executemany(
            "INSERT OR REPLACE INTO vectors (stripped_hash, vec) VALUES (?,?)",
            [(h, to_blob(v)) for (_, _, h), v in zip(batch, vecs)])
        for (key, _, _), v in zip(batch, vecs):
            v = np.asarray(v, dtype=np.float32)
            n = np.linalg.norm(v)
            if n:
                out[key] = v / n
    if todo:
        index.db.commit()
    return out


def _proximity_candidates(index, encoder, k: int = 3
                          ) -> list[tuple[str, str, str, str]]:
    """Nearest neighbours by the embedding of the NAME.

    The recorded design said mention context - the paragraphs a name appears
    in - and that was measured and found dead. On the real vault every context
    similarity came back at 1.000, because concepts in a single-topic vault
    share the same chunks, so their context vectors are not merely close but
    IDENTICAL. What it surfaced was `Ariel Space Telescope` against `Lagrange
    2 point`: two things that share a paper and nothing else. Removing the
    vault centroid did not rescue it.

    The name alone, on the same vault, put these at the top:
        radiation efficiency  ~ radiative efficiency        0.990
        initial Lorentz factor ~ Lorentz factor             0.966
        peak energy           ~ peak spectral energy        0.956
        coasting period       ~ coasting phase              0.955
    Every one of them is a pair a person has to judge. Name and context
    combined scored slightly worse than the name by itself, which is the
    clearest statement that context was adding noise rather than signal.

    The recorded objection to names - `version1`~`version2` scoring 87.5 while
    `Ariel`~`Ariel Space Telescope` scored 38.5 - was an argument against
    similarity as a DECIDER, and it stands as one. This is a GENERATOR, with
    the decider still in front of it: `version1`/`version2` is settled by the
    designation rule and never reaches the queue. A false positive here costs
    one queue entry.

    What does NOT change is the queue entry itself. It still carries two
    PARAGRAPHS, because deciding still needs context even though generating
    does not.
    """
    if encoder is None:
        return []
    import numpy as np

    names, seeds = _candidate_names(index)
    if not seeds:
        return []

    vectors = _name_vectors(index, names, encoder)
    keys = sorted(vectors)
    if len(keys) < 2:
        return []
    mat = np.array([vectors[x] for x in keys], dtype=np.float32)
    pos = {x: i for i, x in enumerate(keys)}

    # MUTUAL nearest neighbours: each side must have the other in its own
    # top-k. One-sided top-k forces every seed to contribute k pairs whether
    # or not it has a plausible partner - measured on the real vault, that put
    # `blackbody components` against `GRB` and `burst duration` against
    # `circumburst medium`, because those seeds have no synonym and their
    # nearest neighbour is merely the least distant thing in the room.
    #
    # This is not the threshold the design rules out. It adds no number to
    # tune; it asks the ranking a question it can already answer, and the
    # asymmetry is exactly what separates "these two are alike" from "this one
    # had to point somewhere".
    top = {}
    for x in keys:
        sims = mat @ vectors[x]
        sims[pos[x]] = -2.0
        top[x] = {keys[j] for j in np.argsort(-sims)[:k]}

    out = set()
    for seed in sorted(seeds & vectors.keys()):
        for other in top[seed]:
            if seed in top[other]:
                a, b = sorted((seed, other))
                out.add((a, b, names[a], names[b]))
    return sorted(out)


def build_queue(index, encoder=None, k: int = 3) -> dict:
    """Insert new candidates; leave decided pairs alone.

    A pair answered `same` or `different` is FINAL and never re-asked - a
    wrong merge is irreversible, so a settled decision does not get
    re-litigated. A pair answered `unclear` reopens only when a NEW mention
    arrives, which is the one thing that could change the answer. Without that
    rule an `unclear` comes back every single run and costs its tokens again
    for no new evidence.
    """
    ensure_schema(index)
    stats = {"structure": 0, "proximity": 0, "reopened": 0, "skipped": 0}
    aliased = {r["alias_key"] for r in index.db.execute(
        "SELECT alias_key FROM aliases")}

    candidates = [(*c, "structure") for c in _structural_candidates(index)] + \
                 [(*c, "proximity") for c in
                  _proximity_candidates(index, encoder, k)]

    # The decider runs on candidates from EVERY source, not just the
    # structural one. Proximity does not consult it while generating, so
    # without this a pair the acronym rule already settles - `GRB` against
    # `gamma-ray bursts` - would be put to the model anyway: tokens spent on
    # something code knows, and a chance for a wrong answer to override a
    # correct rule.
    candidates = [c for c in candidates
                  if resolve_pair(c[2], c[3]) == DEFER]

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for a, b, da, db, src in candidates:
        if a in aliased or b in aliased:
            stats["skipped"] += 1
            continue
        pk = _pair_key(a, b)
        row = index.db.execute(
            "SELECT status, mentions_at_ask FROM queue WHERE pair_key=?",
            (pk,)).fetchone()
        if row is None:
            index.db.execute(
                "INSERT INTO queue (pair_key, key_a, key_b, display_a, "
                "display_b, source, status, weight) "
                "VALUES (?,?,?,?,?,?,'open',?)",
                (pk, a, b, da, db, src, _mentions(index, a, b)))
            stats[src] += 1
        elif row["status"] == "unclear" and \
                _mentions(index, a, b) > row["mentions_at_ask"]:
            index.db.execute(
                "UPDATE queue SET status='open' WHERE pair_key=?", (pk,))
            stats["reopened"] += 1
        else:
            stats["skipped"] += 1
    index.db.commit()
    return stats


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def _paragraph_for(index, key: str, display: str) -> str:
    row = index.db.execute(
        """SELECT c.content FROM chunks c
           WHERE c.path IN (SELECT source FROM links WHERE target_key=?)
             AND c.is_summary=0 AND c.content LIKE ?
           ORDER BY LENGTH(c.content) LIMIT 1""",
        (key, f"%{display}%")).fetchone()
    if not row:
        return f"(no paragraph on file for '{display}')"
    # Measured on the first real batch: 600 characters a side put ten pairs
    # at 2966 tokens, about 297 each, against a recorded figure of ~151. The
    # answers were good at that size, so the question is what can be cut
    # without losing them - one paragraph of 350 characters still shows how
    # the name is used, which is the whole job.
    text = " ".join(row["content"].split())
    return text[:350]


def next_batch(index, limit: int = 10) -> list[dict]:
    ensure_schema(index)
    # Heaviest first. The queue is a backlog, not a work list to finish: at
    # scale, proximity alone yields K entries per unresolved target, so what
    # decides its value is that the pairs worth answering are answered first.
    rows = list(index.db.execute(
        "SELECT * FROM queue WHERE status='open' "
        "ORDER BY weight DESC, source, pair_key LIMIT ?", (limit,)))
    return [{"pair_key": r["pair_key"], "a": r["key_a"], "b": r["key_b"],
             "display_a": r["display_a"], "display_b": r["display_b"],
             "para_a": _paragraph_for(index, r["key_a"], r["display_a"]),
             "para_b": _paragraph_for(index, r["key_b"], r["display_b"])}
            for r in rows]


def format_batch(batch: list[dict]) -> str:
    """Numbered, so the answer is an index rather than a copied string - the
    same reason choices are offered by number elsewhere. It closes the surface
    where a model invents a name that was never on the list."""
    parts = []
    for i, e in enumerate(batch, 1):
        parts.append(
            f"{i}.\n"
            f"A: {e['display_a']}\n   {e['para_a']}\n"
            f"B: {e['display_b']}\n   {e['para_b']}")
    return PROMPT.replace("{pairs}", "\n\n".join(parts))


def apply_answers(index, batch: list[dict], answers: list[dict]) -> dict:
    """An accepted `same` writes an ALIAS entry, and every link already
    written with either spelling resolves from then on - no rewrite pass, no
    file touched."""
    ensure_schema(index)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats = {"same": 0, "different": 0, "unclear": 0, "ignored": 0}
    by_n = {int(a.get("n", 0)): str(a.get("answer", "")).lower()
            for a in answers if isinstance(a, dict)}

    for i, e in enumerate(batch, 1):
        ans = by_n.get(i)
        if ans not in ("same", "different", "unclear"):
            stats["ignored"] += 1
            continue
        stats[ans] += 1
        index.db.execute(
            "UPDATE queue SET status=?, decided_at=?, asked_at=?, "
            "mentions_at_ask=? WHERE pair_key=?",
            (ans, now, now, _mentions(index, e["a"], e["b"]), e["pair_key"]))
        if ans == "same":
            # Whichever side has a page is canonical; if neither does, the one
            # with more mentions. Deterministic either way, and if a page is
            # created later the alias already points the right direction.
            pages = {r["page_key"] for r in index.db.execute(
                "SELECT page_key FROM files")}
            if e["a"] in pages and e["b"] not in pages:
                alias, canon = e["b"], e["a"]
            elif e["b"] in pages and e["a"] not in pages:
                alias, canon = e["a"], e["b"]
            else:
                alias, canon = sorted(
                    (e["a"], e["b"]), key=lambda k: _mentions(index, k))[:2]
            index.db.execute(
                "INSERT OR REPLACE INTO aliases (alias_key, canonical_key, "
                "kind, added) VALUES (?,?,'judged',?)", (alias, canon, now))
    index.refresh_resolved()
    index.db.commit()
    return stats


def _ask(model, batch: list[dict]):
    from .analysis import repair_json
    text, _ = model.complete(format_batch(batch))
    obj, _ = repair_json(_wrap(text))
    answers = obj.get("answers") if isinstance(obj, dict) else None
    if answers is None:
        return None
    return {int(a.get("n", 0)): str(a.get("answer", "")).lower()
            for a in answers if isinstance(a, dict)}


def run_resolution(index, model, batch_size: int = 10, rounds: int = 1,
                   confirm: bool = True, release: bool = True) -> dict:
    """Maintenance pass. Never call this from a conversational turn.

    A `same` answer is CONFIRMED before it is applied; `different` and
    `unclear` are not. The asymmetry is the design's own principle applied to
    the model rather than to the code: a wrong merge destroys information
    irreversibly, a missed one costs a single unresolved link that a later
    alias repairs. The cost runs one way, so the doubt should too.

    It is here because it was measured, not anticipated. On the first two real
    batches the model produced two wrong merges out of twenty, and both
    contradicted its own answers - it called `acceleration phase` and
    `deceleration phase` different, then `coasting phase` and `deceleration
    phase` the same in the SAME batch; and it called `isotropic energy` and
    `isotropic luminosity` different in one batch and the same in the next.
    The instability is real, so the premise that this is pure transport does
    not hold for the merge direction.

    The confirmation asks again with the two sides SWAPPED, so agreement means
    the pair survived a differently-shaped question rather than a repeated
    one. Disagreement is recorded as `unclear`, which leaves the pair queued at
    no cost - exactly what the third answer is for.
    """
    totals = {"same": 0, "different": 0, "unclear": 0, "ignored": 0,
              "unparseable": 0, "unconfirmed": 0}
    for _ in range(rounds):
        batch = next_batch(index, batch_size)
        if not batch:
            break
        first = _ask(model, batch)
        if first is None:
            totals["unparseable"] += 1
            continue

        answers = [{"n": i, "answer": a} for i, a in first.items()]

        if confirm:
            merges = [i for i, a in first.items() if a == "same"]
            if merges:
                swapped = [{**batch[i - 1],
                            "display_a": batch[i - 1]["display_b"],
                            "display_b": batch[i - 1]["display_a"],
                            "para_a": batch[i - 1]["para_b"],
                            "para_b": batch[i - 1]["para_a"]}
                           for i in merges]
                second = _ask(model, swapped) or {}
                for pos, i in enumerate(merges, 1):
                    if second.get(pos) != "same":
                        first[i] = "unclear"
                        totals["unconfirmed"] += 1
                answers = [{"n": i, "answer": a} for i, a in first.items()]

        got = apply_answers(index, batch, answers)
        for k, v in got.items():
            totals[k] = totals.get(k, 0) + v
    if release:
        from .analysis import _release
        _release(model, verbose=False)
    return totals


def _wrap(text: str) -> str:
    """repair_json expects an object; the answer is an array. Wrap it rather
    than teaching the repair function a second shape."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = re.sub(r"^```[a-z]*\n?|```$", "", t).strip()
    if "[" in t and "]" in t:
        t = t[t.index("["):t.rindex("]") + 1]
    return json.dumps({"answers": json.loads(t)}) if _parses(t) else text


def _parses(t: str) -> bool:
    try:
        json.loads(t)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The moment a page is created
# ---------------------------------------------------------------------------

def _looks_like_acronym(short: str, long: str) -> bool:
    """Case-INSENSITIVE, unlike the deciding rule.

    The deciding rule requires uppercase, because that is what distinguishes
    an acronym from an ordinary short word and it is the form measured at zero
    wrong merges. But a filename is usually lowercase - `grb.md` - so the
    strict rule cannot see that it collides with `gamma-ray burst`. Loose here
    is safe precisely because nothing is merged on it: it only produces a
    sentence for the user to read.
    """
    s = short.strip().lower()
    if not re.fullmatch(r"[a-z]{2,6}s?", s):
        return False
    return s.rstrip("s") == initials(long).rstrip("s")


def on_page_created(index, path: str) -> dict:
    """Apply the acronym rule at the one moment it can act, and warn about the
    collision it cannot.

    THE FILENAME IS THE CANONICAL NAME - the resolution index is built from
    filenames only. So creating `grb.md` for a concept that forty pages link
    as `[[gamma-ray burst]]` leaves all forty unresolved, forever, and nothing
    in the system would have said a word. The information was always there;
    it was simply never spoken.

    Two different strengths, deliberately:
      merged - the deciding rule says SAME. An alias is written and every
               earlier link comes alive, no file rewritten.
      near   - code must NOT decide. Report it and let the user rename or
               leave it; a wrong merge is irreversible and a missed one costs
               one alias entry later.
    """
    ensure_schema(index)
    row = index.db.execute("SELECT page_key FROM files WHERE path=?",
                           (path,)).fetchone()
    if not row:
        return {"merged": [], "near": []}
    new_key = row["page_key"]
    new_name = path.rsplit("/", 1)[-1][:-3]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    merged, near = [], []
    for r in index.db.execute(
            "SELECT target_key, MIN(display) d, COUNT(*) n FROM links "
            "WHERE resolved=0 GROUP BY target_key ORDER BY n DESC"):
        key, display, n = r["target_key"], r["d"], r["n"]
        if key == new_key:
            continue
        if index.db.execute("SELECT 1 FROM aliases WHERE alias_key=?",
                            (key,)).fetchone():
            continue
        verdict = resolve_pair(new_name, display)
        if verdict == SAME:
            index.db.execute(
                "INSERT OR REPLACE INTO aliases (alias_key, canonical_key, "
                "kind, added) VALUES (?,?,'acronym',?)", (key, new_key, now))
            merged.append((display, n))
        elif verdict == DEFER and (
                _looks_like_acronym(new_name, display)
                or _looks_like_acronym(display, new_name)
                or (set(tokens(new_name)) & set(tokens(display)))):
            near.append((display, n))
    if merged:
        index.refresh_resolved()
    index.db.commit()
    return {"merged": merged, "near": near[:3]}


def revise(index, pair_key: str, answer: str) -> str:
    """Correct a decision. The only way back from a wrong merge.

    The design's rule is that CODE must never merge wrongly, and it does not:
    the three mechanisms are measured at zero wrong merges. But the queue
    exists precisely for the pairs code will not decide, and a model can be
    wrong about those - measured, on the first real batch, once.

    "Irreversible" was true of the alias table alone. With the queue recording
    which pair produced which alias, a wrong answer is reversible, and saying
    so is more honest than pretending the model will not err.
    """
    ensure_schema(index)
    if answer not in ("same", "different", "unclear", "open"):
        return '[RETRY] answer must be same, different, unclear or open'
    row = index.db.execute("SELECT * FROM queue WHERE pair_key=?",
                           (pair_key,)).fetchone()
    if not row:
        return f"[STOP] no queued pair {pair_key}"

    removed = index.db.execute(
        "DELETE FROM aliases WHERE (alias_key=? AND canonical_key=?) "
        "OR (alias_key=? AND canonical_key=?)",
        (row["key_a"], row["key_b"], row["key_b"], row["key_a"])).rowcount
    index.db.execute("UPDATE queue SET status=? WHERE pair_key=?",
                     (answer, pair_key))
    if answer == "same":
        index.db.execute(
            "INSERT OR REPLACE INTO aliases (alias_key, canonical_key, kind, "
            "added) VALUES (?,?,'user',?)",
            (row["key_b"], row["key_a"],
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
    index.refresh_resolved()
    index.db.commit()
    return (f"[DONE] {row['display_a']} ? {row['display_b']} -> {answer}"
            + (f", {removed} alias removed" if removed else ""))


def decided(index, status: str | None = None):
    """What has been answered, so it can be reviewed. A merge nobody can see
    is a merge nobody can correct."""
    ensure_schema(index)
    q = "SELECT * FROM queue WHERE status != 'open'"
    args: tuple = ()
    if status:
        q += " AND status = ?"
        args = (status,)
    return list(index.db.execute(q + " ORDER BY status, weight DESC", args))
