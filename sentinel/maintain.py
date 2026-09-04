"""The Sentinel - the maintenance pass.

Everything the vault owes, in one ordered run, on a timer.

Until now each pass was a separate command typed by hand, which is not a thing
a person keeps doing. The design already said maintenance surfaces itself
through the health line and runs at a natural pause; this is the pause.

    python3 -m sentinel.maintain --vault ~/obsidian/Obsidian-1 \\
        --model qwen3.5-9b-jinja:latest --exclude agent_workspace --dry-run

NOT DURING A CONVERSATION. The analysis model and the conversation model
cannot share a 6 GB card, so this evicts whatever is loaded and takes several
minutes. Run it when nobody is talking to the vault - overnight, on a timer.

THE ORDER IS LOAD-BEARING:

  sync        the index, so everything after it sees the vault as it is
  analysis    summaries and concepts, which is what fills the growth queue
  embeddings  over the chunks analysis just produced
  resolution  merges names BEFORE pages get written for them, so a page is
              never written for a name that was about to become an alias
  concepts    writes and grows pages from what survived
  sync        again, because writing pages resolved links

THE MODEL IS LOADED ONCE. Each pass can release it on its own, which is right
when it runs alone and wrong here: three passes would load and unload three
times, and loading measured 6.4 seconds. It is released once, at the end.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .analysis import _release, run_pass
from .concepts import bloated, candidates, run_concepts, stale_concepts
from .index import Index, default_db
from .models import OllamaModel
from .resolve_queue import build_queue, next_batch, run_resolution
from .tools import Sentinel

LOCK = ".sentinel-maintenance.lock"


def _held(vault: Path) -> Path | None:
    """A lock, so two runs never work on the vault at once.

    A stale lock from a killed run is cleared after an hour rather than
    requiring someone to find and delete a file they did not know existed.
    """
    lock = vault / LOCK
    if lock.exists() and time.time() - lock.stat().st_mtime < 3600:
        return lock
    lock.write_text(str(os.getpid()))
    return None


def survey(index) -> dict:
    """What is owed, without touching the model. This is the dry run, and it
    is worth having: a first maintenance run on a real vault is otherwise a
    process of unknown length doing unknown work."""
    from .resolve_queue import ensure_schema
    ensure_schema(index)
    return {
        "pages": index.db.execute(
            "SELECT COUNT(*) c FROM files").fetchone()["c"],
        "to analyse": len(index.pending_analysis()),
        "to embed": len(index.pending_embed()),
        "name pairs open": index.db.execute(
            "SELECT COUNT(*) c FROM queue WHERE status='open'").fetchone()["c"],
        "concepts with no page": len(candidates(index, 1000)),
        "concept pages with new material": len(stale_concepts(index)),
        "concept pages grown large": len(bloated(index)),
    }


def maintain(sentinel, model, encoder=None, concept_limit: int = 5,
             resolution_rounds: int = 2, minutes: float | None = None,
             verbose: bool = True, summary_model=None) -> dict:
    idx = sentinel.index
    started = time.monotonic()
    out: dict = {}

    def over_budget() -> bool:
        return minutes is not None and (time.monotonic() - started) > minutes * 60

    def say(s):
        if verbose:
            print(s)

    say("sync")
    out["sync"] = idx.sync()

    if not over_budget():
        say("analysis")
        results = run_pass(sentinel, model, verbose=verbose, release=False,
                           summary_model=summary_model)
        out["analysed"] = len(results)
        out["analysis failed"] = sum(1 for r in results
                                     if r["status"] == "failed")

    if encoder is not None and not over_budget():
        from .embed import embed_pending
        say("embeddings")
        out["embedded"] = embed_pending(idx, encoder)

    if not over_budget():
        say("name resolution")
        out["queue"] = build_queue(idx, encoder)
        if next_batch(idx, 1):
            out["resolved"] = run_resolution(
                idx, model, rounds=resolution_rounds, release=False)

    if not over_budget():
        say("concept pages")
        results = run_concepts(sentinel, model, limit=concept_limit,
                               verbose=verbose, release=False)
        out["written"] = sum(1 for r in results if r["status"] == "written")
        out["grown"] = sum(1 for r in results if r["status"] == "grown")

    say("sync")
    idx.sync()
    _release(model, verbose)
    if summary_model is not None:
        _release(summary_model, verbose=False)
    out["seconds"] = round(time.monotonic() - started, 1)
    out["over budget"] = over_budget()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--summary-model", default=None,
                    help="A second model, with a large context, to write the "
                         "summary of pages the extraction pass can only read "
                         "in windows - transcripts and anything past the "
                         "context. Without it those pages keep the "
                         "placeholder summary `write` gave them.")
    ap.add_argument("--summary-num-ctx", type=int, default=120000)
    ap.add_argument("--concepts", type=int, default=5,
                    help="Concept pages to write in one run. Small on "
                         "purpose: a run that writes forty pages produces "
                         "forty pages nobody has read.")
    ap.add_argument("--minutes", type=float, default=None,
                    help="Stop starting new work after this long. A nightly "
                         "run that has not finished by morning should stop, "
                         "not keep the card.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what is owed and do nothing.")
    args = ap.parse_args()

    vault = Path(args.vault).expanduser()
    idx = Index(vault, args.db or default_db(vault),
                exclude=args.exclude)
    if idx.hash_version_changed is not None:
        n = idx.carry_analysis_forward()
        print(f"[note] content-hash definition changed; {n} pages restored")
    idx.sync()

    if args.dry_run:
        for k, v in survey(idx).items():
            print(f"  {k:<34}{v}")
        return

    held = _held(vault)
    if held:
        print(f"[STOP] another maintenance run holds {held}")
        return

    try:
        encoder = None
        if args.embed_model:
            from .embed import E5Encoder
            encoder = E5Encoder(args.embed_model)
        model = OllamaModel(args.model, args.host, num_ctx=args.num_ctx)
        summary_model = None
        if args.summary_model:
            summary_model = OllamaModel(args.summary_model, args.host,
                                        num_ctx=args.summary_num_ctx)
        out = maintain(Sentinel(idx), model, encoder,
                       concept_limit=args.concepts, minutes=args.minutes,
                       summary_model=summary_model)
        print()
        for k, v in out.items():
            print(f"  {k:<20}{v}")
    finally:
        (vault / LOCK).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
