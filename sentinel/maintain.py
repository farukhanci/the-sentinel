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
from .models import OllamaModel, OpenAICompatModel
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


def build_model(provider: str, model: str, *, host: str, num_ctx: int,
                base_url: str | None, api_key_env: str | None, role: str):
    """One role's model, from that role's own flags.

    EVERY DEFAULT IS TODAY'S BEHAVIOUR. `--provider` unset means Ollama on
    localhost with the same arguments the pass has always been given, so a
    command line that worked yesterday produces the identical object.

    The key is read from the ENVIRONMENT, never taken on the command line: a
    key passed as an argument is in the shell history, in `ps`, and in the
    systemd unit the timer writes.
    """
    if provider == "ollama":
        return OllamaModel(model, host, num_ctx=num_ctx)
    if not base_url:
        raise SystemExit(f"--{role}base-url is required with "
                         f"--{role}provider openai")
    key = ""
    if api_key_env:
        key = os.environ.get(api_key_env, "")
        if not key:
            # Loudly, and before the pass starts. The failure without this is
            # an unauthenticated request per page, hundreds of them, each one
            # a round trip that could never have worked.
            raise SystemExit(f"{api_key_env} is empty or unset in the "
                             f"environment")
    return OpenAICompatModel(model, base_url, api_key=key, num_ctx=num_ctx)


def _provider_flags(ap, role: str, label: str) -> None:
    ap.add_argument(f"--{role}provider", choices=["ollama", "openai"],
                    default="ollama",
                    help=f"Where the {label} model runs. `ollama` (default) "
                         f"is the direct local client and is the only one "
                         f"that reports load, prefill and generation "
                         f"separately. `openai` is any OpenAI-shaped "
                         f"endpoint, which reports a total and nothing else.")
    ap.add_argument(f"--{role}base-url", default=None,
                    help=f"The OpenAI-compatible base URL for the {label} "
                         f"model, e.g. http://localhost:3000/api. Required "
                         f"with --{role}provider openai.")
    ap.add_argument(f"--{role}api-key-env", default=None,
                    help="NAME of the environment variable holding the API "
                         "key - not the key itself, which would land in the "
                         "shell history and in `ps`.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    _provider_flags(ap, "", "extraction")
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
    _provider_flags(ap, "summary-", "summary")
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
        model = build_model(
            args.provider, args.model, host=args.host, num_ctx=args.num_ctx,
            base_url=args.base_url, api_key_env=args.api_key_env, role="")
        summary_model = None
        if args.summary_model:
            # INDEPENDENT of the analysis role, not inherited from it.
            # `--summary-model` exists to be a large-context model; moving the
            # extraction role to a cloud endpoint says nothing about where
            # that one should run, and silently dragging it along would send a
            # model id the endpoint has never heard of.
            summary_model = build_model(
                args.summary_provider, args.summary_model,
                host=args.host, num_ctx=args.summary_num_ctx,
                base_url=args.summary_base_url,
                api_key_env=args.summary_api_key_env, role="summary-")
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
