"""The Sentinel - an OpenAPI tool server.

Open WebUI runs in Docker here, so putting a tool file inside it would mean
installing this package in the container and mounting the vault into it: two
dependencies, both fragile, both invisible when they break. A tool server
needs neither. The vault access and the whole library stay on the host, and
Open WebUI is given an address.

    pip install fastapi uvicorn --break-system-packages
    python3 -m sentinel.server --vault ~/obsidian/Obsidian-1 \\
        --exclude agent_workspace

Then in Open WebUI: Settings -> Tools -> add the address printed at startup.

BOUND TO LOCALHOST BY DEFAULT, and `write`, `relocate` and `remove` are real.
A tool server is an unauthenticated door onto the vault; --host is there for
the Docker case and for nothing else.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .embed import E5Encoder
from .index import Index, default_db
from .tools import Sentinel

app = FastAPI(
    title="The Sentinel",
    version="1.0.0",
    description="Read and write an Obsidian vault through seven primitives.",
)

_S: Sentinel | None = None


def sentinel() -> Sentinel:
    if _S is None:
        raise RuntimeError("server not initialised")
    return _S


# The docstrings ARE the tool descriptions Open WebUI reads, so they are the
# same text `schema.py` measures - one source, no drift between what the
# developer maintains and what the model is shown.


class ReadIn(BaseModel):
    path: str = Field(..., description="A path like wiki/afterglow.md, or a "
                                       "bare page name.")
    depth: str = Field("meta", description="meta | outline | part | window | full")
    target: str | None = Field(None, description="A heading (depth=part) or a "
                                                 "word to centre on (depth=window).")
    from_part: int = Field(1, description="Which part of a split read. Starts at 1.")


class SearchIn(BaseModel):
    query: str = Field(..., description="What you are looking for.")
    depth: str = Field("summary", description="summary | body")
    limit: int = Field(5, description="How many pages to return.")


class ListingIn(BaseModel):
    by: str = Field(..., description="tag | path | recent | all_tags")
    value: str | None = Field(None, description="The tag or folder. Required "
                                                "for tag and path.")
    limit: int = Field(20, description="How many rows.")


class GraphIn(BaseModel):
    subject: str | None = Field(None, description="A page or concept. Omit for "
                                                  "the whole vault.")
    limit: int = Field(10, description="How many incoming links to show.")


class WriteIn(BaseModel):
    path: str = Field(..., description="Exact path. No name resolution on writes.")
    content: str = Field(..., description="What to put there.")
    where: str = Field("section", description="section | whole | end | frontmatter")
    target: str | None = Field(None, description="The heading (where=section) "
                                                 "or the field (where=frontmatter).")
    expect: str = Field(..., description='The expect value from your last read '
                                         'of this page, or "new".')


class RelocateIn(BaseModel):
    path: str = Field(..., description="The page to move.")
    to: str = Field(..., description="Its new path.")


class RemoveIn(BaseModel):
    path: str = Field(..., description="The page to delete.")


@app.post("/read", operation_id="read")
def read(body: ReadIn) -> str:
    """Read one note. Accepts a path or a bare page name.

    depth: meta (default) frontmatter, tags and link counts, ~30 tokens - skip
    it right after `search`, you already hold the summary. outline: the heading
    tree with section sizes. part: one section, named by `target`. window: the
    text around `target`. full: the whole page, split into parts.

    Use `search` instead when you do not yet know which page you want.
    """
    return sentinel().read(body.path, body.depth, body.target, body.from_part)


@app.post("/search", operation_id="search")
def search(body: SearchIn) -> str:
    """Find pages by meaning AND by wording, fused. Use this whenever you do
    not already know the page name.

    depth: summary (default) gives one line per hit saying what that page is -
    five hits cost about 75 tokens, and THIS IS THE NORMAL STOPPING PLACE. Go
    to body only if the summaries showed a page is relevant but did not contain
    the answer.

    Results are ranked and rank is the only signal; no similarity score is
    shown because none is meaningful. If no page matched the wording the output
    says so - treat those hits as unconfirmed.
    """
    return sentinel().search(body.query, body.depth, body.limit)


@app.post("/listing", operation_id="listing")
def listing(body: ListingIn) -> str:
    """Enumerate pages by an exact attribute. Deterministic, not ranked by
    relevance.

    by: tag (pages carrying `value`), path (pages under the folder `value`),
    recent (most recently changed), all_tags (every tag with a count).

    Use `search` instead when you are looking by meaning rather than by an
    attribute you can name exactly.
    """
    return sentinel().listing(body.by, body.value, body.limit)


@app.post("/graph", operation_id="graph")
def graph(body: GraphIn) -> str:
    """Link structure. With no subject, the shape of the whole vault, which is
    how you find out which concepts have no page yet.

    Grouped by what you would DO with each group: a link whose page exists is
    something to read, a link with no page yet is something to write. Asking
    about a concept that has no page is normal here - unresolved links are the
    growth queue, not errors.
    """
    return sentinel().graph(body.subject, body.limit)


@app.post("/write", operation_id="write")
def write(body: WriteIn) -> str:
    """Put content into a note. Exact paths only.

    where: section (default) replaces the section named by `target`; whole
    replaces the body; end appends; frontmatter sets the field named by
    `target`.

    expect is REQUIRED: the value from your last read of this page, or "new"
    for a page that does not exist. If the file changed since you read it the
    write is refused so you can merge rather than overwrite something you never
    saw.

    Links are NOT placed here. A separate pass does that, and a page without
    them is not broken.
    """
    return sentinel().write(body.path, body.content, body.where, body.target,
                            body.expect)


@app.post("/relocate", operation_id="relocate")
def relocate(body: RelocateIn) -> str:
    """Rename or move a note, redirecting every link that points at it.

    The incoming links are rewritten so the words on those pages do not change.
    """
    return sentinel().relocate(body.path, body.to)


@app.post("/remove", operation_id="remove")
def remove(body: RemoveIn) -> str:
    """Delete a note.

    Links pointing at it are left alone: they become unresolved and rejoin the
    growth queue, which is correct - the concept is still referenced, it just
    has no page again.
    """
    return sentinel().remove(body.path)


def main() -> None:
    global _S
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1",
                    help="Localhost by default. `write`, `relocate` and "
                         "`remove` are real and there is no authentication, so "
                         "widen this only for a container that needs to reach "
                         "in - not for a network.")
    args = ap.parse_args()

    vault = Path(args.vault).expanduser()
    idx = Index(vault, args.db or default_db(vault),
                exclude=args.exclude)
    if idx.hash_version_changed is not None:
        n = idx.carry_analysis_forward()
        print(f"[note] content-hash definition changed; {n} analysed pages "
              f"restored")
    stats = idx.sync()
    print(f"vault: {stats['seen']} pages, "
          f"{len(idx.pending_analysis())} awaiting analysis")

    encoder = None
    if args.embed_model:
        from .embed import embed_pending
        encoder = E5Encoder(args.embed_model)
        n = embed_pending(idx, encoder)
        if n:
            print(f"embedded {n} new chunks")
    else:
        print("[note] no --embed-model: search runs on the literal half alone")

    _S = Sentinel(idx, encoder)

    shown = "localhost" if args.host == "127.0.0.1" else args.host
    print(f"\nOpen WebUI -> Settings -> Tools -> add:")
    print(f"    http://{shown}:{args.port}")
    print("If Open WebUI is in Docker, it cannot reach 127.0.0.1 - that is the")
    print("container's own loopback. Start this with")
    print(f"    --host 0.0.0.0")
    print("and give Open WebUI http://host.docker.internal:%d, adding" % args.port)
    print("    --add-host=host.docker.internal:host-gateway")
    print("to the container, or use the bridge address http://172.17.0.1:%d."
          % args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
