"""
title: The Sentinel
author: kozmos
version: 1.0.0
description: Read and write an Obsidian vault through seven primitives.
requirements:
"""

# The Sentinel - Open WebUI Workspace Tool.
#
# Paste this whole file into Workspace -> Tools -> +. It runs IN PROCESS
# inside the Open WebUI container, which means two things have to be true:
#
#   1. this package must be importable there
#   2. the vault must be visible there
#
# Both are bind mounts, and no package needs installing: the core is pure
# standard library. `numpy` is imported only inside the semantic half of
# search, so without it the literal half still works and everything else is
# unaffected.
#
#   docker run -d -p 3000:8080 \
#     --user $(id -u):$(id -g) \
#     -v open-webui:/app/backend/data \
#     -v ~/the-sentinel:/sentinel:ro \
#     -v ~/obsidian/Obsidian-1:/vault \
#     --name open-webui ghcr.io/open-webui/open-webui:main
#
# `--user` is not optional in practice. A container running as root writes
# root-owned files into the vault, and then nothing on the host can edit them -
# which is exactly how `agent_workspace` ended up unwritable.
#
# A bind mount cannot be added to a running container; the container has to be
# recreated. Data survives because it lives in the named volume.

import sys
from pathlib import Path

from pydantic import BaseModel, Field

SENTINEL_PATH = "/sentinel"
if SENTINEL_PATH not in sys.path:
    sys.path.insert(0, SENTINEL_PATH)


class Tools:
    class Valves(BaseModel):
        vault: str = Field(
            default="/vault",
            description="The vault path AS SEEN INSIDE THE CONTAINER.",
        )
        db_path: str = Field(
            default="/vault/.sentinel/index.db",
            description="Where the derived index lives. Inside the vault by "
            "default so it survives a container rebuild; it is derived data "
            "and a full rebuild is always possible.",
        )
        exclude: str = Field(
            default="agent_workspace",
            description="Comma-separated folders to leave out of the index. "
            "Dot-folders are always skipped.",
        )
        embed_model: str = Field(
            default="",
            description="Path to multilingual-e5-small inside the container. "
            "Leave empty to run search on the literal half alone - which "
            "works, and needs no numpy or onnxruntime.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._s = None

    # -- lazily built, so a bad path is a message and not a broken tool list --

    def _sentinel(self):
        if self._s is not None:
            return self._s
        from sentinel.index import Index
        from sentinel.tools import Sentinel

        vault = Path(self.valves.vault)
        if not vault.is_dir():
            raise RuntimeError(
                f"vault not found at {vault} inside the container - check the "
                f"bind mount"
            )
        db = Path(self.valves.db_path)
        db.parent.mkdir(parents=True, exist_ok=True)
        idx = Index(
            vault,
            db,
            exclude=[e.strip() for e in self.valves.exclude.split(",") if e.strip()],
        )
        if idx.hash_version_changed is not None:
            idx.carry_analysis_forward()
        idx.sync()

        encoder = None
        if self.valves.embed_model:
            from sentinel.embed import E5Encoder, embed_pending

            encoder = E5Encoder(self.valves.embed_model)
            embed_pending(idx, encoder)
        self._s = Sentinel(idx, encoder)
        return self._s

    def _scoped(self, metadata):
        """Give the tools the conversation id, so one chat's search history
        never speaks for another's.

        Without it the repeat guard shares a single bucket across every
        conversation, and a query from a previous chat gets attributed to this
        one - measured, and it made the model report results it had never
        received.
        """
        s = self._sentinel()
        s.scope = str((metadata or {}).get("chat_id") or "default")
        return s

    def _transcript(self, messages, metadata) -> str:
        """Write the conversation to conversations/ and return its page name.

        THE CODE COPIES IT, NOT THE MODEL. A model asked to reproduce a long
        conversation reproduces what is left in its context, and by then the
        older turns have been dropped to stay inside the window - so the
        record would be missing exactly the part that a record is for, and it
        would be missing it silently.

        One file per conversation, found again by its chat id, so a second
        write in the same chat updates the record instead of starting a new
        one.
        """
        import datetime as _dt

        s = self._sentinel()
        folder = s.index.vault / "conversations"
        folder.mkdir(parents=True, exist_ok=True)
        chat_id = str((metadata or {}).get("chat_id") or "")

        existing = None
        if chat_id:
            for f in folder.glob("*.md"):
                if f"chat_id: {chat_id}" in f.read_text(encoding="utf-8")[:400]:
                    existing = f
                    break

        if existing is None:
            day = _dt.date.today().isoformat()
            n = 1 + sum(1 for f in folder.glob(f"{day}-*.md"))
            existing = folder / f"{day}-{n}.md"

        lines = []
        for m in messages or []:
            role = m.get("role")
            text = str(m.get("content") or "").strip()
            if role not in ("user", "assistant") or not text:
                continue
            lines.append(f"## {'You' if role == 'user' else 'The Sentinel'}\n\n{text}")

        existing.write_text(
            f"---\ntype: transcript\nchat_id: {chat_id}\n"
            f"origin: conversation\ncreated: {_dt.date.today().isoformat()}\n"
            f"---\n\n# {existing.stem}\n\n" + "\n\n".join(lines) + "\n",
            encoding="utf-8")
        return existing.stem

    def _call(self, name, *a, __metadata__=None):
        """Every failure is a STATUS the model can act on, never a traceback.

        A traceback reaching the model is a result it cannot do anything with,
        and Open WebUI surfaces an exception as a tool error rather than as
        content - so the model would not even see what went wrong.
        """
        try:
            return getattr(self._scoped(__metadata__), name)(*a)
        except Exception as e:
            return f"[STOP] {name} failed: {type(e).__name__}: {e}"

    # -- the seven ---------------------------------------------------------

    def read(
        self,
        path: str,
        depth: str = "meta",
        target: str = "",
        from_part: int = 1,
        __metadata__: dict = {},
    ) -> str:
        """
        Read one note. Accepts a path like wiki/afterglow.md or a bare page name.

        :param path: A path, or a bare page name.
        :param depth: meta (default, ~30 tokens - skip it right after search, you already hold the summary) | outline (the heading tree with section sizes) | part (one section, named by target) | window (the text around target) | full (the whole page, split into parts).
        :param target: A heading when depth=part, or a word to centre on when depth=window. Ignored otherwise.
        :param from_part: Which part of a split read. Starts at 1.
        """
        return self._call("read", path, depth, target or None, from_part, __metadata__=__metadata__)

    def search(self, query: str, depth: str = "summary", limit: int = 5, __metadata__: dict = {}) -> str:
        """
        Find pages by meaning AND by wording, fused. Use this whenever you do not already know the page name.

        Results are ranked, and rank is the only signal - no similarity score is shown because none is meaningful here. If no page matched the wording the output says so; treat those hits as unconfirmed.

        :param query: What you are looking for.
        :param depth: summary (default) gives one line per hit saying what that page is, about 75 tokens for five hits, and THIS IS THE NORMAL STOPPING PLACE. Use body only when the summaries showed a page is relevant but did not contain the answer.
        :param limit: How many pages to return. 5 suits a conversation.
        """
        return self._call("search", query, depth, limit, __metadata__=__metadata__)

    def listing(self, by: str, value: str = "", limit: int = 20, __metadata__: dict = {}) -> str:
        """
        Enumerate pages by an exact attribute. Deterministic, not ranked by relevance. Use search instead when looking by meaning.

        :param by: tag (pages carrying value) | path (pages under the folder value) | recent (most recently changed) | all_tags (every tag with a count).
        :param value: The tag or folder. Required for tag and path.
        :param limit: How many rows.
        """
        return self._call("listing", by, value or None, limit, __metadata__=__metadata__)

    def graph(self, subject: str = "", limit: int = 10, __metadata__: dict = {}) -> str:
        """
        Link structure. With no subject, the shape of the whole vault - which is how you find out which concepts have no page yet.

        Grouped by what you would DO with each group: a link whose page exists is something to read, a link with no page yet is something to write. Asking about a concept that has no page is normal here; unresolved links are the growth queue, not errors.

        :param subject: A page or concept. Leave empty for the whole vault.
        :param limit: How many incoming links to show.
        """
        return self._call("graph", subject or None, limit, __metadata__=__metadata__)

    def write(
        self,
        path: str,
        content: str,
        expect: str,
        where: str = "section",
        target: str = "",
        __metadata__: dict = {},
        __messages__: list = [],
    ) -> str:
        """
        Put content into a note. Exact paths only, no name resolution.

        USE THIS TO RECORD WHAT THE USER HAS JUST WORKED OUT, not only when they ask you to. If they state a finding, a decision, or what a term means, and no page holds it, create the page with expect="new". One search first is fine; if what comes back is about other subjects then the page does not exist and this is the next call to make.

        When you create a page this way the conversation itself is recorded alongside it, and a link to that record is added to the page. You do not have to do either.

        Write it the way you understood it - organised, with headings, in your own arrangement. The concepts come from the conversation record, not from this page, so shaping the page costs nothing. TWO THINGS DO NOT BELONG: no task list, because three ticked boxes read as a commitment the user made; and no number that was not said, because a figure you worked out reads as a measurement when everything around it is one.

        Links are NOT placed here. A separate pass does that, and a page without them is not broken.

        :param path: Exact path, such as wiki/afterglow.md.
        :param content: What to put there.
        :param expect: REQUIRED. The expect value from your last read of this page, or "new" for a page that does not exist. If the file changed since you read it the write is refused, so you can merge rather than overwrite something you never saw.
        :param where: section (default, replaces the section named by target) | whole | end | frontmatter (sets the field named by target).
        :param target: The heading when where=section, or the field when where=frontmatter.
        """
        # Creating a page from a conversation records the conversation too,
        # and links the page to it. The record holds what the page left out,
        # which is the reason for keeping one.
        if expect == "new" and __messages__:
            try:
                name = self._transcript(__messages__, __metadata__)
                content = content.rstrip() + f"\n\nKayıt: [[{name}]]\n"
            except Exception as e:
                return f"[STOP] could not record the conversation: {e}"
        return self._call("write", path, content, where, target or None,
                          expect, __metadata__=__metadata__)

    def relocate(self, path: str, to: str, __metadata__: dict = {}) -> str:
        """
        Rename or move a note, redirecting every link that points at it. The incoming links are rewritten so the words on those pages do not change.

        :param path: The page to move.
        :param to: Its new path.
        """
        return self._call("relocate", path, to, __metadata__=__metadata__)

    def remove(self, path: str, __metadata__: dict = {}) -> str:
        """
        Delete a note. Links pointing at it are left alone: they become unresolved and rejoin the growth queue, which is correct - the concept is still referenced, it just has no page again.

        :param path: The page to delete.
        """
        return self._call("remove", path, __metadata__=__metadata__)
