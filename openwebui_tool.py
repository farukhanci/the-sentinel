"""
title: The Sentinel
author: Faruk Hancı
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

# DROP THE CACHED PACKAGE so saving this file picks up the code on disk.
#
# Open WebUI is a long-running process and Python caches what it imports.
# Editing the mounted package therefore changes nothing until the container
# restarts - and the failure is silent, which is what makes it dangerous. It
# was diagnosed only because a `graph` result came back with wording that had
# been replaced hours earlier: every test that day had been run against code
# that was no longer on disk.
#
# Saving this file re-executes it, so clearing the modules here makes that
# the reload.
for _cached in [
    m for m in list(sys.modules) if m == "sentinel" or m.startswith("sentinel.")
]:
    del sys.modules[_cached]


class Tools:
    class Valves(BaseModel):
        vault: str = Field(
            default="/vault",
            description="The vault path AS SEEN INSIDE THE CONTAINER.",
        )
        db_path: str = Field(
            default="",
            description="Where the derived index lives. Leave empty to use "
            "the one path the rest of the system derives from the vault - "
            "set it only to override. It sits inside the vault so it survives "
            "a container rebuild; it is derived data and a full rebuild is "
            "always possible.",
        )
        pages: str = Field(
            default="notes",
            description="Folder for pages written from conversation. A path "
            "with no folder lands here.",
        )
        sources: str = Field(
            default="sources",
            description="Folder of filed raw sources. Indexed and searchable "
            "at the deeper rung, but held out of summary search and placing "
            "no links, like a conversation record.",
        )
        records: str = Field(
            default="conversations",
            description="Folder the code writes conversation records into. "
            "Writing there is refused, so the record stays a record.",
        )
        concepts: str = Field(
            default="wiki",
            description="Folder the maintenance pass writes concept pages "
            "into. Writing there from a conversation is refused, so the "
            "folder stays an answer to what the vault knows.",
        )
        exclude: str = Field(
            default="",
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
        from sentinel.index import Index, default_db
        from sentinel.tools import Sentinel

        vault = Path(self.valves.vault)
        if not vault.is_dir():
            raise RuntimeError(
                f"vault not found at {vault} inside the container - check the "
                f"bind mount"
            )
        # One place decides this path. A second copy of it here is exactly
        # how the index split in two before: both halves worked, separately.
        db = Path(self.valves.db_path) if self.valves.db_path else default_db(vault)
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
        self._s = Sentinel(
            idx,
            encoder,
            pages=self.valves.pages,
            concepts=self.valves.concepts,
            records=self.valves.records,
            sources=self.valves.sources,
        )
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
        folder = s.index.vault / s.records
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
            encoding="utf-8",
        )
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
        return self._call(
            "read", path, depth, target or None, from_part, __metadata__=__metadata__
        )

    def search(
        self,
        query: str,
        depth: str = "summary",
        limit: int = 5,
        __metadata__: dict = {},
    ) -> str:
        """
        Find pages by meaning AND by wording, fused. Use this whenever you do not already know the page name.

        Results are ranked, and rank is the only signal - no similarity score is shown because none is meaningful here. If no page matched the wording the output says so; treat those hits as unconfirmed.

        :param query: What you are looking for.
        :param depth: summary (default) gives one line per hit saying what that page is, about 75 tokens for five hits, and THIS IS THE NORMAL STOPPING PLACE. Use body only when the summaries showed a page is relevant but did not contain the answer.
        :param limit: How many pages to return. 5 suits a conversation.
        """
        return self._call("search", query, depth, limit, __metadata__=__metadata__)

    def listing(
        self, by: str, value: str = "", limit: int = 20, __metadata__: dict = {}
    ) -> str:
        """
        Enumerate pages by an exact attribute. Deterministic, not ranked by relevance. Use search instead when looking by meaning.

        :param by: tag (pages carrying value) | path (pages under the folder value) | recent (most recently changed) | all_tags (every tag with a count).
        :param value: The tag or folder. Required for tag and path.
        :param limit: How many rows.
        """
        return self._call(
            "listing", by, value or None, limit, __metadata__=__metadata__
        )

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

        USE THIS WHEN THE USER ASKS YOU TO RECORD SOMETHING - "kaydet", "save this", "write that down". Not on your own judgement of what matters. When they ask, do it in ONE call: do not search first, do not ask which page, do not describe what you are about to write. If they name a page, use it; otherwise choose a name and say which one you used.

        When you create a page this way the conversation itself is recorded alongside it, and a link to that record is added to the page. You do not have to do either.

        WHERE IT GOES: notes/<name>.md - a path with no folder lands there. Name it by SUBJECT, never by date or conversation: the page written today is the page added to next week from a different chat, and it can only be found again if its address is the subject. wiki/ belongs to the maintenance pass, which writes concept pages there from sources; writing to it is refused. Never the vault root either.

        WRITE WHAT THE CONVERSATION SETTLED, NOT WHAT YOU KNOW ABOUT THE SUBJECT. If they name what to record, record that; if they only say "kaydet", write what was worked out in the exchange - including a conclusion you drew from what the vault told you. What does not belong is the rest of what you know about the topic. Measured: asked to record a conversation about a device, the model wrote an encyclopaedia entry - dates, formulas, a build guide - none of which had been discussed, on a page that carries the user's own origin marking.

        Write it the way you understood it - organised, with headings, in your own arrangement. The concepts come from the conversation record, not from this page, so shaping the page costs nothing. TWO THINGS DO NOT BELONG: no task list, because three ticked boxes read as a commitment the user made; and no number that was not said, because a figure you worked out reads as a measurement when everything around it is one. A date is a number: you do not know today's, and the system writes it into the page for you, so never put one in the text.

        Links are NOT placed here. A separate pass does that, and a page without them is not broken.

        :param path: Exact path, such as wiki/afterglow.md.
        :param content: What to put there.
        :param expect: REQUIRED. The expect value from your last read of this page, or "new" for a page that does not exist. If the file changed since you read it the write is refused, so you can merge rather than overwrite something you never saw.
        :param where: section (default, replaces the section named by target) | whole | end | frontmatter (sets the field named by target).
        :param target: The heading when where=section, or the field when where=frontmatter.
        """
        # EVERY write during a conversation refreshes the record, not only
        # the first.
        #
        # It used to be tied to `expect="new"`, and the effect was that the
        # record froze at whatever had been said by the time the page was
        # first written. The conversation carried on, the page grew, and the
        # transcript still ended at the opening message - so the concepts the
        # vault gates against were drawn from a fraction of what was said.
        name = None
        if __messages__:
            try:
                name = self._transcript(__messages__, __metadata__)
            except Exception as e:
                return f"[STOP] could not record the conversation: {e}"

        # The link to it goes in once. Checked against the file rather than
        # against `expect`, because a second write to the same page arrives as
        # `expect="new"` too and would otherwise add the line again.
        if name:
            sen = self._sentinel()
            try:
                # Ask where the write will land rather than working it out
                # here. The rule lives in one place; a second copy of it in
                # this file is how the link came to be added twice.
                target = sen.index.vault / sen.write_path(path)
                existing = target.read_text(encoding="utf-8")
            except Exception:
                existing = ""
            # ONE LINE PER CONVERSATION THAT CONTRIBUTED, not one per page.
            #
            # A note is addressed by its subject, so a page written on the 1st
            # is the same page added to on the 3rd from a different chat. Each
            # of those conversations is part of where the page came from, and
            # dropping the second would leave the page pointing at only half
            # its own history. What must not repeat is the SAME record.
            line = f"Kayıt: [[{name}]]"
            if line not in existing and line not in content:
                content = content.rstrip() + f"\n\n{line}\n"
        return self._call(
            "write",
            path,
            content,
            where,
            target or None,
            expect,
            __metadata__=__metadata__,
        )

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

