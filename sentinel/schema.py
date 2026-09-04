"""The Sentinel - tool schemas and what they cost.

The guards are not style. Required fields left optional is the single most
common cause of invocation errors; enums on finite-valued fields remove the
argument the model would otherwise invent; a default on every parameter means
routing rarely happens at all.

Fixed overhead is the whole point of the seven-tool design:

    earlier system   32 tools   8188 description tokens
    The Sentinel      7 tools   1592 (estimated, pre-implementation)

Run this module to measure the real figure against your own tokenizer -
whichever model will actually see these descriptions, since that is the only
count that means anything.
"""

from __future__ import annotations

import inspect
import json

MAX_PARAMS = 8

TOOLS = {
    "read": {
        "path": {"type": "string",
                 "description": "A path like wiki/afterglow.md, or a bare page name."},
        "depth": {"type": "string", "default": "meta",
                  "enum": ["meta", "outline", "part", "window", "full"]},
        "target": {"type": "string",
                   "description": "A heading (depth=part) or a word to centre on "
                                  "(depth=window). Ignored otherwise."},
        "from_part": {"type": "integer", "default": 1,
                      "description": "Which part of a split read. Starts at 1."},
    },
    "search": {
        "query": {"type": "string", "description": "What you are looking for."},
        "depth": {"type": "string", "default": "summary",
                  "enum": ["summary", "body"]},
        "limit": {"type": "integer", "default": 5},
    },
    "listing": {
        "by": {"type": "string", "enum": ["tag", "path", "recent", "all_tags"]},
        "value": {"type": "string",
                  "description": "The tag or folder. Required for tag and path."},
        "limit": {"type": "integer", "default": 20},
    },
    "graph": {
        "subject": {"type": "string",
                    "description": "A page or concept. Omit for the whole vault."},
        "limit": {"type": "integer", "default": 10},
    },
    "write": {
        "path": {"type": "string",
                 "description": "Exact path. No name resolution on writes."},
        "content": {"type": "string", "description": "What to put there."},
        "where": {"type": "string", "default": "section",
                  "enum": ["section", "whole", "end", "frontmatter"]},
        "target": {"type": "string",
                   "description": "The heading (where=section) or the field "
                                  "(where=frontmatter)."},
        "expect": {"type": "string",
                   "description": 'The expect value from your last read of this '
                                  'page, or "new" for a page that does not exist.'},
    },
    "relocate": {
        "path": {"type": "string", "description": "The page to move."},
        "to": {"type": "string", "description": "Its new path."},
    },
    "remove": {
        "path": {"type": "string", "description": "The page to delete."},
    },
}

REQUIRED = {"read": ["path"], "search": ["query"], "listing": ["by"], "graph": [],
            "write": ["path", "content", "expect"],
            "relocate": ["path", "to"], "remove": ["path"]}


def schemas(cls) -> list[dict]:
    """Function schemas built from the class's own docstrings, so the text the
    model reads and the text the developer maintains cannot drift apart."""
    out = []
    for name, params in TOOLS.items():
        fn = getattr(cls, name)
        out.append({
            "name": name,
            "description": inspect.getdoc(fn),
            "input_schema": {"type": "object", "properties": params,
                             "required": REQUIRED[name]},
        })
    return out


def check_guards(specs: list[dict]) -> list[str]:
    """`all_tags` is deliberately not `tags`: two enum values one letter apart
    are an invitation to the argument error the research names as the most
    common one."""
    fails = []
    for s in specs:
        props = s["input_schema"]["properties"]
        if len(props) > MAX_PARAMS:
            fails.append(f"{s['name']}: {len(props)} parameters, over {MAX_PARAMS}")
        if not s["description"] or len(s["description"]) < 120:
            fails.append(f"{s['name']}: description too thin to route on")
        for pname, p in props.items():
            if p["type"] == "string" and "enum" not in p and "description" not in p:
                fails.append(f"{s['name']}.{pname}: no enum and no description")
            if "enum" in p and "default" not in p and pname not in s["input_schema"]["required"]:
                fails.append(f"{s['name']}.{pname}: optional enum with no default")
        for r in s["input_schema"]["required"]:
            if r not in props:
                fails.append(f"{s['name']}: required field {r} is not a parameter")
    return fails


def count_tokens(text: str, tokenizer=None) -> tuple[int, str]:
    if tokenizer is not None:
        return len(tokenizer.encode(text).ids), "your tokenizer"
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text)), "cl100k"
    except Exception:
        return round(len(text) / 4), "ESTIMATE at 4 chars/token"


def main(model_dir: str | None = None) -> None:
    from .tools import Sentinel

    tok = None
    if model_dir:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")

    specs = schemas(Sentinel)
    fails = check_guards(specs)
    print("schema guards:", "all pass" if not fails else "")
    for f in fails:
        print("  FAIL " + f)

    total = 0
    print(f"\n{'tool':<10}{'chars':>8}{'tokens':>9}")
    for s in specs:
        blob = json.dumps(s)
        n, how = count_tokens(blob, tok)
        total += n
        print(f"{s['name']:<10}{len(blob):>8}{n:>9}")
    print(f"{'':<10}{'':>8}{'-' * 8:>9}")
    print(f"{'7 tools':<10}{'':>8}{total:>9}   ({how})")
    print(f"\nRecorded pre-build guess:  1592")
    print(f"Earlier system, 32 tools:  8188")
    print(f"\nFixed overhead {total} + 5548 system prompt = {total + 5548}"
          f", {round((total + 5548) / 17000 * 100)}% of 17k")
    print(f"Working budget: {17000 - total - 5548} against the earlier 3264")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
