"""Run: python3 -m sentinel.tests.test_server"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from .. import server as srv
from ..index import Index
from ..tools import Sentinel

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


tmp = Path(tempfile.mkdtemp())
vault = tmp / "vault"
(vault / "wiki").mkdir(parents=True)
(vault / "wiki" / "afterglow.md").write_text(
    "---\ntype: concept\ntags: [astro]\nsummary: Late-time emission.\n"
    "summary_provisional: 0\n---\n\n# afterglow\n\n## Origin\n\n"
    "It follows the [[gamma-ray burst]].\n\n## Decay\n\nIt fades.\n",
    encoding="utf-8")

idx = Index(vault, tmp / "s.db")
idx.sync()
srv._S = Sentinel(idx)
client = TestClient(srv.app)

# --- the OpenAPI document Open WebUI reads -------------------------------
spec = client.get("/openapi.json").json()
paths = set(spec["paths"])
ok("all seven primitives are exposed",
   paths == {"/read", "/search", "/listing", "/graph", "/write", "/relocate",
             "/remove"}, str(sorted(paths)))
ops = {p: spec["paths"][p]["post"]["operationId"] for p in paths}
ok("each has a stable operation id, which is the tool name",
   set(ops.values()) == {"read", "search", "listing", "graph", "write",
                         "relocate", "remove"}, str(ops))
descs = [spec["paths"][p]["post"].get("description", "") for p in paths]
ok("every tool carries its docstring as the description",
   all(len(d) > 120 for d in descs), str([len(d) for d in descs]))

schemas = spec["components"]["schemas"]
ok("required fields are marked required",
   set(schemas["WriteIn"]["required"]) == {"path", "content", "expect"},
   str(schemas["WriteIn"].get("required")))
ok("optional fields carry their default",
   schemas["SearchIn"]["properties"]["depth"]["default"] == "summary")
ok("the enum values are spelled out in the description",
   "outline" in schemas["ReadIn"]["properties"]["depth"]["description"])

# --- dispatch -------------------------------------------------------------
r = client.post("/read", json={"path": "afterglow"})
ok("read works through the wire", r.json().startswith("[OK] wiki/afterglow.md"),
   r.text[:120])
ok("and hands out the expect value", "expect" in r.json())

r = client.post("/search", json={"query": "afterglow"})
ok("search works", r.json().startswith("[OK]") or r.json().startswith("[STOP]"),
   r.text[:120])

r = client.post("/listing", json={"by": "recent"})
ok("listing works", "wiki/afterglow.md" in r.json(), r.text[:200])

r = client.post("/graph", json={})
ok("graph with no subject gives the vault shape",
   "vault link shape" in r.json(), r.text[:200])
r = client.post("/graph", json={"subject": ""})
ok("an empty subject is the same as none", "vault link shape" in r.json())

# --- failures arrive as statuses, not as HTTP errors ---------------------
r = client.post("/read", json={"path": "nope"})
ok("a missing page is a [STOP] with status 200, not a 404",
   r.status_code == 200 and r.json().startswith("[STOP]"),
   f"{r.status_code} {r.text[:80]}")
r = client.post("/read", json={"path": "afterglow", "depth": "summary"})
ok("a borrowed enum value is [RETRY] and says so",
   r.json().startswith("[RETRY]") and "is a `search` depth" in r.json(),
   r.text[:160])
r = client.post("/write", json={"path": "wiki/x.md", "content": "x"})
ok("a missing required field is refused by the schema, before the tool",
   r.status_code == 422, str(r.status_code))

# --- the write path is real ----------------------------------------------
r = client.post("/write", json={"path": "wiki/new.md",
                                "content": "# new\n\nThe jet break steepens it.",
                                "expect": "new"})
ok("write creates a page", r.json().startswith("[DONE] created"), r.text[:120])
ok("and the page is on disk", (vault / "wiki" / "new.md").exists())

h = idx.meta("wiki/new.md")["content_hash"][:8]
r = client.post("/write", json={"path": "wiki/new.md", "content": "# new\n\nEdited.",
                                "where": "whole", "expect": "deadbeef"})
ok("a stale expect is refused over the wire too", r.json().startswith("[RETRY]"),
   r.text[:120])
r = client.post("/write", json={"path": "wiki/new.md", "content": "# new\n\nEdited.",
                                "where": "whole", "expect": h})
ok("the current expect succeeds", r.json().startswith("[DONE]"), r.text[:120])

r = client.post("/relocate", json={"path": "wiki/new.md", "to": "wiki/moved.md"})
ok("relocate works", r.json().startswith("[DONE]"), r.text[:120])
r = client.post("/remove", json={"path": "wiki/moved.md"})
ok("remove works", r.json().startswith("[DONE]"), r.text[:120])
ok("and the file is gone", not (vault / "wiki" / "moved.md").exists())

# --- concurrent requests, which is what a server actually gets -----------
import threading  # noqa: E402

errors: list[str] = []


def hammer(i):
    try:
        for _ in range(5):
            r = client.post("/read", json={"path": "afterglow",
                                           "depth": "outline"})
            if r.status_code != 200:
                errors.append(f"{r.status_code} {r.text[:80]}")
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")


threads = [threading.Thread(target=hammer, args=(i,)) for i in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join()
ok("six concurrent clients do not break the index", not errors, str(errors[:2]))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
sys.exit(1 if FAIL else 0)
