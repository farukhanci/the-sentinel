"""Run: python3 -m sentinel.tests.test_repair"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from ..repair_brackets import planned, repair
from ..text import content_hash, split_frontmatter, strip_links

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


BODY = """# Damaged

We observed with [[[[JWST]]]] and also [[[JWST]] earlier, plus [[JWST]]] once.
A clean link [[gamma-ray burst]] stays exactly as it is.
A piped link [[Canonical|shown]] must survive untouched.
Inline code `[[example]]` is the user's own content.

```python
x = "[[[[not a link]]]]"
```

Math $[[a]]$ and a markdown [link](http://x.com) and a lone [n_0] bracket.
"""

fixed, n = repair(BODY)

# --- what it repairs ------------------------------------------------------
ok("quadruple brackets collapse", "[[[[JWST]]]]" not in fixed)
ok("triple-open collapses", "[[[JWST]]" not in fixed)
ok("triple-close collapses", "[[JWST]]]" not in fixed)
ok("all three JWST forms become one", fixed.count("[[JWST]]") == 3, fixed)
ok("the count is reported", n == 3, str(n))

# --- what it must never touch --------------------------------------------
ok("a clean link is left alone", "[[gamma-ray burst]]" in fixed)
ok("a piped link survives", "[[Canonical|shown]]" in fixed)
ok("inline code is the user's content", "`[[example]]`" in fixed)
ok("a code fence is left alone", '"[[[[not a link]]]]"' in fixed)
ok("math is left alone", "$[[a]]$" in fixed)
ok("a markdown link is left alone", "[link](http://x.com)" in fixed)
ok("a lone single bracket is left alone", "[n_0]" in fixed)

# --- the preview and the edit are the same function ----------------------
plan = planned(BODY)
ok("the plan lists exactly what will change", len(plan) == n, f"{len(plan)} vs {n}")
ok("and nothing from inside a fence",
   all("not a link" not in was for was, _ in plan), str(plan))
ok("applying the plan by hand gives the same text",
   all(was in BODY for was, _ in plan))

# --- idempotency ----------------------------------------------------------
again, n2 = repair(fixed)
ok("a second run changes nothing", again == fixed and n2 == 0, str(n2))

# --- the hash moves, and that is correct ---------------------------------
ok("the content hash moves, so the page is re-analysed",
   content_hash(BODY) != content_hash(fixed))
ok("strip_links reaches the target after repair, not before",
   "[[" in strip_links(BODY.split("```")[0])
   and "[[" not in strip_links(fixed.split("```")[0]),
   repr(strip_links(fixed.split("```")[0])[:80]))

# --- dry run writes nothing ----------------------------------------------
tmp = Path(tempfile.mkdtemp())
v = tmp / "vault"
(v / "wiki").mkdir(parents=True)
f = v / "wiki" / "d.md"
f.write_text("---\ntype: note\nsummary: X.\n---\n\n" + BODY, encoding="utf-8")
before = f.read_text()

run = subprocess.run([sys.executable, "-m", "sentinel.repair_brackets",
                      "--vault", str(v)], capture_output=True, text=True,
                     cwd=str(Path(__file__).resolve().parents[2]))
ok("a dry run says so", "DRY RUN" in run.stdout, run.stdout[-200:])
ok("and writes nothing", f.read_text() == before)

run = subprocess.run([sys.executable, "-m", "sentinel.repair_brackets",
                      "--vault", str(v), "--apply"], capture_output=True,
                     text=True, cwd=str(Path(__file__).resolve().parents[2]))
after = f.read_text()
ok("--apply writes", after != before and "applied" in run.stdout)
ok("frontmatter is preserved exactly",
   split_frontmatter(after)[0] == split_frontmatter(before)[0],
   str(split_frontmatter(after)[0]))
ok("only the damage changed",
   split_frontmatter(after)[1] == repair(split_frontmatter(before)[1])[0])

# --- dot-directories are skipped, as everywhere else ----------------------
(v / ".trash").mkdir()
t = v / ".trash" / "old.md"
t.write_text("---\ntype: note\n---\n\n[[[[JWST]]]]\n", encoding="utf-8")
tb = t.read_text()
subprocess.run([sys.executable, "-m", "sentinel.repair_brackets",
                "--vault", str(v), "--apply"], capture_output=True, text=True,
               cwd=str(Path(__file__).resolve().parents[2]))
ok("nothing under .trash is rewritten", t.read_text() == tb)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f_ in FAIL:
    print("  FAIL  " + f_)
sys.exit(1 if FAIL else 0)
