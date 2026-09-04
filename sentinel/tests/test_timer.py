"""Run: python3 -m sentinel.tests.test_timer"""

from __future__ import annotations

import argparse
import sys

from ..timer import build

PASS: list[str] = []
FAIL: list[str] = []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}{' - ' + detail if detail and not cond else ''}")


def args(**kw):
    d = dict(vault="~/obsidian/Obsidian-1", model="qwen3.5-9b-jinja:latest",
             host="http://localhost:11434", exclude=[], embed_model=None,
             num_ctx=8192, concepts=5, at="23:00", minutes=30,
             summary_model=None, summary_num_ctx=120000)
    d.update(kw)
    return argparse.Namespace(**d)


service, timer = build(args(exclude=["agent_workspace", "raw"]))

ok("the service runs the maintenance module",
   "-m sentinel.maintain" in service, service)
ok("with the vault expanded to a real path",
   "~" not in service.split("ExecStart")[1].split("\n")[0], service)
ok("and every exclusion passed through",
   service.count("--exclude") == 2, service)
ok("it runs at low priority, because this is a laptop",
   "Nice=10" in service and "IOSchedulingClass=idle" in service)
ok("systemd's timeout is looser than the run's own budget, so a finishing "
   "run is never killed",
   "TimeoutStartSec=3600" in service, service)

ok("the timer fires at the hour asked for", "OnCalendar=23:00" in timer)
ok("a missed run is caught up, because the machine sleeps",
   "Persistent=true" in timer)
ok("with a delay, so it does not start the instant the lid opens",
   "RandomizedDelaySec" in timer)
ok("and it installs into timers.target", "WantedBy=timers.target" in timer)

s2, t2 = build(args(at="Mon *-*-* 09:00:00", minutes=None,
                    embed_model="~/models/multilingual-e5-small"))
ok("any OnCalendar expression works", "OnCalendar=Mon *-*-* 09:00:00" in t2)
ok("the embedding model is passed and expanded",
   "--embed-model" in s2 and "~" not in s2.split("--embed-model")[1][:40], s2)
ok("no budget means no --minutes flag", "--minutes" not in s2, s2)
ok("and the timeout still has a sane default", "TimeoutStartSec=7200" in s2, s2)

s5, _ = build(args(summary_model="qwen3.5-4b-xl"))
ok("a summary model is carried into the unit",
   "--summary-model qwen3.5-4b-xl" in s5 and "--summary-num-ctx 120000" in s5,
   s5)
ok("and is absent when none is given",
   "--summary-model" not in build(args())[0])

s3, _ = build(args(host="http://other:11434"))
ok("a non-default host is passed", "--host" in s3)
s4, _ = build(args())
ok("the default host is not", "--host" not in s4)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed\n")
for f in FAIL:
    print("  FAIL  " + f)
sys.exit(1 if FAIL else 0)
