"""The Sentinel - the nightly timer.

    python3 -m sentinel.timer --install \\
        --vault ~/obsidian/Obsidian-1 --model qwen3.5-9b-jinja:latest \\
        --exclude agent_workspace --at 23:00 --minutes 30

A USER timer, not a system one. The vault is in a home directory, Ollama runs
as the user, and a system unit would need permissions it has no business
having.

PERSISTENT, because the machine is a laptop. A timer that only fires at 23:00
fires never on a machine that is asleep at 23:00. `Persistent=true` runs the
job it missed once the machine is awake again, which is the behaviour a person
means when they say "every night".

Not run while you are talking to the vault, and nothing enforces that: the
analysis model and the conversation model cannot share a 6 GB card, so a run
started mid-conversation evicts the model you are talking to. 23:00 is a
choice about when you are usually finished, not a guarantee.

    systemctl --user start sentinel-maintenance     # run it now
    systemctl --user list-timers sentinel-maintenance
    journalctl --user -u sentinel-maintenance -n 50  # what the last run did
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

UNIT_DIR = Path.home() / ".config" / "systemd" / "user"

SERVICE = """[Unit]
Description=The Sentinel - vault maintenance
Documentation=file://{project}/sentinel/maintain.py

[Service]
Type=oneshot
WorkingDirectory={project}
Environment=PYTHONPATH={project}
# Low priority: this is a laptop and the run takes minutes. It should lose
# every scheduling contest with whatever the user is actually doing.
Nice=10
IOSchedulingClass=idle
ExecStart={python} -m sentinel.maintain {args}
TimeoutStartSec={timeout}
"""

TIMER = """[Unit]
Description=The Sentinel - nightly vault maintenance

[Timer]
OnCalendar={at}
# The machine is a laptop and is not always awake at {at}. Without this the
# job is simply skipped, which over a month means it never runs at all.
Persistent=true
# So several sleeping machines do not all wake and start at once, and so a
# missed run does not begin the instant the lid opens.
RandomizedDelaySec=600

[Install]
WantedBy=timers.target
"""


def build(args) -> tuple[str, str]:
    cli = ["--vault", str(Path(args.vault).expanduser()),
           "--model", args.model]
    for e in args.exclude:
        cli += ["--exclude", e]
    if args.embed_model:
        cli += ["--embed-model", str(Path(args.embed_model).expanduser())]
    if args.host != "http://localhost:11434":
        cli += ["--host", args.host]
    cli += ["--num-ctx", str(args.num_ctx), "--concepts", str(args.concepts)]
    if args.summary_model:
        cli += ["--summary-model", args.summary_model,
                "--summary-num-ctx", str(args.summary_num_ctx)]
    if args.minutes:
        cli += ["--minutes", str(args.minutes)]

    project = Path(__file__).resolve().parents[1]
    service = SERVICE.format(
        project=project, python=sys.executable,
        args=" ".join(shlex.quote(c) for c in cli),
        # Give it room past its own budget, so systemd never kills a run that
        # is finishing up.
        timeout=int((args.minutes or 60) * 60 * 2))
    return service, TIMER.format(at=args.at)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--summary-model", default=None)
    ap.add_argument("--summary-num-ctx", type=int, default=120000)
    ap.add_argument("--concepts", type=int, default=5)
    ap.add_argument("--at", default="23:00",
                    help="A systemd OnCalendar time. 23:00 means daily.")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--install", action="store_true",
                    help="Write the units and enable the timer. Without it, "
                         "the units are printed and nothing is touched.")
    args = ap.parse_args()

    service, timer = build(args)
    if not args.install:
        print(f"# {UNIT_DIR}/sentinel-maintenance.service\n{service}")
        print(f"# {UNIT_DIR}/sentinel-maintenance.timer\n{timer}")
        print("# add --install to write these and enable the timer")
        return

    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    (UNIT_DIR / "sentinel-maintenance.service").write_text(service)
    (UNIT_DIR / "sentinel-maintenance.timer").write_text(timer)
    for cmd in (["systemctl", "--user", "daemon-reload"],
                ["systemctl", "--user", "enable", "--now",
                 "sentinel-maintenance.timer"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(f"[STOP] {' '.join(cmd)}: {r.stderr.strip()}")
            return

    print(f"installed in {UNIT_DIR}")
    print(f"  runs at {args.at}, and catches up a missed run when the "
          f"machine wakes")
    print("  systemctl --user start sentinel-maintenance   # run it now")
    print("  systemctl --user list-timers sentinel-maintenance")
    print("  journalctl --user -u sentinel-maintenance -n 50")
    if not os.path.exists(f"/var/lib/systemd/linger/{os.getlogin()}"):
        # Worth saying plainly rather than leaving as a surprise.
        print("\n[note] user timers run while you are logged in. If the "
              "laptop is\n       logged out at that hour, the run happens on "
              "your next login\n       instead. `loginctl enable-linger` "
              "changes that if you want it.")


if __name__ == "__main__":
    main()
