#!/usr/bin/env python3
"""agent_manager.py

Drive a loop between two coding-agent sessions running in tmux (claude or
codex), until a goal is reached or declared unreachable.

Roles
-----
- controller : the agent that plans and supervises (in `--controller` tmux)
- executor   : the agent that writes the code              (in `--executor`  tmux)
- manager    : this script. Owns the loop, bridges controller<->executor, and
               appends a compact log to `--memory` (default ./MEMORY.md).

The controller emits sentinel-prefixed lines that this script parses:

    STATUS:  <one short line>
    NOTE:    <one short line>
    EXECUTOR: <prompt for the executor>          (multi-line ok; end with line
                                                  containing exactly:
                                                  END_EXECUTOR)
    DONE:    <reason>                            -> exit 0
    BLOCKED: <reason>                            -> exit 3

For every EXECUTOR block the manager pastes the prompt into the executor
pane, waits for the pane to settle, then pastes the captured reply back to
the controller as:

    EXECUTOR_REPLY:
    <captured output>
    END_EXECUTOR_REPLY

Assumptions
-----------
Both tmux sessions are already attached to their respective coding-agent
prompts (claude code / codex) and ready to accept input.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
SENTINEL_RE = re.compile(
    r"^\s*(?P<tag>STATUS|NOTE|DONE|BLOCKED|EXECUTOR)\s*:\s*(?P<body>.*)$"
)
END_EXECUTOR = "END_EXECUTOR"

# Patterns that identify claude-code / codex UI chrome we want to strip out
# of executor pane captures before forwarding the reply to the controller.
# These are intentionally aggressive about box-drawing / spinner glyphs --
# real shell command output rarely contains them, while every CLI agent's UI
# chrome does.
_HORIZONTAL_RULE_RE = re.compile(r"^\s*[─-╿\-=_]{12,}\s*$")
_CHROME_PATTERNS = [
    # claude-code feedback prompt
    re.compile(r"^\s*●?\s*How is Claude doing this session\?"),
    re.compile(
        r"^\s*\d+\s*:\s*(Bad|Fine|Good|Dismiss)"
        r"(\s+\d+\s*:\s*\w+)*\s*$"
    ),
    # claude-code permission-mode hint and token-usage hint
    re.compile(r"⏵⏵\s+bypass permissions"),
    re.compile(r"new task\?\s+/clear to save"),
    re.compile(r"shift\+tab to cycle"),
    # claude-code thinking spinner: "✻ Crunched for 9m 30s",
    # "✶ Tinkering for 12s", "✳ Pondering ..." etc. claude rotates through
    # several dingbat glyphs; cover the ones it actually uses.
    re.compile(r"^\s*[✱✲✳✴✵✶✷✸✹✺✻✼✽✾✿❀❁❂❃✦✧·\*]\s+\S.*$"),
    # empty input-box prompt symbols
    re.compile(r"^\s*[❯›]\s*$"),
    # codex header / placeholder lines
    re.compile(r"^\s*[❯›]\s+(Improve documentation|Try |Ask |Type )"),
    re.compile(r"^\s*gpt-[0-9.]+\s+\w+\s+·\s+"),
    re.compile(r"^\s*>_\s+OpenAI Codex"),
    # any line that begins with a box-drawing vertical/corner glyph
    re.compile(r"^\s*[│┃┌┐└┘├┤"
               r"┬┴┼═║╔╗╚╝"
               r"╭╮╯╰]"),
]


def clean_pane_text(text: str) -> str:
    """Drop UI chrome (separators, spinners, feedback prompts) from a captured
    pane so we don't poison the controller's context when forwarding."""
    kept: list[str] = []
    for line in text.splitlines():
        if _HORIZONTAL_RULE_RE.match(line):
            continue
        if any(p.search(line) for p in _CHROME_PATTERNS):
            continue
        kept.append(line.rstrip())
    # collapse runs of blank lines to a single blank
    collapsed: list[str] = []
    blanks = 0
    for ln in kept:
        if not ln.strip():
            blanks += 1
            if blanks <= 1:
                collapsed.append("")
        else:
            blanks = 0
            collapsed.append(ln)
    while collapsed and not collapsed[0].strip():
        collapsed.pop(0)
    while collapsed and not collapsed[-1].strip():
        collapsed.pop()
    return "\n".join(collapsed)

KICKOFF = """\
You are now CONTROLLER in an agent-manager loop. A separate program (the
manager) reads this pane and will keep you running until the goal is met
or determined unreachable.

GOAL:
{goal}

PROTOCOL -- produce single lines that begin (at column 0) with one of the
tags below, followed immediately by a colon and a space. The manager
parses these lines from the pane.

  Tag         Purpose
  ----------  ----------------------------------------------------------
  STATUS      brief progress update (one short line)
  NOTE        one short line worth remembering long-term
  EXECUTOR    prompt for the executor agent (multi-line allowed; close
              the block with a line whose only content is the closing
              marker shown below)
  DONE        goal achieved -- ends the loop
  BLOCKED     fundamental limitation, cannot proceed -- ends the loop

The closing marker for an EXECUTOR block is the literal string {end} on
its own line.

After every EXECUTOR block the manager will forward the prompt to the
executor tmux session, capture its output, and paste it back to you
between the markers EXECUTOR_REPLY and END_EXECUTOR_REPLY.

Rules:
  - This is an endless loop. Only DONE or BLOCKED ends it.
  - Prefer many small EXECUTOR steps over one giant prompt.
  - After every reply from the executor, evaluate, emit one STATUS line,
    then either another EXECUTOR block or DONE/BLOCKED.
  - Be terse. STATUS/NOTE lines go into a global memory file -- write as
    little as possible while staying useful.
  - Make concise suggestions, as necessary as needed and as minimal as possible.
  - Tagged lines must start at column 0; no leading prose on the same
    line as a tag.

Begin: emit one STATUS line, then your first EXECUTOR block.
"""

NUDGE = (
    "(manager nudge) Continue the loop. Emit STATUS then "
    "EXECUTOR/DONE/BLOCKED."
)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def tmux_session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], capture_output=True
    ).returncode == 0


def tmux_capture(session: str, lines: int) -> str:
    r = sh(["tmux", "capture-pane", "-t", session, "-p", "-J", "-S", f"-{lines}"])
    return strip_ansi(r.stdout)


def tmux_send_text(session: str, text: str) -> None:
    """Paste text into the pane (bracketed paste if supported), then Enter."""
    fd, path = tempfile.mkstemp(prefix="agentmgr.", suffix=".buf")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        sh(["tmux", "load-buffer", "-b", "agentmgr", path])
        # -p: bracketed paste so multi-line content is treated as one paste
        sh(["tmux", "paste-buffer", "-p", "-b", "agentmgr", "-t", session])
        time.sleep(0.25)
        sh(["tmux", "send-keys", "-t", session, "Enter"])
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@dataclasses.dataclass
class Event:
    tag: str
    body: str
    raw: tuple[str, ...]


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def parse_events(pane: str, seen: set[str]) -> list[Event]:
    """Walk pane top-down; return new sentinel events in order."""
    raw = pane.splitlines()
    out: list[Event] = []
    i = 0
    while i < len(raw):
        line = raw[i].rstrip()
        m = SENTINEL_RE.match(line)
        if not m:
            i += 1
            continue
        tag = m.group("tag")
        body = m.group("body").rstrip()

        if tag == "EXECUTOR":
            collected = [line]
            payload: list[str] = []
            if body:
                payload.append(body)
            j = i + 1
            closed = False
            auto_closed = False
            while j < len(raw):
                stripped = raw[j].strip()
                if stripped == END_EXECUTOR:
                    collected.append(raw[j])
                    closed = True
                    break
                # Auto-close on a new sentinel: prevents an unterminated
                # EXECUTOR block from swallowing later DONE/BLOCKED lines.
                if SENTINEL_RE.match(raw[j]):
                    auto_closed = True
                    break
                collected.append(raw[j])
                payload.append(raw[j])
                j += 1
            if not closed and not auto_closed:
                # End-of-buffer mid-block; retry next poll for an END marker.
                break
            ev = Event(tag, "\n".join(payload).strip(), tuple(collected))
            i = j + 1 if closed else j
        else:
            ev = Event(tag, body.strip(), (line,))
            i += 1

        h = _hash("\n".join(ev.raw))
        if h in seen:
            continue
        seen.add(h)
        out.append(ev)

    # Bound the seen-set so it can't grow without limit on long runs.
    if len(seen) > 4096:
        # rebuild keeping only the most recent ~half
        keep = list(seen)[-2048:]
        seen.clear()
        seen.update(keep)
    return out


def relay_executor(
    executor: str,
    prompt: str,
    capture_lines: int,
    settle_seconds: float,
    hard_timeout: float,
) -> str:
    pre = tmux_capture(executor, capture_lines)
    pre_count = len(pre.splitlines())
    tmux_send_text(executor, prompt)
    poll = 1.0
    stable = 0.0
    last = ""
    deadline = time.monotonic() + hard_timeout
    while time.monotonic() < deadline:
        time.sleep(poll)
        cur = tmux_capture(executor, capture_lines)
        if cur == last and cur != "":
            stable += poll
            if stable >= settle_seconds:
                break
        else:
            stable = 0.0
            last = cur
    cur_lines = last.splitlines()
    new = cur_lines[pre_count:] if len(cur_lines) > pre_count else cur_lines
    return clean_pane_text("\n".join(new))


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def memory_append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Agent Manager Memory\n")
    with path.open("a") as f:
        f.write(line.rstrip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Drive a controller<->executor agent loop in tmux.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--goal", required=True, help="Goal the controller must achieve.")
    ap.add_argument("--controller", required=True, help="tmux session of controller agent.")
    ap.add_argument("--executor", required=True, help="tmux session of executor agent.")
    ap.add_argument("--memory", default="./MEMORY.md", help="Path to memory markdown.")
    ap.add_argument("--poll-interval", type=float, default=5.0,
                    help="Seconds between controller-pane reads.")
    ap.add_argument("--nudge-after", type=float, default=90.0,
                    help="Re-prompt the controller if no new sentinel for this long.")
    ap.add_argument("--executor-settle", type=float, default=8.0,
                    help="Seconds of pane-stable that mark the executor reply complete.")
    ap.add_argument("--executor-timeout", type=float, default=900.0,
                    help="Hard cap on a single executor turn (seconds).")
    ap.add_argument("--capture-lines", type=int, default=4000,
                    help="Pane capture depth (lines).")
    ap.add_argument("--max-iters", type=int, default=0,
                    help="Stop after this many polls (0 = unbounded).")
    ap.add_argument("--no-kickoff", action="store_true",
                    help="Skip sending the kickoff prompt (use when resuming).")
    args = ap.parse_args()

    for name in (args.controller, args.executor):
        if not tmux_session_exists(name):
            print(f"error: tmux session not found: {name}", file=sys.stderr)
            return 2
    if args.controller == args.executor:
        print("error: controller and executor must be different sessions",
              file=sys.stderr)
        return 2

    memory_path = Path(args.memory).resolve()
    seen: set[str] = set()

    memory_append(memory_path, f"\n## run {now_iso()}")
    memory_append(memory_path, f"- goal: {args.goal}")
    memory_append(memory_path,
                  f"- controller: {args.controller}  executor: {args.executor}")

    interrupted = {"flag": False}

    def _sigint(signum, frame):
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    if not args.no_kickoff:
        print(f"[manager] kickoff -> controller `{args.controller}`")
        tmux_send_text(args.controller, KICKOFF.format(goal=args.goal, end=END_EXECUTOR))
        # Let the kickoff settle into the pane before baselining.
        time.sleep(2.0)

    # Baseline: any sentinel-looking lines already in the pane (kickoff echo,
    # prior history, the user's own goal text) must NOT replay. Pre-fill the
    # `seen` set with everything that currently parses as an event.
    baseline_pane = tmux_capture(args.controller, args.capture_lines)
    pre_existing = parse_events(baseline_pane, seen)
    if pre_existing:
        print(f"[manager] baseline-skipped {len(pre_existing)} pre-existing "
              f"sentinel lines in controller pane")

    last_event_t = time.monotonic()
    last_status = ""
    iters = 0
    rc = 1
    try:
        while True:
            if interrupted["flag"]:
                memory_append(memory_path, f"- {now_iso()} INTERRUPTED")
                print("[manager] interrupted")
                rc = 130
                break
            if args.max_iters and iters >= args.max_iters:
                memory_append(memory_path, f"- {now_iso()} terminated: max-iters")
                print("[manager] hit max-iters")
                break
            iters += 1
            time.sleep(args.poll_interval)

            pane = tmux_capture(args.controller, args.capture_lines)
            events = parse_events(pane, seen)

            if not events:
                if time.monotonic() - last_event_t > args.nudge_after:
                    print("[manager] nudging controller")
                    tmux_send_text(args.controller, NUDGE)
                    last_event_t = time.monotonic()
                continue

            terminated = False
            for ev in events:
                last_event_t = time.monotonic()
                ts = now_iso()
                if ev.tag == "STATUS":
                    if ev.body != last_status:
                        memory_append(memory_path, f"- {ts} STATUS: {ev.body}")
                        last_status = ev.body
                    print(f"[manager] STATUS: {ev.body}")
                elif ev.tag == "NOTE":
                    memory_append(memory_path, f"- {ts} NOTE: {ev.body}")
                    print(f"[manager] NOTE: {ev.body}")
                elif ev.tag == "EXECUTOR":
                    head = ev.body.splitlines()[0][:80] if ev.body else ""
                    print(f"[manager] EXECUTOR -> {args.executor}: {head!r}")
                    reply = relay_executor(
                        args.executor,
                        ev.body,
                        args.capture_lines,
                        args.executor_settle,
                        args.executor_timeout,
                    )
                    tmux_send_text(
                        args.controller,
                        f"EXECUTOR_REPLY:\n{reply}\nEND_EXECUTOR_REPLY",
                    )
                elif ev.tag in ("DONE", "BLOCKED"):
                    memory_append(memory_path, f"- {ts} {ev.tag}: {ev.body}")
                    print(f"[manager] {ev.tag}: {ev.body}")
                    rc = 0 if ev.tag == "DONE" else 3
                    terminated = True
                    break
            if terminated:
                break
    finally:
        memory_append(memory_path, f"- {now_iso()} run-end rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
