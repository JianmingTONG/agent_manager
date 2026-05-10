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
    DONE:    <reason>                            -> exit 0 (terminates)
    BLOCKED: <reason>                            -> resume (does NOT exit):
        the manager scans the recent controller pane for a
        'Recommended next steps' / 'Next' / 'Suggestions' block and, if
        found, forwards it verbatim to the executor as the next prompt and
        keeps looping. If no such block is present, the manager pastes a
        short request back into the controller asking it to produce one.
    USER_INTERVENTION: <reason>                  -> exit 3 (genuinely
        requires the human user -- missing secret, design call, auth)

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
    r"^\s*(?P<tag>STATUS|NOTE|DONE|BLOCKED|EXECUTOR|MEMORY|WAIT|USER_INTERVENTION)"
    r"\s*:\s*(?P<body>.*)$"
)
END_EXECUTOR = "END_EXECUTOR"
END_MEMORY = "END_MEMORY"

# Multi-line block tags: tag -> (closing marker, auto_close_on_sentinel).
# auto_close_on_sentinel keeps the parser robust against a controller that
# forgets the explicit closer; for MEMORY we keep that same behaviour because
# the controller is told (in KICKOFF) that tagged lines must start at column
# 0, so authentic markdown notes inside a MEMORY block won't accidentally
# trigger a false close.
_MULTILINE_BLOCKS = {
    "EXECUTOR": (END_EXECUTOR, True),
    "MEMORY":   (END_MEMORY,   True),
}

# Markers that delimit the controller-owned region of the memory file. The
# controller rewrites the contents between these markers via MEMORY blocks;
# the manager's auto-log lives below the end marker, untouched.
MEMORY_BEGIN_MARK = "<!-- begin:controller-memory -->"
MEMORY_END_MARK = "<!-- end:controller-memory -->"

# Heuristic for spotting a "Recommended / Next steps / Suggestions" block in
# the controller's pane after a BLOCKED. The header line is matched; we then
# extend through subsequent non-blank, non-sentinel lines.
_SUGGESTION_HEADER_RE = re.compile(
    r"(?i)\b("
    r"recommended(\s+next\s+steps?)?"
    r"|next\s+(steps?|actions?|moves?|ideas?)"
    r"|suggestions?"
    r"|options?\s+to\s+(try|consider|unblock)"
    r"|ideas?\s+to\s+(try|unblock)"
    r")\b"
)

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
You are the CONTROLLER in an agent-manager loop. A separate program (the
manager) reads this pane and keeps you running until the goal is met or
declared unreachable. A second coding agent -- the EXECUTOR -- does the
actual implementation work in a different tmux session; you do not edit
files or run commands yourself.

GOAL:
{goal}

YOUR ROLE -- high-level management, not step-by-step instruction.
You plan and supervise; the executor figures out the implementation.
On every iteration of the loop:

  1. UNDERSTAND STATUS. Read the latest EXECUTOR_REPLY and the prior
     STATUS/NOTE lines in this pane. Decide what is actually true now:
     what has landed, what is failing, what is still unknown. Do not
     assume the previous plan still holds.
  2. BREAK DOWN THE GOAL into the next sub-goal under the current
     status. Maintain the full decomposition in your own head, but pick
     the smallest sub-goal that meaningfully advances the plan from
     where things actually are.
  3. SUGGEST DIRECTION, DO NOT DICTATE STEPS. Give the executor a
     sub-goal plus rationale, constraints, and improvement directions
     (what to optimise for, pitfalls to avoid, options worth
     considering). Let the executor choose files, commands, and
     approach. Do NOT enumerate shell commands or paste exact diffs
     unless the sub-goal genuinely is "run this exact command".
  4. UPDATE THE PLAN DYNAMICALLY. When the executor's reply changes
     what you thought was true, revise the plan. Record the new
     picture as a STATUS line and any durable insight as a NOTE line.

EXTERNAL MEMORY. The manager keeps a durable memory file for you at:

  {memory_path}

It has two zones, and you should treat them differently.

  1. A controller-owned region delimited by these markers in the file:

       <!-- begin:controller-memory -->
       ...your curated notes...
       <!-- end:controller-memory -->

     This region is YOURS. Shape it however serves you best -- a
     decisions log, an open-questions list, a fact table, a glossary,
     a plan tree, a freeform brief. Read the file at the start of a
     session if you're picking up prior work, then update the region
     by emitting:

         MEMORY:
         <freeform markdown -- whatever structure you prefer>
         {end_memory}

     The block REPLACES everything between the two markers, so include
     the full curated state you want preserved (not a delta). Re-emit
     MEMORY whenever the picture meaningfully changes; you can do so
     as often as you like.

  2. An auto-appended audit log below the end marker. The manager
     timestamps and appends every STATUS / NOTE / DONE / BLOCKED line
     you emit, plus run boundaries and executor handoffs. You cannot
     edit this part; treat it as a write-only ledger the manager
     maintains for traceability.

Do not rely on chat scrollback to remember the plan: re-emit a fresh
STATUS whenever the plan shifts, capture hard-won facts as NOTE lines
so they survive context loss, and use MEMORY blocks to keep the
curated region a faithful summary of where things stand.

PROTOCOL -- produce single lines that begin (at column 0) with one of
the tags below, followed immediately by a colon and a space. The
manager parses these lines from the pane.

  Tag                Purpose
  -----------------  ---------------------------------------------------
  STATUS             one short line: current picture + next sub-goal
  NOTE               one short line of durable insight worth remembering
  EXECUTOR           high-level sub-goal for the executor (multi-line
                     allowed; close the block with a line whose only
                     content is the closing marker shown below)
  MEMORY             freeform markdown that REPLACES the controller-
                     owned region of the memory file. Multi-line; close
                     the block with the MEMORY closing marker shown
                     below. See the EXTERNAL MEMORY section above.
  WAIT               ask the manager to suppress its inactivity-nudge
                     for the given duration. Body is a number (seconds)
                     or a value with a unit suffix: ms / s / m / h
                     (e.g. 'WAIT: 90', 'WAIT: 5m', 'WAIT: 1h'). The
                     manager keeps polling the pane for new sentinels;
                     it just stops sending the periodic nudge until the
                     deadline passes or you emit any new event.
                     Subsequent WAIT replaces the prior deadline. Use
                     this when you need time to think, or when an
                     external process must finish before the next
                     sub-goal is meaningful.
  DONE               goal achieved -- ends the loop
  BLOCKED            soft signal: you are stuck on the current sub-goal
                     and want a different angle. Pair this sentinel with
                     a 'Recommended next steps:' (or 'Next steps:' /
                     'Suggestions:') section listing concrete ideas the
                     executor could try. The manager scans the recent
                     pane for that section and, if present, forwards it
                     verbatim to the executor as the next prompt and the
                     loop continues. If no such section is present, the
                     manager will paste a request back here asking you
                     for one. BLOCKED does NOT end the loop.
  USER_INTERVENTION  hard escape: the goal genuinely requires the human
                     user (missing secret, missing access, design call,
                     external decision). Use this sparingly. Ends the
                     loop and surfaces control to the user.

The closing markers (each on its own line, no surrounding prose) are:
  {end}        -- closes an EXECUTOR block
  {end_memory} -- closes a MEMORY block

After every EXECUTOR block the manager forwards the prompt to the
executor tmux session, captures its output, and pastes it back to you
between the markers EXECUTOR_REPLY and END_EXECUTOR_REPLY.

EXECUTOR PROMPT STYLE -- write sub-goals, not scripts.
  - Frame the work as: <sub-goal>. Context: <what's true now>.
    Constraints: <must / must-not>. Success looks like: <observable
    signal>. Directions to consider: <options, trade-offs>.
  - One focused sub-goal per block; let the executor decompose it.
  - If a reply is ambiguous, ask the executor to verify, summarise, or
    investigate -- do not redo the work in your own head.

RULES
  - This is an endless loop. Only DONE or USER_INTERVENTION ends it;
    BLOCKED triggers a resume (see above) and the loop continues.
  - After every executor reply: emit one STATUS line (updated picture +
    next sub-goal), then either an EXECUTOR block or DONE / BLOCKED /
    USER_INTERVENTION.
  - When you emit BLOCKED, also write a short 'Recommended next steps:'
    list immediately above or below it -- 1-5 concrete bullets the
    executor can try. The manager will forward those bullets to the
    executor verbatim as the next prompt.
  - Be terse. STATUS/NOTE lines are written verbatim to memory -- keep
    them short.
  - Tagged lines must start at column 0; no leading prose on the same
    line as a tag.

Begin: emit one STATUS line summarising the current picture and the
first sub-goal, then your first EXECUTOR block.
"""

NUDGE = (
    "(manager nudge) Continue the loop. Emit STATUS then "
    "EXECUTOR / DONE / BLOCKED / USER_INTERVENTION."
)

RESUME_NO_SUGGESTIONS = (
    "(manager) BLOCKED received but no 'Recommended next steps' / "
    "'Next steps' / 'Suggestions' section was found in your recent "
    "output. Emit one now: a short bulleted list of concrete ideas the "
    "executor could try to break this blocker. The manager will forward "
    "the list to the executor and resume the loop. If the blocker truly "
    "requires the human user (missing secret, missing access, design "
    "call), emit USER_INTERVENTION: <reason> instead."
)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_wait(body: str) -> float | None:
    """Parse a WAIT sentinel body into a positive number of seconds.

    Accepts a bare number (treated as seconds) or one of the unit
    suffixes ``ms`` / ``s`` / ``m`` / ``h``. Whitespace is tolerated.
    Returns None if the body is missing, non-numeric, or non-positive.
    """
    s = body.strip().lower()
    if not s:
        return None
    multiplier = 1.0
    if s.endswith("ms"):
        multiplier, s = 0.001, s[:-2]
    elif s.endswith(("s", "m", "h")):
        multiplier, s = {"s": 1.0, "m": 60.0, "h": 3600.0}[s[-1]], s[:-1]
    try:
        v = float(s.strip())
    except ValueError:
        return None
    if v <= 0:
        return None
    return v * multiplier


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

        if tag in _MULTILINE_BLOCKS:
            closer, auto_close_on_sentinel = _MULTILINE_BLOCKS[tag]
            collected = [line]
            payload: list[str] = []
            if body:
                payload.append(body)
            j = i + 1
            closed = False
            auto_closed = False
            while j < len(raw):
                stripped = raw[j].strip()
                if stripped == closer:
                    collected.append(raw[j])
                    closed = True
                    break
                # Auto-close on a new sentinel: prevents an unterminated
                # block from swallowing later DONE/BLOCKED lines.
                if auto_close_on_sentinel and SENTINEL_RE.match(raw[j]):
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


def extract_suggestions(pane: str, max_tail: int = 120) -> str | None:
    """Look for a 'Recommended next steps' / 'Next steps' / 'Suggestions'
    block in the most recent slice of the controller pane.

    Scans the last ``max_tail`` lines bottom-up. The first line whose text
    matches a suggestion-header pattern (and is not itself a sentinel) is
    treated as the start of the block. Subsequent lines are appended --
    including blank separators and follow-on paragraphs (extra context,
    constraints, caveats the controller wrote after the bulleted list) --
    until a sentinel line or the end of the pane terminates the block.
    Trailing blanks are trimmed and UI chrome (spinners, prompt boxes,
    box-drawing) is stripped via ``clean_pane_text`` so the forwarded
    prompt is just the controller's prose. Returns the joined block text,
    or None if nothing matched.
    """
    lines = pane.splitlines()
    if not lines:
        return None
    start = max(0, len(lines) - max_tail)
    tail = lines[start:]
    for i in range(len(tail) - 1, -1, -1):
        line = tail[i]
        if SENTINEL_RE.match(line):
            continue
        if not _SUGGESTION_HEADER_RE.search(line):
            continue
        collected = [line.rstrip()]
        j = i + 1
        while j < len(tail):
            ln = tail[j].rstrip()
            if SENTINEL_RE.match(ln):
                break
            collected.append(ln)
            j += 1
        while collected and not collected[-1].strip():
            collected.pop()
        text = clean_pane_text("\n".join(collected)).strip()
        if text:
            return text
    return None


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


def _ensure_memory_markers(path: Path) -> None:
    """Make sure the memory file exists and contains a controller-owned
    region delimited by ``MEMORY_BEGIN_MARK`` / ``MEMORY_END_MARK``.

    Creates a fresh file with an empty region if missing, and inserts the
    markers near the top of an existing file that pre-dates this feature.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Agent Manager Memory\n\n"
            f"{MEMORY_BEGIN_MARK}\n{MEMORY_END_MARK}\n"
        )
        return
    text = path.read_text()
    if MEMORY_BEGIN_MARK in text and MEMORY_END_MARK in text:
        return
    lines = text.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#"):
        insert_at = 1
        if insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
    block = f"{MEMORY_BEGIN_MARK}\n{MEMORY_END_MARK}\n\n"
    lines.insert(insert_at, block)
    path.write_text("".join(lines))


def read_controller_memory(path: Path) -> str:
    """Return the contents of the controller-owned region (without the
    surrounding markers), or '' if the file or region is missing."""
    if not path.exists():
        return ""
    text = path.read_text()
    b = text.find(MEMORY_BEGIN_MARK)
    e = text.find(MEMORY_END_MARK)
    if b < 0 or e < 0 or e <= b:
        return ""
    return text[b + len(MEMORY_BEGIN_MARK):e].strip("\n")


def write_controller_memory(path: Path, content: str) -> None:
    """Replace the contents of the controller-owned region with ``content``.

    The auto-log below the end marker is preserved verbatim.
    """
    _ensure_memory_markers(path)
    text = path.read_text()
    b = text.find(MEMORY_BEGIN_MARK)
    e = text.find(MEMORY_END_MARK)
    if b < 0 or e < 0 or e <= b:
        return
    body = content.rstrip()
    new = (
        text[:b + len(MEMORY_BEGIN_MARK)]
        + ("\n" + body + "\n" if body else "\n")
        + text[e:]
    )
    path.write_text(new)


def memory_append(path: Path, line: str) -> None:
    _ensure_memory_markers(path)
    with path.open("a") as f:
        f.write(line.rstrip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Drive a controller<->executor agent loop in tmux.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--goal", help="Goal the controller must achieve "
                    "(inline string). Mutually exclusive with --goal-file.")
    ap.add_argument("--goal-file", help="Path to a file whose contents are "
                    "used as the goal. Mutually exclusive with --goal.")
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
    ap.add_argument("--resume", action="store_true",
                    help="Resume an in-progress controller pane that already "
                         "contains a BLOCKED + suggestion block. Implies "
                         "--no-kickoff. On startup the manager extracts the "
                         "most recent 'Recommended next steps' / 'Next steps' "
                         "/ 'Suggestions' block from the controller pane, "
                         "forwards it to the executor, pastes the reply back "
                         "as EXECUTOR_REPLY, and then enters the normal loop.")
    args = ap.parse_args()
    if args.resume:
        args.no_kickoff = True

    if bool(args.goal) == bool(args.goal_file):
        print("error: provide exactly one of --goal or --goal-file",
              file=sys.stderr)
        return 2
    if args.goal_file:
        try:
            args.goal = Path(args.goal_file).read_text().strip()
        except OSError as e:
            print(f"error: cannot read --goal-file {args.goal_file}: {e}",
                  file=sys.stderr)
            return 2
        if not args.goal:
            print(f"error: --goal-file {args.goal_file} is empty",
                  file=sys.stderr)
            return 2

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
        tmux_send_text(args.controller, KICKOFF.format(
            goal=args.goal,
            memory_path=str(memory_path),
            end=END_EXECUTOR,
            end_memory=END_MEMORY,
        ))
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

    # One-shot resume: the user pointed us at an existing controller pane that
    # already has a BLOCKED + suggestion block in it. The baseline above just
    # marked that BLOCKED as 'seen', so the normal loop would never trigger
    # the resume branch. Do it explicitly here, once, before entering the
    # poll loop.
    if args.resume:
        suggestions = extract_suggestions(baseline_pane)
        if suggestions:
            head = suggestions.splitlines()[0][:80]
            ts = now_iso()
            memory_append(
                memory_path,
                f"- {ts} RESUME: forwarding suggestions to executor: "
                f"{head!r}",
            )
            print(
                f"[manager] resume: forwarding suggestions to executor "
                f"({len(suggestions)} chars)"
            )
            reply = relay_executor(
                args.executor,
                suggestions,
                args.capture_lines,
                args.executor_settle,
                args.executor_timeout,
            )
            tmux_send_text(
                args.controller,
                f"EXECUTOR_REPLY:\n{reply}\nEND_EXECUTOR_REPLY",
            )
        else:
            ts = now_iso()
            memory_append(
                memory_path,
                f"- {ts} RESUME: no suggestions found, "
                f"asking controller to produce one",
            )
            print(
                "[manager] resume: no Recommended/Next/Suggestions block "
                "found in controller pane; asking controller for one"
            )
            tmux_send_text(args.controller, RESUME_NO_SUGGESTIONS)

    last_event_t = time.monotonic()
    last_status = ""
    wait_until = 0.0
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
                now = time.monotonic()
                if now < wait_until:
                    continue
                if now - last_event_t > args.nudge_after:
                    print("[manager] nudging controller")
                    tmux_send_text(args.controller, NUDGE)
                    last_event_t = now
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
                elif ev.tag == "MEMORY":
                    write_controller_memory(memory_path, ev.body)
                    n_lines = len(ev.body.splitlines())
                    memory_append(
                        memory_path,
                        f"- {ts} MEMORY: controller rewrote curated "
                        f"region ({n_lines} lines)",
                    )
                    print(
                        f"[manager] MEMORY: rewrote curated region "
                        f"({n_lines} lines, {len(ev.body)} chars)"
                    )
                elif ev.tag == "WAIT":
                    secs = parse_wait(ev.body)
                    if secs is None:
                        memory_append(
                            memory_path,
                            f"- {ts} WAIT: invalid duration "
                            f"{ev.body!r} -- ignoring",
                        )
                        print(
                            f"[manager] WAIT: invalid duration "
                            f"{ev.body!r}; ignoring"
                        )
                    else:
                        wait_until = time.monotonic() + secs
                        memory_append(
                            memory_path,
                            f"- {ts} WAIT: suppressing nudges for "
                            f"{secs:g}s",
                        )
                        print(
                            f"[manager] WAIT: suppressing nudges for "
                            f"{secs:g}s"
                        )
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
                elif ev.tag == "DONE":
                    memory_append(memory_path, f"- {ts} DONE: {ev.body}")
                    print(f"[manager] DONE: {ev.body}")
                    rc = 0
                    terminated = True
                    break
                elif ev.tag == "USER_INTERVENTION":
                    memory_append(
                        memory_path,
                        f"- {ts} USER_INTERVENTION: {ev.body}",
                    )
                    print(f"[manager] USER_INTERVENTION: {ev.body}")
                    rc = 3
                    terminated = True
                    break
                elif ev.tag == "BLOCKED":
                    suggestions = extract_suggestions(pane)
                    if suggestions:
                        head = suggestions.splitlines()[0][:80]
                        memory_append(
                            memory_path,
                            f"- {ts} BLOCKED: {ev.body} -- "
                            f"resuming via suggestions: {head!r}",
                        )
                        print(
                            f"[manager] BLOCKED: {ev.body} -- "
                            f"forwarding suggestions to executor "
                            f"({len(suggestions)} chars)"
                        )
                        reply = relay_executor(
                            args.executor,
                            suggestions,
                            args.capture_lines,
                            args.executor_settle,
                            args.executor_timeout,
                        )
                        tmux_send_text(
                            args.controller,
                            f"EXECUTOR_REPLY:\n{reply}\nEND_EXECUTOR_REPLY",
                        )
                    else:
                        memory_append(
                            memory_path,
                            f"- {ts} BLOCKED: {ev.body} -- "
                            f"no suggestions found, requesting from controller",
                        )
                        print(
                            f"[manager] BLOCKED: {ev.body} -- "
                            f"no Recommended/Next/Suggestions block found, "
                            f"asking controller for one"
                        )
                        tmux_send_text(args.controller, RESUME_NO_SUGGESTIONS)
            if terminated:
                break
    finally:
        memory_append(memory_path, f"- {now_iso()} run-end rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
