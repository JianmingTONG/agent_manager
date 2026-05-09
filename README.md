# agent_manager

Drives a loop between two coding-agent sessions running in tmux (claude code
or codex), until a goal is achieved or declared unreachable.

```
                +-----------+
   manager -->  | CONTROLLER|  (tmux: --controller)
   reads pane,  +-----------+
   parses sentinels, ^   |
                  reply  | EXECUTOR: ...
                     |   v
                +-----------+
   manager -->  | EXECUTOR  |  (tmux: --executor)
                +-----------+
```

The manager owns the loop. It pastes prompts into tmux panes, captures
output, bridges controller<->executor, and appends a compact log to
`MEMORY.md`.

## Usage

```bash
python3 agent_manager.py \
  --goal "Make tabV.py pass at composite_degree=2" \
  --controller ctrl \
  --executor exec \
  --memory ./MEMORY.md
```

Both tmux sessions must already be at the prompt of their coding agent
(claude or codex), ready to accept input. Different sessions are required
for controller and executor.

## Protocol

The controller is told (via a kickoff prompt) to emit sentinel-prefixed
lines in its pane:

| Sentinel       | Meaning                                                  |
|----------------|----------------------------------------------------------|
| `STATUS: ...`  | one short line of progress; appended to MEMORY.md        |
| `NOTE: ...`    | one short line worth remembering; appended to MEMORY.md  |
| `EXECUTOR: ...`| prompt for executor; multi-line OK; close with `END_EXECUTOR` on its own line |
| `DONE: ...`    | goal achieved; manager exits 0                           |
| `BLOCKED: ...` | fundamental limitation; manager exits 3                  |

For each `EXECUTOR` block the manager pastes the prompt into the executor
pane, waits for the pane to settle, then pastes the captured reply back
into the controller as:

```
EXECUTOR_REPLY:
<captured output>
END_EXECUTOR_REPLY
```

If the controller goes silent for `--nudge-after` seconds (default 90),
the manager pastes a short nudge.

## Flags

```
--goal              goal the controller must achieve            [required]
--controller        tmux session of the controller agent         [required]
--executor          tmux session of the executor agent           [required]
--memory            path to memory markdown          [default ./MEMORY.md]
--poll-interval     seconds between controller-pane reads        [5.0]
--nudge-after       re-prompt controller if silent this long     [90.0]
--executor-settle   pane-stable seconds that mark reply complete [8.0]
--executor-timeout  hard cap on a single executor turn (s)       [900.0]
--capture-lines     pane capture depth                           [4000]
--max-iters         stop after this many polls (0 = unbounded)   [0]
--no-kickoff        skip kickoff prompt (use when resuming)
```

## Exit codes

| code | meaning                |
|------|------------------------|
| 0    | controller said DONE   |
| 1    | hit `--max-iters`      |
| 2    | bad arguments / sessions missing |
| 3    | controller said BLOCKED|
| 130  | SIGINT/SIGTERM         |

## For me

  Usage
  python3 agent_manager.py \
    --goal "..." \
    --controller <ctrl-tmux> \
    --executor <exec-tmux> \
    --memory ./MEMORY.md

  Design (per the choices you confirmed)
  - Manager owns the loop; controller and executor are stateless w.r.t. it.
  - I/O is tmux load-buffer + paste-buffer -p (bracketed paste, so multi-line prompts don't get submitted line-by-line) + send-keys Enter. ANSI-stripped capture-pane -J for reads.
  - Sentinels: STATUS:, NOTE:, EXECUTOR: ... END_EXECUTOR, DONE:, BLOCKED:. Unterminated EXECUTOR blocks auto-close on the next sentinel so DONE/BLOCKED can never be swallowed.
  - Manager bridges every EXECUTOR: block to the executor pane, waits for the pane to settle (--executor-settle, hard cap --executor-timeout), pastes back as
  EXECUTOR_REPLY:\n...\nEND_EXECUTOR_REPLY.
  - MEMORY.md is manager-owned, append-only, dedupes consecutive identical STATUS lines. Nudges fire after --nudge-after (default 90s) of no new sentinel.
  - Exit codes: 0 DONE, 1 max-iters, 2 bad args, 3 BLOCKED, 130 SIGINT.

  Both tmux sessions need to already be at their agent prompt (claude or codex) and ready to accept input — the manager doesn't start them.