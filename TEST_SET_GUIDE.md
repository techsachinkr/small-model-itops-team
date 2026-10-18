# Writing the test set (task 5.1)

The test set was to be written by humans, and Claude never sees it (CLAUDE.md hard rule 4). None had been written by
2026-10-18, so the user changed rule 4: a fresh AI subagent, briefed with this guide and the helper scripts only, wrote
it, and it is disclosed as AI-written wherever results are reported. The session that built the system hasn't seen
it. Claude wrote this guide and the helper scripts from the code, the fixtures and the practice cases' format, without
any test case.

## Files and size

The final run (`eval/run_eval.py --split test --final`) reads the same layout as `data/practice/`, one case per line:

| file | cases | kinds |
|---|---|---|
| `data/test/helpdesk.jsonl` | 10 (fewer if time is short) | single-API helpdesk: `single`, plus `no_tool`, `missing_info`, `read_only` |
| `data/test/ops.jsonl` | 10 | single-API ops, the same kinds |
| `data/test/multi.jsonl` | 15 | multi-API (`apis: ["helpdesk", "ops"]`): `multi`, plus the other kinds |

About 20% of each file should be `no_tool` or `missing_info`, including a risky request where the right move is to
refuse or ask (like "roll back everything in prod"), and a few `read_only`. Write the drafts outside the repo (for
example `~/test_drafts/helpdesk.yaml`, `ops.yaml` and `multi.yaml`), check them, and only then move the JSONL files
the checker writes into `data/test/`.

## One case

Write each file as YAML, a list of cases (easier by hand than one JSON object per line); the checker turns it into
the JSONL that `data/test/` needs. A skeleton with placeholders (every `<...>` is yours to fill):

```yaml
- id: te-hd-01
  request: "<what the user writes, in quotes>"
  user_id: <E10xx from the fixture overview>
  fixture: default            # or an incident fixture
  apis: [helpdesk]            # [ops] in ops.yaml, [helpdesk, ops] in multi.yaml
  kind: single                # single, multi, no_tool, missing_info or read_only
  gold_calls:                 # empty ([]) for no_tool and missing_info
    - {api: helpdesk, tool: <tool>, args: {<argument>: <value>}}
  answer_must_contain: ["<...>"]          # needed for read_only; optional otherwise
  answer_must_not_contain: ["<...>"]      # optional
```

Quote the request (it often has colons). `try_calls.py` (below) shows what each gold call returns on the fixture, like
the ID a new ticket gets and the strings a read-only answer must contain.

- **id**: unique, and not a practice ID (`pr-...`): for example `te-hd-01`, `te-ops-01`, `te-multi-01`.
- **request**: the user's words. **user_id**: the acting user (the prompt says "Current user: <id>"). **fixture**:
  `default` or an incident variant. Print what each holds with `fixture_overview.py` below.
- **gold_calls**: what a correct agent does, in order, look-ups included (a device before a hardware ticket, the
  rollout history before a rollback). The verifier replays them on a fresh copy of the fixture and passes a run only
  if its final state matches: look-ups change nothing, so only the changes are compared. Summaries and notes only need
  to be there; a ticket the gold creates or updates needs at least the gold's number of notes.
- A new ticket gets the fixture's next ID (`fixture_overview.py` prints it, like HD-1042 on `default`). A later gold
  call that uses it, such as a prod change's `change_ticket`, names that ID.
- Any rollback or scale in `prod` needs `change_ticket`: an open (`open` or `in_progress`) helpdesk ticket's ID.
  Staging needs none.
- **kind**: `single` (one API, changes something), `multi` (two APIs), `no_tool` (no gold calls: decline, or answer
  without tools), `missing_info` (no gold calls: the agent must ask), `read_only` (gold calls only look things up;
  needs `answer_must_contain`, strings those look-ups return).
- **answer_must_contain** and **answer_must_not_contain**: short substrings, matched ignoring case (and look-alike
  hyphens and spaces). For example the device model for a read-only question, or "namespace" for a question the
  agent must ask. `answer_must_not_contain` catches over-inclusive answers, such as both devices when one was asked.

## So that a correct agent can pass

- Pin down every field the gold sets. If the gold opens a P1 `incident` ticket, the request must make P1 and an
  incident clear. Categories as the model sees them: hardware is a physical device (it needs `device_id`), software is
  an application problem on the user's machine, access is accounts, logins and permissions, network is VPN, Wi-Fi and
  connectivity, and incident is an outage or failure of a shared or production service.
- One right answer only: if two revisions, tickets or namespaces would do, say which, or make it `missing_info`.
- Don't copy or reword practice cases (`data/practice/`) or generated tasks. Step 3
  (`eval/decontaminate.py --against test`) flags near-copies, and tasks with the same changes as a test case.

## The helpers (run them yourself; they never print anything from `data/test/`)

```bash
.venv/bin/python scripts/checks/fixture_overview.py                    # the fixtures
.venv/bin/python scripts/checks/fixture_overview.py default incident_prod_payments   # who and what is in them
.venv/bin/python scripts/checks/try_calls.py default list_devices '{"employee_id": "E1003"}'   # what calls return
.venv/bin/python scripts/checks/check_cases.py ~/test_drafts/*.yaml             # check the drafts
.venv/bin/python scripts/checks/check_cases.py ~/test_drafts/*.yaml --jsonl     # all ok: write the .jsonl files
```

`check_cases.py` checks what `tests/test_practice_cases.py` checks for the practice cases: each case parses, its
gold calls run, a gold run passes the verifier, doing nothing fails exactly the cases that ask for a change, and a
read-only case's strings come back from its look-ups. It also checks unique IDs, which cases go in which file, and
the mix of kinds. It refuses files in `data/test/`. Its output, and `try_calls.py`'s, shows your cases: don't paste it
to Claude.

When every case is `ok`, write the JSONL files with `--jsonl`, move them to `data/test/`, commit, and run 5.1 step 3:

```bash
mkdir -p data/test && mv ~/test_drafts/{helpdesk,ops,multi}.jsonl data/test/
uv run python -m eval.decontaminate --against test --tasks data/generated/helpdesk/tasks_round_1.jsonl data/generated/ops/tasks_round_1.jsonl
```

## Tools (arguments; `?` is optional)

- helpdesk: `list_devices(employee_id)`, `create_ticket(employee_id, category, summary, priority, device_id?)`,
  `list_tickets(employee_id, status?)`, `get_ticket(ticket_id)`, `update_ticket(ticket_id, status?, priority?, note?)`,
  `reset_password(employee_id, delivery)`. Priorities P1 to P4; statuses open, in_progress, resolved; delivery email
  or sms.
- ops: `list_deployments(namespace)`, `get_pods(namespace, deployment)`, `get_logs(namespace, pod, tail?)`,
  `rollout_history(namespace, deployment)`, `rollback(namespace, deployment, to_revision, change_ticket?)`,
  `scale(namespace, deployment, replicas, change_ticket?)`. Namespaces prod and staging.
