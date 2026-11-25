# Computer agent (`kadan-runner`)

The [agent action](../../user-expectations/kadan/agent-actions.md) and
[computer view](../../user-expectations/kadan/computer.md) expectations describe
the user experience. The conversation agent no longer binds computer tools. It
hands work to a second agent that uses the same Hikari language model in its own
long-lived thread, so conversation runs stay short and the user can keep talking
while the computer works.

## Responsibilities

- The conversation agent binds `use_computer(instructions, message)` beside its
  conversation, workspace, memory, persona, and [Remote MCP](mcp-integrations.md) tools. Remote MCP integrations (including T3 Code) stay with the
  conversation agent because coding threads are delegated externally rather than run inside the sandbox.
- The computer agent binds the sandbox tools, `computer_look` and
  `finish_task(summary, succeeded)`. Its instructions come from
  `COMPUTER_AGENT.md` (`KADAN_COMPUTER_INSTRUCTION_FILE`), read before every
  inference rather than snapshotted, since the thread outlives any one task.
- Tsuchi owns tasks, the thread transcript, steering and the thread lease.
  `kadan-runner` owns execution. `kadan-api` exposes only task status and
  requests to the browser; the transcript never leaves the services.

## Handover

`use_computer` works like `generate_image`. It creates a task
keyed by the action ID, delivers `message` and ends the run. A replay after a
restart gets the original task back. Each later conversation context lists
pending tasks so the agent does not resubmit them.

When a task reaches `completed`, `failed` or `cancelled`, Tsuchi queues an
event run for the origin conversation, `kind: "computer"`, with an ID derived
from the task, in the same transaction as the terminal state. A retry can
neither lose the event nor queue it twice. The event run's context ends with
`Computer task event: {...}` giving the status, instructions and summary. Tasks
started from the computer view have no origin, so nobody is woken.

## Thread and tasks

Each agent has one `ComputerThread` and an ordered queue of `ComputerTask`
records. `ComputerTasks` polls Tsuchi every 0.5 seconds (or immediately when
`use_computer` wakes it), and runs one asyncio task per agent with queued work.
It claims the thread lease (60 seconds, renewed every 10) and works through
tasks oldest first, releasing the lease when the queue is empty. It does not use
the per-agent slot that `AgentRunner` holds for conversation runs, so a running
computer task never blocks a chat response. Computer assistant inference runs at
`background` priority so queued interactive chat turns start first between turns;
`computer_look` also disables prefix reuse so screenshot reads do not overwrite
text conversation checkpoints.

Each task has explicit start and outcome records. Commands and results carry a shared call ID. A direct request carries its original user text, and the agent’s delegation reaches the model as a task observation. Commands are persisted before execution, exactly like
conversation runs. After a restart only GET tools repeat, and an unconfirmed
change is reported to the model as uncertain. A block left open by a task
cancelled while no worker held the thread is closed as `stopped` before the next
task starts. Tsuchi upgrades earlier journals once and retains their original action positions. If a running task’s saved actions cannot be interpreted, the runner reports that task as failed before replay.

Finished task blocks are folded promptly into a single history record holding
their last twenty summaries whenever a new task starts and before each
inference, so prior failed or completed tasks do not carry verbatim tool calls
and outputs into subsequent turns. The current task is never folded. If it alone
does not fit, the task fails. Each task has an 80-command limit and two argument
corrections.

## Steering

`POST /api/agents/{key}/computer/requests` forwards to Tsuchi's
`computer-requests`. In one transaction, Tsuchi either attaches the text as
steering for the oldest active task or creates a new task with no origin. A
request sent as a task finishes therefore always lands somewhere. Request IDs
make retries idempotent.

Before every inference the worker appends pending steering as
user messages with their original text and marks it consumed in the same versioned thread
write. Tsuchi refuses to complete a task while steering is unread
(`New steering arrived for this task`). The worker then reports that to the
model as the `finish_task` result and continues. Failed and cancelled tasks
consume their remaining steering.

## Cancellation

`POST /api/agents/{key}/computer/tasks/{id}/cancel` marks the task cancelled
immediately and queues its event. The worker checks the task before each step
and closes the block without another inference. An action already sent to the
sandbox is not interrupted. A late completion for a cancelled task is ignored.

## Storage contract

Tsuchi stores threads, tasks and steering in separate tables, so steering
appends never conflict with the worker's versioned thread writes.

- `PUT /api/agents/{key}/computer-thread` carries the transcript, lease,
  `expected_version`, `worker_id`, `consumed_steering_ids` and an optional task
  transition (`running`, `completed` or `failed`). Claiming requires an expired
  or absent lease and cannot change the transcript. Another worker's live lease
  is refused.
- `GET /api/computer-tasks` lists active tasks across agents, and
  `GET /api/agents/{key}/computer-tasks` those for one agent.
- `POST /api/agents/{key}/computer-tasks` requires the origin run's owning
  worker, as image submission does.

`GET /api/agents/{key}/computer` in `kadan-api` returns the current task, how
many more are queued and how many steering messages are unread. When nothing is
active it also returns the latest task (Tsuchi's `computer-tasks/latest`), so
the panel can show the result of a task that reports to no conversation.

Validation uses unit tests with in-memory Tsuchi, sandbox and inference
boundaries. Real task quality, steering timing and recovery against a live
computer need a manual trial.
