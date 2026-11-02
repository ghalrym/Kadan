# Durable goals

Tsuchi owns one unfinished assistant goal per conversation. The goal stores verified
progress, next step, blocker and an explicit state separately from the response
run that advances it. Kadan exposes the records and controls in the agent card;
the capable conversation model remains responsible for planning and checking work.
See [Ongoing goals](../../user-expectations/kadan/durable-goals.md) for the proposed
user experience and validation limits.

`checkpoint_task` creates or updates a goal under the current run's owner/lease.
The stable command action ID identifies an immutable checkpoint receipt. Tsuchi
saves the mutation, optional visible message and delivery receipt in one write
transaction. A repeated command returns the recorded result; altered content
conflicts. Updates compare the goal version, so user controls and later progress
cannot be silently overwritten. Task data enters context as saved evidence,
without replacing current user instructions.

Active checkpoints continue the bounded response. Waiting and terminal checkpoints
end it. Delegating computer, Codex or image work saves a tool wait before submission.
The existing action IDs and uncertain-effect recovery rules still apply. New user
messages and completion events link subsequent runs to the goal. Unconfirmed
external effects are inspected rather than blindly replayed. A response ending
while its goal remains active marks the goal failed for review; it never implies
that the goal succeeded.

The runner's existing reconciliation calls Tsuchi's timer sweep. A storage write
transaction checks due waits and existing linked queued/running work, creates a
stable event run and advances the goal together. Repeated sweeps and independent
runners cannot enqueue duplicates. Timers require at least thirty seconds between
scheduling and eligibility, and never run inference while a user answer is needed.
Existing per-conversation run ordering, runner assignments and leases fence
overlap. A recovered running response retains its wait rather than treating
recovery as a new wakeup.

Pause/cancel fence linked queued and running runs transactionally; they do not
undo or cancel already delegated external work. Later tool results are retained
without executing a paused, cancelled or terminal goal. Resume queues a fresh run
from saved progress and retained outcome evidence, leaving old commands fenced.
Stopping a linked chat response pauses its goal. Browser polling does not drive
work or replay a failed control request.

Goal rows and checkpoint receipts share Tsuchi's SQLite store and complete backup.
Namespace exports include goal records but are not an executable restore/import
format. Matching Kadan and Tsuchi changes must be deployed together; no deployed
state or inference was exercised by this implementation's unit checks.
