# Optional progress decisions

`send_progress_update` is a conversation tool for optional routine timer notices.
It does not end the response, choose a plan or mutate the goal. `send_message` and
atomic checkpoint messages remain unfiltered. See [Optional progress
notices](../../user-expectations/kadan/progress-updates.md) for user-visible limits.

The hard gate requires an actual saved timer event for the current goal, no direct
user or image trigger, no retained completion evidence, an active/waiting goal
without a blocker, and explicit `importance=routine` plus
`requested_by_user=false`. Both fields default to delivery. Completion/failure
events and explicit requests bypass this gate, even if the specialist would have
returned suppression. Correctly recognizing textual importance still depends on
the capable assistant's tool choice; this is not a semantic proof of urgency.

Eligible notices use Hikari's existing `/v1/decisions` Choice contract, fixed d1
worker and resource admission. The input contains complete goal/progress/next-step
evidence, the last delivered assistant text and proposed notice. Evidence beyond
the existing 8,000-character state limit sends without truncation. One exact typed
answer must choose `suppress`, with confidence and suppression probability both at
least 0.98, complete probability keys and a valid distribution. Invalid, ambiguous,
unavailable or timed-out results send. The caller allows two seconds for the
optional read/decision; executor cleanup may continue after cancellation. No speed
or calibration claim is made for d1, whose existing CPU limitations still apply.

Before delivery or suppression, the runner checkpoints the outcome under the
stable command action ID in `AgentRunState.progress_update_decisions`. The map is
bounded by the existing 24-command response limit. Tsuchi's version/owner checks
fence the write, preserve previous entries immutably, and prohibit changing it
during expired-run recovery. The runner uses its existing update lock so heartbeat
version changes cannot race the checkpoint. A replay uses the saved outcome and
ordinary message identity; it neither repeats classification nor duplicates
recorded delivery. Cancellation during classification propagates, and ownership
is checked before saving or delivering. The bounded map is omitted from the
existing public activity summary along with other internal run state.
