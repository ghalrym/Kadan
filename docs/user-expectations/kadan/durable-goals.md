# Ongoing goals

On this branch, Kadan's text assistant can save an ongoing goal with its progress, next step and blocker. Saved goals belong to the conversation and survive closing the browser or restarting the service. This proposed behavior has unit coverage; the project owner's manual validation with the assembled application and model is still pending.

Open the agent's card from its pet to see **Goals** above Activity. Goals distinguish active work, waiting, paused, finished, failed and cancelled. A saved goal is separate from an individual response: finishing a response or sending a message does not prove that the goal or an external action succeeded.

This first version supports one unfinished goal per conversation. Other chat messages can still be sent; they do not establish independent parallel goals. A paused goal must be resumed or cancelled before another goal is saved.

The assistant can wait for your reply, for work it handed off, or until a later check. When it needs an answer, it asks in chat and waits. Reply in the same chat to continue. A timer or completed tool can wake the assistant without another user message. Saved progress is included when it continues, even after service recovery. These wakeups do not guarantee that an unavailable model or computer can complete the next step.

**Pause** stops further goal work until **Resume**. **Cancel goal** ends its continuation. These controls stop the associated response work, but do not undo actions or stop work already handed to the computer, Codex or image generation. Use the corresponding work's own stop control when needed. Check the resulting file or application when confirmation matters.

The card refreshes saved goals periodically while open. A failed refresh leaves the last received goals visible, labels them as the last update and disables changes until a fresh read succeeds. A failed change may have been saved; Kadan checks the saved state before enabling another change. Finished and failed goals remain available under **Finished and failed**, with their recorded progress and blocker. Failed goals are not automatically retried. Choose **Resume** on a failed goal to start a fresh response from its saved progress.

See [Conversation activity](conversation-activity.md) and [Stopping chat work](stopping-chat-work.md).

The assistant can offer [optional progress notices](progress-updates.md) for routine
timer checks. Omitting such a notice does not change the goal's saved state.
