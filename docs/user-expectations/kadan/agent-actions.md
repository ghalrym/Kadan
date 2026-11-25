# Agent actions

Kadan agents can have their configured computer browse websites, read and change files, and run commands or code during text conversations. The agent does not operate the computer itself. It hands each job to its computer assistant, tells the user what is happening and stays free to keep talking. When the assistant finishes, fails or is stopped, the agent is woken in the same conversation with the assistant's summary and reports the outcome. Hikari supplies both models' decisions with the same language model; Kadan executes the actions. Voice calls do not provide these tools.

The computer assistant has one continuing working thread per agent, so later tasks can build on what earlier ones did. Tasks run one at a time in the order they were handed over or requested. Users can [request or steer its work](computer.md) directly from the computer view. When the thread grows long, the assistant keeps only short summaries of its oldest finished tasks. A single task that does not fit its context, exceeds its action limit or keeps sending invalid arguments fails with an explanation.

For coding work, the agent can start [Codex threads](coding.md) on this same
computer and check on them later.

The standard deployment uses Hikari's native Inkling 2 Mini language worker (or the selected Claude/Codex Language subscription) with the same saved session, personas, tool permissions and action recovery. When using Local, its checkpoint requires a user-approved download; unavailable workers or incomplete checkpoints produce an error. Its bounded context can reject long conversations; tool-use quality and speed need manual evaluation for the work being attempted.

The current deployment shares one computer across agents. Actions and manual input can affect the same browser, files and processes. Agent separation in Tsuchi does not create separate desktops. Files created on the computer are not automatically added to Tsuchi's file library. Agents can explicitly [save completed files into Workspace](file-delivery.md) and [copy stored files onto the computer](opening-workspace-files.md).

A computer action can fail because the computer, website, program or requested file is unavailable. Kadan returns action errors to the agent so it can decide what to do next. This does not guarantee that it will recover or send a visible explanation. Kadan rejects invalid tool arguments before executing the action and gives the model field-level feedback with up to two correction attempts per response or computer task. Repeated invalid arguments fail the response or task with a diagnostic; rejected argument values are omitted. Unknown tools and malformed commands can still fail immediately. Corrections consume the normal command and context limits and do not reset after successful actions or a runner restart.

Action results have a limited text size. The computer assistant uses the active multimodal language model (Inkling 2 Mini locally, or the selected Claude/Codex Language subscription) to inspect a fresh screenshot of the computer. It returns a text observation to the assistant; it does not control the computer. Oversized screenshots, unavailable workers and failed inference report an error. Observations can be inaccurate, and a screenshot describes only the moment captured. Inspect the computer when confirmation matters.

An interrupted or timed-out action may already have happened. After a runner restart, the assistant resumes its task, but Kadan does not automatically replay an unconfirmed action that could change state; the assistant is told to check before retrying. Stopping a chat response does not stop a computer task already handed over; stop it from the computer view. Stopping a task does not undo actions or guarantee that an already launched external process has stopped.

See [Viewing and controlling the computer](computer.md), [Stopping chat work](stopping-chat-work.md) and [operator responsibility](../security.md).
