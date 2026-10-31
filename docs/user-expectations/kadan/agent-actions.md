# Agent actions

Kadan agents can use their configured computer to browse websites, read and change files, and run commands or code during text conversations. Hikari supplies the model's decisions; Kadan executes the actions. Voice calls do not provide these tools.

For scoped coding work, the agent can delegate to [Codex coding jobs](coding.md)
on this same computer, with durable progress and separate process cancellation.

The standard deployment uses the released LFM3 BF16 conversation worker with the same saved session, personas, tool permissions and action recovery. Its fixed checkpoint requires a user-approved download; unavailable workers or incomplete checkpoints produce an error. Its bounded context can reject long conversations; tool-use quality and speed need manual evaluation for the work being attempted.

The current deployment shares one computer across agents. Actions and manual input can affect the same browser, files and processes. Agent separation in Tsuchi does not create separate desktops. Files created on the computer are not automatically added to Tsuchi's file library. Agents can explicitly [save completed files into Workspace](file-delivery.md) and [copy stored files onto the computer](opening-workspace-files.md).

A computer action can fail because the computer, website, program or requested file is unavailable. Kadan returns action errors to the agent so it can decide what to do next. This does not guarantee that it will recover or send a visible explanation. Kadan rejects invalid tool arguments before executing the action and gives the agent field-level feedback with up to two correction attempts per response. Repeated invalid arguments fail the response with a diagnostic; rejected argument values are omitted. Unknown tools and malformed commands can still fail immediately. Corrections consume the normal command and context limits and do not reset after successful actions or a runner restart.

Action results have a limited text size. The current agent cannot inspect screenshot pixels through its text model, even though users can view the computer. A delivered chat message does not prove that an external change succeeded; inspect the resulting page, file or application when confirmation matters.

A separate released BF16 LFM3-VL-2B worker lets the agent ask about a fresh screenshot of its configured computer. The standard deployment enables this capability, but its checkpoint still requires a user-approved download. It returns a text observation to the same conversation; it does not control the computer or replace the conversation model. Oversized screenshots, unavailable workers and failed inference report an error. Observations can be inaccurate, and a screenshot describes only the moment captured. Inspect the computer when confirmation matters.

An interrupted or timed-out action may already have happened. After a runner restart, Kadan does not automatically replay an unconfirmed action that could change state; the agent is told to check before retrying. Stopping a response does not undo actions or guarantee that an already launched external process has stopped.

See [Viewing and controlling the computer](computer.md), [Stopping chat work](stopping-chat-work.md) and [operator responsibility](../security.md).
