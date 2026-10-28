# Agent actions

Kadan agents can use their configured computer to browse websites, read and change files, and run commands or code during text conversations. Hikari supplies the model's decisions; Kadan executes the actions. Voice calls do not provide these tools.

The current deployment shares one computer across agents. Actions and manual input can affect the same browser, files and processes. Agent separation in Tsuchi does not create separate desktops. Files created on the computer are not automatically added to Tsuchi's file library.

A computer action can fail because the computer, website, program or requested file is unavailable. Kadan returns action errors to the agent so it can decide what to do next. This does not guarantee that it will recover or send a visible explanation. Invalid agent commands can fail the response, and work remains subject to the conversation's command and context limits.

Action results have a limited text size. The current agent cannot inspect screenshot pixels through its text model, even though users can view the computer. A delivered chat message does not prove that an external change succeeded; inspect the resulting page, file or application when confirmation matters.

An interrupted or timed-out action may already have happened. After a runner restart, Kadan does not automatically replay an unconfirmed action that could change state; the agent is told to check before retrying. Stopping a response does not undo actions or guarantee that an already launched external process has stopped.

See [Viewing and controlling the computer](computer.md), [Stopping chat work](stopping-chat-work.md) and [operator responsibility](../security.md).
