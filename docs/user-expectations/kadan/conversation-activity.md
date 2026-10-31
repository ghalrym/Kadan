# Conversation activity

Kadan's chat **Activity** panel shows the current conversation's saved response work and image jobs. It groups the latest response failure or stop and image failures or stops under **Needs review**, queued or running work under **In progress**, and completed work under **Finished**. Response failures and stops followed by a newer response or user message remain under expandable **Earlier failures and stops**. Earlier does not mean resolved. Each group shows its ten most recently updated items and indicates when more exist.

Messages repeats the latest response failure or stop until a newer response or user message arrives. Earlier outcomes stay in Activity without accumulating below the latest message. An error on the latest response, an unavailable update, or an unconfirmed send remains visible. Opening Activity with a pointer or keyboard reveals a bounded, scrollable area; its controls remain accessible beside the composer. Changing the panel's size keeps the latest messages in view while following the conversation, without returning readers who have scrolled away or are using Find to the bottom.

Response completion and message delivery are separate. A finished response can have no recorded message delivery. An image can be ready while awaiting delivery, then show a recorded delivery to the conversation. These states do not prove that an external file, website or application changed successfully.

Updates arrive periodically and can lag saved changes. Loading and failed reads must remain distinct from no activity. After a refresh failure, previously loaded activity remains visible with an error; it is the last received state. Switching conversations resets the panel to that conversation's evidence.

Activity counts cover response and image records, not Codex jobs. The separate [Codex status](coding.md) in Messages comes from the shared computer. Response work can fail or stop after Codex completes, and stopping a response can leave Codex running.

Activity is a summary of retained work, not the agent's hidden reasoning, command transcript or a complete audit of external effects. Failures and stops remain available for review without being retried automatically. Use [Stop and Stop image](stopping-chat-work.md) for active work; viewing Activity does not cancel, resume or replay it.
