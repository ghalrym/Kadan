# Conversation activity

Kadan shows the current conversation's saved response work and image jobs in the **Activity** section of the agent's card, below **Computers**. Click the agent's pet to open the card. Queued or running work comes first, followed by items that need review (the latest response failure or stop, and image failures or stops), each marked by its status. Completed work is under expandable **Finished**. Response failures and stops followed by a newer response or user message remain under expandable **Earlier**. The section heading counts the items in progress and to review. Earlier does not mean resolved. Each list shows its ten most recently updated items and indicates when more exist. In-progress work has a separate allowance from items to review, so older failures cannot displace it.

Messages repeats the latest response failure or stop until a newer response or user message arrives. Earlier outcomes stay in Activity without accumulating below the latest message. An error on the latest response, an unavailable update, or an unconfirmed send remains visible. Activity shows a compact error preview; expand **Error details** to read the full recorded error, including for failures under **Earlier**. The list scrolls within the card, and **Finished** and **Earlier** start collapsed each time the card opens. The card, and with it Activity, is unavailable while Find is open.

Response completion and message delivery are separate. A finished response can show that no message was sent. An image can be ready but not yet sent, then show as sent to the conversation. These states do not prove that an external file, website or application changed successfully.

Updates arrive periodically and can lag saved changes. Loading and failed reads must remain distinct from no activity. After a refresh failure, previously loaded activity remains visible with an error; it is the last received state. Switching conversations resets Activity to that conversation's evidence.

Activity counts cover response and image records, not Codex jobs. The separate [Codex status](coding.md) in Messages comes from the shared computer. Response work can fail or stop after Codex completes, and stopping a response can leave Codex running.

Activity is a summary of retained work, not the agent's hidden reasoning, command transcript or a complete audit of external effects. Failures and stops remain available for review without being retried automatically. Use [Stop and Stop image](stopping-chat-work.md) for active work; viewing Activity does not cancel, resume or replay it.
