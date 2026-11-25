# Model-facing tool context

Tool definitions share the conversation model's context with instructions,
personas, recent delivered chat and this run's results. This baseline measures
the current definitions without inference or checkpoint downloads; it does not
recommend removing tools.

Kadan sends the enabled tool schemas to Hikari separately from authored messages. `tool_characters` measures their compact JSON representation. Conversation runs reserve that many characters before admitting history or saved tool results. Hikari accounts for its model-specific representation when checking the actual token limit.

| Available capabilities | Tools | Reserved characters |
| --- | ---: | ---: |
| Chat | 18 | 11,422 |
| Chat with a computer | 22 | 14,007 |
| Computer assistant | 14 | 13,736 |

Selecting a saved reference adds one tool and 365 characters to a chat request. The shipped agent instructions occupy 2,463 characters before any persona or selected-reference notice. These counts were measured from the tool registry and the instruction file without inference. A changed file, schema or dependency can change them.

The runner estimates three characters per token and reserves its generation budgets plus template overhead. Conversation history is limited to the most recent 97 authored messages through the triggering turn. The actual number of prompt tokens and the time spent preparing them depend on the model, history, tool results and prefix reuse.

See [conversation context and delivery](agent-conversations.md) for admission
and durable message semantics, and [agent action expectations](../../user-expectations/kadan/agent-actions.md)
for user-visible limits.
