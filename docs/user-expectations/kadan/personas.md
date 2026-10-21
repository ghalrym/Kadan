# Personas

Personas are saved instructions for responses, scoped to the selected agent. Open Personas, choose New persona, enter a name and instructions, then Save persona. The editor accepts names up to 200 characters and instructions up to 16,000 characters. Saving requires Tsuchi but does not run inference. Unsaved changes are browser-session drafts, not saved instructions.

When chat is ready and the agent has personas, the composer offers Persona for next response. It starts at No persona. A selection applies to subsequent messages sent from that chat view until changed; opening a persona in the editor does not select it for chat. Reopening the chat view resets the selector to No persona. Saving a note does not apply the selected persona.

The current prototype saves the chosen persona's identity with the queued message, then reads its saved instructions when that run prepares its context. Editing the persona before a queued run starts can therefore change the instructions that run receives. A prepared run retains its instruction snapshot when resumed. A persona that is missing or in trash causes an explicit failure rather than a silent fallback.

A persona adds instructions to the agent's existing instructions. It does not select a different model, create another conversation or clear the conversation history. No persona removes that additional instruction choice for a new run; earlier messages remain in the history and can still influence the response. Persona instructions remain subject to the model's context limits and ability to follow them.

Edits check the saved version and must not silently overwrite a newer edit. On an error, Load latest for comparison shows the current saved version while retaining the draft. The user can compare and revise before saving again. Revisions and trash are managed in Tsuchi; Kadan's persona editor changes only the name and instructions, preserving the other saved metadata.
