# Personas

Kadan is responsible for the persona used by each agent. Tsuchi stores saved persona records. The intended experience below is separate from the currently implemented editor and selection behavior.

## Intended experience

Each Kadan agent has its own persona and manages it. The agent curates its persona to the user, shaping how it responds and works with them. Users should not need to write and select persona instructions for each conversation.

Persona management should stay behind the agent experience rather than occupy a separate user-facing page. Hiding a persona editor alone does not provide agent curation. A persona guides responses; it does not guarantee that the model follows every instruction.

## Current behavior

Kadan lets users save named instructions for an agent's responses. Open **Personas**, create or edit a persona, and save it. Unsaved edits are browser drafts and can be lost.

Kadan does not yet automatically maintain or select a persona for each agent. The manual controls below describe the current implementation, not the intended agent-managed experience.

**Persona for next response** starts at **No persona**. The selection applies to messages sent from that chat view until changed; reopening chat resets it. Opening the editor does not select a persona, and saving a note does not apply one.

Editing a persona before queued work starts can change the instructions that work receives. Work that already prepared its instructions keeps them when resumed. A missing or trashed persona causes an error.

A persona adds instructions. It does not change the model or clear conversation history. Choosing **No persona** leaves earlier messages in the conversation, and instructions remain subject to the model's context and ability to follow them.

Personas supplement Kadan's [shared agent instructions](agent-instructions.md); choosing **No persona** does not remove those shared instructions.

Saving must not silently replace a newer edit. **Load latest for comparison** shows the saved version while retaining the draft so the user can revise it. Tsuchi manages [stored persona metadata](../tsuchi/personas.md), revisions and trash.

See [Persona instructions](../../architecture/kadan/personas.md) for the saved-state contract.
