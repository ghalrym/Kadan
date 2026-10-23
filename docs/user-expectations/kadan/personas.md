# Personas

Kadan is responsible for the persona used by each agent. Tsuchi stores saved persona records. The intended experience below is separate from the current manual selection behavior.

## Intended experience

Each Kadan agent has its own persona and manages it. The agent curates its persona to the user, shaping how it responds and works with them. Users should not need to write and select persona instructions for each conversation.

Persona management should stay behind the agent experience rather than occupy a separate user-facing page. Hiding a persona editor alone does not provide agent curation. A persona guides responses; it does not guarantee that the model follows every instruction.

## Current behavior

Kadan hides the standalone **Personas** page from its navigation. Existing saved instructions remain available to the selected agent; hiding the page does not delete them. Users can create and edit those records in [Tsuchi's persona editor](../tsuchi/personas.md).

Kadan does not yet automatically maintain or select a persona for each agent. The manual controls below describe the current implementation, not the intended agent-managed experience.

**Persona for next response** starts at **No persona**. The selection applies to messages sent from that chat view until changed; reopening chat resets it. Editing a persona in Tsuchi does not select it in chat, and saving a note does not apply one.

Editing a persona before queued work starts can change the instructions that work receives. Work that already prepared its instructions keeps them when resumed. A missing or trashed persona causes an error.

A persona adds instructions. It does not change the model or clear conversation history. Choosing **No persona** leaves earlier messages in the conversation, and instructions remain subject to the model's context and ability to follow them.

Personas supplement Kadan's [shared agent instructions](agent-instructions.md); choosing **No persona** does not remove those shared instructions.

Saving in Tsuchi must not silently replace a newer edit. Tsuchi manages stored persona metadata, revisions and trash. The hidden Kadan page does not add automatic persona selection or agent editing tools.

See [Persona instructions](../../architecture/kadan/personas.md) for the saved-state contract.
