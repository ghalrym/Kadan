# Personas

Each Kadan agent owns its persona and should curate it for the user. Kadan hides the standalone **Personas** page from its navigation. Existing saved persona instructions remain available to that agent; hiding the page does not delete them.

**Persona for next response** starts at **No persona**. The selection applies to messages sent from that chat view until changed; reopening chat resets it. Opening the editor does not select a persona, and saving a note does not apply one.

Editing a persona before queued work starts can change the instructions that work receives. Work that already prepared its instructions keeps them when resumed. A missing or trashed persona causes an error.

A persona adds instructions. It does not change the model or clear conversation history. Choosing **No persona** leaves earlier messages in the conversation, and instructions remain subject to the model's context and ability to follow them.

Personas supplement Kadan's [shared agent instructions](agent-instructions.md); choosing **No persona** does not remove those shared instructions.

Tsuchi retains its [stored persona editor](../tsuchi/personas.md), metadata, revisions and trash. Saving must not silently replace a newer edit. Hiding Kadan's editor does not add automatic persona selection or agent editing tools.

See [Persona instructions](../../architecture/kadan/personas.md) for the saved-state contract.
