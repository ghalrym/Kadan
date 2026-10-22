# Personas

Kadan gives each agent its own saved persona: instructions describing how it should respond and work with the user. The agent can read and update this persona during chat, based on the user's preferences. Users do not have to select it for each response. Tsuchi stores the persona and its revision history.

An agent starts without a saved persona and uses Kadan's [shared instructions](agent-instructions.md). Once the agent saves a persona, later text responses and voice calls include it automatically. Curation is model-directed; it does not happen on a background schedule, and asking for a change is not confirmation that it was saved. A persona guides behavior without guaranteeing that the model follows every instruction.

Queued work uses the latest saved persona when it begins. Running or resumed work keeps the instructions it already prepared. A save therefore does not replace the current response's instructions. [Voice calls](voice-calls.md) take their own snapshot when they start; persona edits and selection changes apply to the next call.

A failed save must be reported. Conflicting edits require reading the latest version before trying again. After an interrupted save, the agent checks the saved persona instead of automatically repeating the write. Stopping work does not undo a save already in progress.

The owned persona remains a normal persona record in Tsuchi, where users can inspect its history or edit it. If it is trashed, new text responses and calls report the problem until it is restored; Kadan does not silently choose another persona. Existing named personas remain separate and are not adopted automatically.

The current **Personas** editor also lets users save named instructions. **Persona for next response** adds the selected named persona to the agent's owned persona. This selection resets when chat opens. Saving a note does not apply it. Choosing **No persona** removes this additional selection; the agent's owned persona and shared instructions still apply. Selecting the owned persona itself does not include it twice.

The editor preserves drafts when a save conflicts. **Load latest for comparison** shows saved content without discarding the draft. Unsaved browser drafts can be lost when the tab closes. See [Tsuchi personas](../tsuchi/personas.md) for storage controls and [Persona instructions](../../architecture/kadan/personas.md) for the persistence and response contract.
