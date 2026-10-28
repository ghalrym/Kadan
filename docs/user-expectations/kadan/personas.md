# Personas

Kadan gives each agent its own saved persona: instructions describing how it should respond and work with the user. The agent can read and update this persona during chat, based on the user's preferences. Users do not have to select it for each response. Tsuchi stores the persona and its revision history.

An agent starts without a saved persona and uses Kadan's [shared instructions](agent-instructions.md). Once the agent saves a persona, later text responses include it automatically. Curation is model-directed; it does not happen on a background schedule, and asking for a change is not confirmation that it was saved. A persona guides behavior without guaranteeing that the model follows every instruction.

Queued work uses the latest saved persona when it begins. Running or resumed work keeps the instructions it already prepared. A save therefore does not replace the current response's instructions. New [voice calls](voice-calls.md) use the saved persona at call start; active calls keep their starting instructions. Calls have no persona-editing tools.

A failed save must be reported. Conflicting edits require reading the latest version before trying again. After an interrupted save, the agent checks the saved persona instead of automatically repeating the write. Stopping work does not undo a save already in progress.

The owned persona remains a normal persona record in Tsuchi, where users can inspect its history or edit it. If it is trashed, new text responses report the problem until it is restored; Kadan does not silently choose another persona. Existing named personas remain separate and are not adopted automatically.

Kadan hides the standalone **Personas** editor from normal navigation. Users can inspect and edit saved records in [Tsuchi](../tsuchi/personas.md). **Persona for next response** adds the selected named persona to the agent's owned persona. This selection resets when chat opens. Saving a note does not apply it. Choosing **No persona** removes this additional selection; the agent's owned persona and shared instructions still apply. Selecting the owned persona itself does not include it twice.

Tsuchi reports save conflicts without silently replacing newer changes. Unsaved editor drafts can be lost when the page reloads. See [Tsuchi personas](../tsuchi/personas.md) for storage controls and [Persona instructions](../../architecture/kadan/personas.md) for the persistence and response contract.
