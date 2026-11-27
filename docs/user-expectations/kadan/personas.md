# Personas

Kadan gives each agent one saved persona: Markdown instructions describing how it should respond and work with the user. The agent can read and update its persona during chat, and users can edit it directly in Kadan using the **Persona** button on the agent's card (under **Call**) or on the [Agent page](navigation.md). Tsuchi stores the persona and its revision history.

An agent starts without a saved persona and uses Kadan's [shared instructions](agent-instructions.md). Once a persona is saved—either by the user in the Markdown editor or by the agent during chat—later text responses include it automatically. Agent curation is model-directed; it does not happen on a background schedule, and asking for a change is not confirmation that it was saved. A persona guides behavior without guaranteeing that the model follows every instruction.

Queued work uses the latest saved persona when it begins. Running or resumed work keeps the instructions it already prepared. A save therefore does not replace the current response's instructions. New [voice calls](voice-calls.md) use the saved persona at call start; active calls keep their starting instructions. Calls have no persona-editing tools.

A failed save must be reported. Conflicting edits require loading the latest saved version for comparison rather than silently replacing newer changes. After an interrupted save, the agent checks the saved persona instead of automatically repeating the write. Stopping work does not undo a save already in progress.

The agent's persona is a normal persona record in Tsuchi, where users can also inspect its revision history. If it is trashed, new text responses and calls report the problem until it is restored; Kadan does not silently choose another persona.

A persona adds instructions; it does not change the selected model or clear saved conversation history. All instructions remain subject to the model's context and ability to follow them. Unsaved editor drafts can be lost if the page reloads or the editor is dismissed. See [Tsuchi personas](../tsuchi/personas.md) for storage controls and [Persona instructions](../../architecture/kadan/personas.md) for the persistence and response contract.
