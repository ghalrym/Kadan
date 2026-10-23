# Persona instructions

Personas are versioned Tsuchi resources scoped to an agent. Kadan's normal navigation does not expose the standalone persona editor. The resource APIs and editor source remain available, and updates to the name and Markdown instructions preserve other resource metadata. See the privately provided expectations for personas.

A queued message stores `persona_id`, not a copy of its instructions. Context preparation reads the current saved persona, verifies its kind and active state, and appends its Markdown to the agent's base instructions. Missing or trashed personas fail explicitly. The prepared instruction snapshot is persisted with the run, so recovery does not substitute later edits into work already started.

Persona selection is chat-view state and resets to no selection when the view opens. The editor and selector are independent. The editor permits names up to 200 characters and instructions up to 16,000 characters; these limits do not replace the model's context admission.

Updates use the saved resource version. A conflict retains the user's draft and offers the latest saved resource for comparison instead of overwriting it. Revisions and trash remain Tsuchi's responsibility.
