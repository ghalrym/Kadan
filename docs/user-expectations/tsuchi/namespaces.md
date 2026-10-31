# Agent namespaces

Tsuchi stores each agent's content in a named namespace. Users can create one with a display name and a unique key, then choose it from **Namespace**. Kadan's **Hatch an agent** creates the same kind of stored agent; creating it in either application makes it available to the other after refreshing.

The key is permanent. Kadan can [rename the display name](../kadan/renaming-agents.md); refresh Tsuchi to see the saved name. Renaming preserves the namespace and all its content. The current interfaces do not delete agents. Creating an agent needs storage but does not download models, run inference or create a separate computer. Duplicate or invalid keys must produce an error without replacing an existing agent.

Switching namespaces changes the resources, files and activity shown. Links between stored resources and files must stay within the same namespace. Switching does not copy content, stop agent work or move it to a different agent. Save editor drafts before switching; unsaved drafts are not durable content.

The browser remembers its selected namespace, but another browser can have a different selection. An unavailable service must show an error rather than an empty initial installation. After an uncertain creation, refresh the agent list before trying again.

Namespace keys select content; they are not credentials or access controls. The [operator security responsibilities](../security.md) apply.
