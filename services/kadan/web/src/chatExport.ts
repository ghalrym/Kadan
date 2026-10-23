import type { Message, Resource } from "./types.ts";

const roles = { user: "User", assistant: "Assistant", system: "System" } as const;

function label(value: string): string {
  return value.replace(/[\r\n]+/g, " ").replace(/[\\`*_{}[\]()<>#!|]/g, "\\$&");
}

function webLink(value: string): string | null {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:"
      ? url.href.replace(/</g, "%3C").replace(/>/g, "%3E") : null;
  } catch { return null; }
}

/** Export only the saved conversation, never the composer's draft or optimistic message. */
export function conversationMarkdown(chain: Resource, agentName: string, origin: string): string {
  const lines = [
    `# Chat with ${label(agentName)}`,
    "",
    `Conversation: ${label(chain.title)} (${label(chain.id)})`,
    `Agent: ${label(chain.agent_key)}`,
    `Saved snapshot: ${label(chain.updated_at)} · version ${chain.version}`,
    "",
    "This snapshot contains saved messages loaded in Kadan. Later messages and unsent drafts are not included. File links require access to the original Kadan installation; file contents are not included.",
  ];
  for (const message of chain.messages.filter((item): item is Message & { role: keyof typeof roles } => item.role !== "tool")) {
    lines.push("", "---", "", `## ${roles[message.role]} · ${label(message.created_at)}`, "", message.content);
    for (const id of message.attachment_ids) {
      const path = `/api/agents/${encodeURIComponent(chain.agent_key)}/assets/${encodeURIComponent(id)}/content?download=true`;
      lines.push("", `Attachment: [${label(id)}](<${new URL(path, origin).href}>)`);
    }
    for (const id of message.resource_ids) lines.push("", `Resource: ${label(id)}`);
    for (const source of message.sources) {
      const url = webLink(source.url);
      lines.push("", url ? `Source: [${label(source.label)}](<${url}>)` : `Source: ${label(source.label)} (link unavailable)`);
    }
  }
  return `${lines.join("\n")}\n`;
}
