export interface McpToolSummary {
  name: string;
  description: string;
  read_only: boolean;
  idempotent: boolean;
}

export interface McpServer {
  id: string;
  slug: string;
  name: string;
  preset: "t3_code" | "custom";
  url: string;
  app_url: string | null;
  auth_type: "oauth" | "bearer" | "none";
  has_credentials: boolean;
  enabled: boolean;
  status: "disconnected" | "connected" | "error";
  status_error: string | null;
  pinned_tools: string[];
  tools: McpToolSummary[];
  tools_synced_at: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface McpConnectResponse {
  server: McpServer;
  authorize_url: string | null;
}

export const T3_TOOL_GROUPS: { id: string; label: string; tools: string[] }[] = [
  {
    id: "core",
    label: "Coding threads & questions (recommended)",
    tools: [
      "t3_project_list",
      "t3_thread_list",
      "t3_thread_search",
      "t3_thread_launch",
      "t3_thread_send",
      "t3_thread_read",
      "t3_thread_interrupt",
      "t3_pending_request",
    ],
  },
  {
    id: "projects",
    label: "Create & clone projects",
    tools: ["t3_project_read", "t3_project_create", "t3_project_clone"],
  },
  {
    id: "pull_requests",
    label: "Pull requests & CI checks",
    tools: ["pull_request_list", "pull_request_read", "pull_request_create", "ci_logs"],
  },
  {
    id: "issues",
    label: "Issue tracker",
    tools: ["list_issues", "read_issue", "create_issue"],
  },
  {
    id: "schedules",
    label: "Scheduled tasks",
    tools: ["list_scheduled_tasks", "schedule_task", "delete_scheduled_task"],
  },
];

export function statusLabel(server: McpServer): string {
  if (!server.enabled) return "Paused";
  if (server.status === "connected") return "Connected";
  if (server.status === "error") return "Needs attention";
  return "Not connected";
}

export function toggleGroupTools(
  currentPinned: string[],
  availableTools: McpToolSummary[],
  groupTools: string[],
  enable: boolean,
): string[] {
  const available = new Set(availableTools.map((t) => t.name));
  const groupSet = new Set(groupTools);
  const kept = currentPinned.filter((name) => !groupSet.has(name));
  if (!enable) return kept;
  for (const name of groupTools) {
    if (available.has(name) && !kept.includes(name)) kept.push(name);
  }
  return kept;
}
