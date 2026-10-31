export interface CodexSetup {
  state: "signed_in" | "signing_in" | "signed_out" | "missing" | "unavailable";
  message: string;
  login?: { url: string; code: string } | null;
}

export interface CodexThreadSummary {
  id: string;
  title: string;
  cwd: string;
  created_at: string | null;
  updated_at: string;
  running: boolean;
}

export interface CodexThread extends CodexThreadSummary {
  items: { kind: "user" | "assistant" | "command"; text: string }[];
  items_truncated: boolean;
}

export interface CodexStatus {
  setup: CodexSetup;
  threads: CodexThreadSummary[];
}
