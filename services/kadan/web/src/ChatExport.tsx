import { useState } from "react";
import { Download } from "lucide-react";
import { conversationMarkdown } from "./chatExport";
import type { Resource } from "./types";

export default function ChatExport({ chain, agentName }: { chain: Resource; agentName: string }) {
  const [error, setError] = useState("");
  const empty = !chain.messages.some((message) => message.role !== "tool");
  function download() {
    if (empty) return;
    setError("");
    try {
      const blob = new Blob([conversationMarkdown(chain, agentName, window.location.origin)], { type: "text/markdown;charset=utf-8" });
      const link = document.createElement("a");
      const url = URL.createObjectURL(blob);
      try {
        link.href = url;
        link.download = `kadan-${chain.agent_key}-${chain.id}.md`;
        document.body.appendChild(link);
        link.click();
      } finally {
        link.remove();
        // Give the browser time to start consuming the download before releasing it.
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    } catch {
      setError("Could not start chat download. Try again.");
    }
  }
  return (
    <div className="chat-export">
      <button type="button" className="icon" onClick={download} disabled={empty}
        aria-label="Download saved chat as Markdown" title={empty ? "No saved messages to download" : "Download saved chat as Markdown"}>
        <Download size={18} aria-hidden="true" />
      </button>
      {error && <p role="alert">{error}</p>}
    </div>
  );
}
