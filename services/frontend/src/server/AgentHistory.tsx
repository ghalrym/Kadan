import { useEffect, useState } from "react";
import Markdown from "../kadan/Markdown";
import type { ConversationSummary, Resource } from "../kadan/types";
import { formatWhen, getJson, kadanPath, type Agent } from "./api";

const PAGE = 50;

export default function AgentHistory({ agent }: { agent: Agent }) {
  const [chat, setChat] = useState<Resource | null | undefined>(undefined);
  const [shown, setShown] = useState(PAGE);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    getJson<ConversationSummary[]>(`${kadanPath(agent.key)}/conversations`, controller.signal)
      .then(async (rows) => {
        const latest = [...rows].sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0];
        setChat(latest ? await getJson<Resource>(`${kadanPath(agent.key)}/resources/${latest.id}`, controller.signal) : null);
      })
      .catch((failure) => !controller.signal.aborted && setError((failure as Error).message));
    return () => controller.abort();
  }, [agent.key]);

  if (error) return <p className="error" role="alert">{error}</p>;
  if (chat === undefined) return <p className="srv-loading">Loading chat history…</p>;
  const visible = chat?.messages.filter((message) => message.role === "user" || message.role === "assistant") ?? [];
  if (!visible.length) return <p className="srv-empty">You and {agent.name} haven’t chatted yet.</p>;

  const messages = visible.slice(-shown);
  const hidden = visible.length - messages.length;
  return (
    <div className="srv-stack">
      <p className="field-hint">
        Your chat with {agent.name}, read-only. {visible.length} messages. To keep chatting, open the Kadan app.
      </p>
      {hidden > 0 && <button className="button srv-earlier" onClick={() => setShown((value) => value + PAGE)}>Show {Math.min(PAGE, hidden)} earlier messages</button>}
      <ol className="srv-history">
        {messages.map((message) => (
          <li key={message.id} className={message.role === "user" ? "from-you" : "from-agent"}>
            <header>
              <strong>{message.role === "user" ? "You" : agent.name}</strong>
              <time dateTime={message.created_at}>{formatWhen(message.created_at)}</time>
            </header>
            <Markdown>{message.content}</Markdown>
          </li>
        ))}
      </ol>
    </div>
  );
}
