import { Paperclip, Send } from "lucide-react";
import { assetUrl, formatDate } from "./api";
import {
  ErrorNotice,
  Markdown,
  SelectItems,
  SourceLinks,
  SourcesField,
} from "./components";
import type { Asset, Message, Resource } from "./types";

export type MessageDraft = Omit<Message, "id" | "created_at">;
export function emptyMessageDraft(): MessageDraft {
  return {
    role: "user",
    content: "",
    attachment_ids: [],
    resource_ids: [],
    sources: [],
  };
}

export function ConversationPanel({
  agentKey,
  messages,
  assets,
  resources,
  canAppend,
  draft,
  onChange,
  busy,
  error,
  onAppend,
}: {
  agentKey: string;
  messages: Message[];
  assets: Asset[];
  resources: Resource[];
  canAppend: boolean;
  draft: MessageDraft;
  onChange: (draft: MessageDraft) => void;
  busy: boolean;
  error: string;
  onAppend: () => Promise<void>;
}) {
  return (
    <div className="conversation">
      <div className="messages">
        {messages.length === 0 ? (
          <p className="muted">No messages.</p>
        ) : (
          messages.map((message) => (
            <article
              className={`message message-${message.role}`}
              key={message.id}
            >
              <div className="message-meta">
                <span>{message.role}</span>
                <time dateTime={message.created_at}>
                  {formatDate(message.created_at)}
                </time>
              </div>
              <Markdown text={message.content} />
              <div className="link-list">
                {message.attachment_ids.map((id) => (
                  <a key={id} href={assetUrl(agentKey, id, true)}>
                    <Paperclip size={14} />
                    {assets.find((asset) => asset.id === id)?.title || id}
                  </a>
                ))}
              </div>
              <div className="link-list">
                {message.resource_ids.map((id) => (
                  <a
                    key={id}
                    target="_blank"
                    rel="noreferrer"
                    href={`/?agent=${encodeURIComponent(agentKey)}&resource=${encodeURIComponent(id)}`}
                  >
                    {resources.find((resource) => resource.id === id)?.title ||
                      id}
                  </a>
                ))}
              </div>
              <SourceLinks sources={message.sources} />
            </article>
          ))
        )}
      </div>
      {canAppend ? (
        <form
          className="message-composer"
          onSubmit={(event) => {
            event.preventDefault();
            void onAppend();
          }}
        >
          <ErrorNotice message={error} />
          <fieldset disabled={busy}>
            <label>
              Role
              <select
                value={draft.role}
                onChange={(event) =>
                  onChange({
                    ...draft,
                    role: event.target.value as Message["role"],
                  })
                }
              >
                <option value="user">User</option>
                <option value="assistant">Assistant</option>
                <option value="system">System</option>
                <option value="tool">Tool</option>
              </select>
            </label>
            <label>
              Message
              <textarea
                value={draft.content}
                onChange={(event) =>
                  onChange({ ...draft, content: event.target.value })
                }
                required
                rows={5}
              />
            </label>
            <div className="field-pair">
              <SelectItems
                label="Attachments"
                items={assets}
                selected={draft.attachment_ids}
                onChange={(attachment_ids) =>
                  onChange({ ...draft, attachment_ids })
                }
              />
              <SelectItems
                label="Resource links"
                items={resources}
                selected={draft.resource_ids}
                onChange={(resource_ids) =>
                  onChange({ ...draft, resource_ids })
                }
              />
            </div>
            <SourcesField
              value={draft.sources}
              onChange={(sources) => onChange({ ...draft, sources })}
            />
            <button
              className="primary"
              disabled={busy || !draft.content.trim()}
            >
              <Send size={16} />
              {busy ? "Appending…" : "Append message"}
            </button>
          </fieldset>
        </form>
      ) : (
        messages.length === 0 && (
          <p className="muted">Save the conversation to append messages.</p>
        )
      )}
    </div>
  );
}
