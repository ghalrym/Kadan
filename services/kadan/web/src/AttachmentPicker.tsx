import { useState } from "react";
import { X } from "lucide-react";
import type { ChatAttachments } from "./attachments";

const labels = { queued: "Waiting to upload…", uploading: "Uploading…", checking: "Checking saved file…",
  ready: "Attached", error: "Upload unconfirmed", reselect: "Original file needed" };

export default function AttachmentPicker({ attachments, disabled }: { attachments: ChatAttachments; disabled: boolean }) {
  const [error, setError] = useState("");
  if (!attachments.entries.length) return null;
  return <div className="attachment-draft">
    {attachments.entries.map(item => <div className="attachment-draft-row" key={item.id}>
      <div className="attachment-draft-detail">
        <strong>{item.name}</strong>
        <span role="status">{labels[item.status]} · {item.size.toLocaleString()} bytes</span>
        {item.error && <span>{item.error}</span>}
      </div>
      {item.status === "error" && <button type="button" disabled={disabled} onClick={() => attachments.retry(item.id)}>Retry</button>}
      {item.status === "reselect" && <label className="attachment-reselect">
        Reselect file
        <input type="file" aria-label={`Reselect ${item.name}`} disabled={disabled} onChange={event => {
          const file = event.target.files?.[0];
          if (file) {
            try { attachments.reselect(item.id, file); setError(""); }
            catch (error) { setError((error as Error).message); }
          }
          event.target.value = "";
        }} />
      </label>}
      <button className="icon" type="button" disabled={disabled} aria-label={`Remove ${item.name}`}
        title="Remove from this message; uploaded files remain in Workspace" onClick={() => attachments.remove(item.id)}><X size={14} /></button>
    </div>)}
    {error && <p role="alert">{error}</p>}
  </div>;
}
