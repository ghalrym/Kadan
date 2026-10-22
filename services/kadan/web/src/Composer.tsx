import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { ArrowUp, ArrowUpRight, ChevronDown, Mic, NotebookPen, Paperclip, Square, UserRound, X } from "lucide-react";
import type { ChatAttachments } from "./attachments";
import AttachmentPicker from "./AttachmentPicker";
import type { Resource, Status } from "./types";
import VoiceCallControl, { type VoiceCall } from "./VoiceCallControl";
import { microphoneUnavailable, VoiceInput, type VoiceState } from "./voiceInput";

export function readiness(status: Status | null) {
  const storage = status?.tsuchi.state === "ready";
  return { storage, chat: storage && status?.hikari.state === "ready" };
}

function attention(status: Status | null) {
  if (!status) return null;
  if (status.tsuchi.state !== "ready")
    return { label: "Storage offline", detail: status.tsuchi.message, href: status.tsuchi.browser_url };
  const hikari = status.hikari;
  if (hikari.state === "ready") return null;
  return {
    label:
      hikari.state === "tier_required"
        ? "Choose a tier · notes only"
        : hikari.state === "model_required"
          ? "Download a model · notes only"
          : "Hikari offline · notes only",
    detail: hikari.message,
    href: hikari.settings_url,
  };
}

interface Props {
  agentName: string;
  attachments: ChatAttachments;
  value: string;
  onChange: (value: string) => void;
  onSend: (generate: boolean) => void;
  onStop?: () => void;
  sending: boolean;
  stopping?: boolean;
  blocked: boolean;
  status: Status | null;
  personas: Resource[];
  persona: string;
  onPersona: (id: string) => void;
  pet: ReactNode;
  autoFocus?: boolean;
  voiceCall: VoiceCall;
  onDictating: (active: boolean) => void;
}

export default function Composer({
  agentName,
  attachments,
  value,
  onChange,
  onSend,
  onStop,
  sending,
  stopping,
  blocked,
  status,
  personas,
  persona,
  onPersona,
  pet,
  autoFocus,
  voiceCall,
  onDictating,
}: Props) {
  const filePicker = useRef<HTMLInputElement>(null);
  const [attachmentError, setAttachmentError] = useState("");
  const field = useRef<HTMLTextAreaElement>(null);
  const microphone = useRef<VoiceInput | null>(null);
  const [voice, setVoice] = useState<VoiceState | null>(null);
  const [voiceError, setVoiceError] = useState("");
  const draft = useRef({ value, onChange });
  draft.current = { value, onChange };
  const ready = readiness(status);
  const issue = attention(status);
  const empty = !value.trim();
  const callActive = voiceCall.active;
  const locked = sending || blocked || voice !== null || callActive;
  const sendLocked = locked || !attachments.ready;
  useEffect(() => () => microphone.current?.cancel(), []);
  useEffect(() => onDictating(voice !== null), [voice, onDictating]);
  useEffect(() => {
    if (blocked || sending || !ready.chat) {
      microphone.current?.cancel();
      microphone.current = null;
      setVoice(null);
    }
  }, [blocked, sending, ready.chat]);

  function record() {
    if (microphone.current) { microphone.current.finish(); return; }
    setVoiceError("");
    const session = new VoiceInput(
      (text) => {
        const current = draft.current;
        const next = current.value ? `${current.value}\n${text}` : text;
        if (next.length > 16000) throw new Error("The transcript would exceed the message limit. Shorten the draft and try again.");
        current.onChange(next);
      },
      setVoice,
      (error) => { microphone.current = null; setVoice(null); setVoiceError(error ?? ""); },
    );
    microphone.current = session;
    void session.start();
  }
  useLayoutEffect(() => {
    const node = field.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 240)}px`;
  }, [value]);

  return (
    <div className="composer-wrap">
      {voiceError && <p role="alert">{voiceError}</p>}
      {voice && <p role="status">{voice === "requesting" ? "Waiting for microphone permission…" : voice === "recording" ? "Recording · up to 20 seconds. Stop to transcribe." : "Transcribing…"}</p>}
      {issue && (
        <a
          className="attention"
          href={issue.href ?? undefined}
          aria-disabled={!issue.href}
          target="_blank"
          rel="noreferrer"
          title={issue.detail}
        >
          <span className="dot warn" />
          {issue.label}
          {issue.href && <ArrowUpRight size={13} />}
        </a>
      )}
      <VoiceCallControl voiceCall={voiceCall} />
      <form
        className={`composer ${sending ? "is-sending" : ""}`}
        onSubmit={(event) => {
          event.preventDefault();
          if (!sendLocked && ready.storage) onSend(ready.chat);
        }}
      >
        <span className="perch">{pet}</span>
        <label className="sr-only" htmlFor="message">
          Message
        </label>
        <textarea
          id="message"
          ref={field}
          autoFocus={autoFocus}
          placeholder={ready.chat ? `Message ${agentName}` : "Write a note"}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          maxLength={16000}
          rows={1}
          // Keep typing while the agent replies, as in a text conversation.
          disabled={blocked || !ready.storage || callActive}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              if (!sendLocked && ready.storage) onSend(ready.chat);
            }
          }}
        />
        <AttachmentPicker attachments={attachments} disabled={blocked || callActive} />
        {attachmentError && <p role="alert">{attachmentError}</p>}
        <input ref={filePicker} type="file" multiple hidden aria-label="Choose attachments" onChange={event => {
          try { attachments.add(Array.from(event.target.files ?? [])); setAttachmentError(""); }
          catch (error) { setAttachmentError((error as Error).message); }
          event.target.value = "";
        }} />
        <div className="composer-bar">
          {ready.chat && personas.length > 0 ? (
            <label className={`chip persona ${persona ? "active" : ""}`}>
              <UserRound size={14} />
              <span className="sr-only">Persona for next response</span>
              <select
                value={persona}
                onChange={(event) => onPersona(event.target.value)}
                disabled={locked}
              >
                <option value="">No persona</option>
                {personas.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.title}
                  </option>
                ))}
              </select>
              <ChevronDown size={13} />
            </label>
          ) : (
            <span />
          )}
          <div className="send-actions">
            <button className="icon" type="button" aria-label="Attach files" title="Attach files · up to 10 files, 100 MiB total"
              disabled={blocked || callActive || !ready.storage} onClick={() => filePicker.current?.click()}>
              <Paperclip size={16} />
            </button>
            {ready.chat && (
              <button
                className="icon"
                type="button"
                aria-label={voice === "recording" ? "Stop recording and transcribe" : "Record a voice message"}
                title={microphoneUnavailable() ?? "Record a voice message · review the text before sending"}
                disabled={callActive || sending || blocked || voice === "requesting" || voice === "transcribing" || !!microphoneUnavailable()}
                onClick={record}
              >
                {voice === "recording" ? <Square size={16} /> : <Mic size={16} />}
              </button>
            )}
            {voice && (
              <button className="icon" type="button" aria-label="Discard recording" onClick={() => {
                microphone.current?.cancel(); microphone.current = null; setVoice(null);
              }}><X size={16} /></button>
            )}
            {ready.chat && (
              <button
                className="icon note"
                type="button"
                aria-label="Save as note without a reply"
                title="Save as note · no reply"
                onClick={() => onSend(false)}
                disabled={sendLocked || empty}
              >
                <NotebookPen size={16} />
              </button>
            )}
            {onStop && (
              <button
                className="send stop"
                type="button"
                aria-label="Stop response"
                title="Stop"
                onClick={onStop}
                disabled={stopping}
              >
                <Square size={13} fill="currentColor" />
              </button>
            )}
            <button
              className="send"
              disabled={sendLocked || empty || !ready.storage}
              aria-label={ready.chat ? "Send message" : "Save note"}
              title={ready.chat ? "Send · Enter" : "Save note · Enter"}
            >
              {ready.chat ? <ArrowUp size={18} /> : <NotebookPen size={16} />}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
