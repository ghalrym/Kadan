import { agentPath, api, ApiError } from "./api.ts";
import { randomUuid } from "./uuid.ts";

export const MAX_ATTACHMENTS = 10;
export const MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024;
export interface DraftAttachment {
  id: string;
  name: string;
  size: number;
  status: "queued" | "uploading" | "checking" | "ready" | "error" | "reselect";
  error?: string;
}
interface UploadedAsset {
  id: string;
  agent_key: string;
  filename: string;
  size_bytes: number;
  deleted_at: string | null;
}

/** Owns one conversation's local upload queue and durable attachment references. */
export class ChatAttachments {
  entries: DraftAttachment[] = [];
  private files = new Map<string, File>();
  private queue: { id: string; check: boolean }[] = [];
  private active = false;
  private running: { id: string; generation: number } | null = null;
  private generation = 0;
  private storageKey: string;

  constructor(
    private agentKey: string,
    conversationId: string,
    private storage: Pick<Storage, "getItem" | "setItem" | "removeItem">,
    private changed: () => void,
    private request: typeof api = api,
    private createId: () => string = randomUuid,
  ) {
    this.storageKey = `kadan.attachments.${agentKey}.${conversationId}`;
    try {
      const saved = JSON.parse(storage.getItem(this.storageKey) || "[]");
      if (Array.isArray(saved) && saved.length <= MAX_ATTACHMENTS && saved.every(item =>
        item && typeof item.id === "string" && typeof item.name === "string"
        && Number.isSafeInteger(item.size) && item.size >= 0
        && ["queued", "uploading", "checking", "ready", "error", "reselect"].includes(item.status))
        && new Set(saved.map(item => item.id)).size === saved.length
        && saved.reduce((total, item) => total + item.size, 0) <= MAX_ATTACHMENT_BYTES) {
        this.entries = saved.map(item => ({ id: item.id, name: item.name, size: item.size,
          status: item.status === "ready" ? "ready" : "reselect", error: item.status === "ready" ? undefined : "Checking the earlier upload before asking for the file again." }));
      }
    } catch { /* A malformed attachment draft cannot become a submission. */ }
  }

  start() {
    this.active = true;
    for (const item of this.entries) if (item.status !== "ready") this.enqueue(item.id, true);
  }
  stop() { this.active = false; this.generation++; }
  get ready() { return this.entries.every(item => item.status === "ready"); }
  ids() {
    if (!this.ready) throw new Error("Finish uploading or remove the unresolved attachments before sending.");
    return this.entries.map(item => item.id);
  }
  add(files: File[]) {
    if (this.entries.length + files.length > MAX_ATTACHMENTS) throw new Error("Attach up to 10 files per message.");
    if (this.entries.reduce((total, item) => total + item.size, 0) + files.reduce((total, file) => total + file.size, 0) > MAX_ATTACHMENT_BYTES) {
      throw new Error("Attachments must total 100 MiB or less.");
    }
    const added = files.map(file => {
      const item: DraftAttachment = { id: this.createId(), name: file.name, size: file.size, status: "queued" };
      this.files.set(item.id, file);
      return item;
    });
    this.entries = [...this.entries, ...added];
    this.publish();
    for (const item of added) this.enqueue(item.id, false);
  }
  remove(id: string) { this.removeMany([id]); }
  removeMany(ids: readonly string[]) {
    this.entries = this.entries.filter(item => !ids.includes(item.id));
    for (const id of ids) this.files.delete(id);
    this.queue = this.queue.filter(item => !ids.includes(item.id));
    this.publish();
  }
  retry(id: string) { this.enqueue(id, true); }
  reselect(id: string, file: File) {
    const item = this.entries.find(item => item.id === id);
    if (!item) return;
    if (file.name !== item.name || file.size !== item.size) throw new Error(`Reselect the original ${item.name} (${item.size} bytes).`);
    this.files.set(id, file);
    this.enqueue(id, true);
  }
  private publish() {
    if (this.entries.length) this.storage.setItem(this.storageKey, JSON.stringify(this.entries));
    else this.storage.removeItem(this.storageKey);
    this.changed();
  }
  private enqueue(id: string, check: boolean) {
    const item = this.entries.find(item => item.id === id);
    if (!item || item.status === "ready" || this.queue.some(job => job.id === id)
      || (this.running?.id === id && this.running.generation === this.generation)) return;
    this.queue.push({ id, check });
    this.patch(id, { status: "queued", error: undefined });
    void this.pump();
  }
  private patch(id: string, patch: Partial<DraftAttachment>) {
    this.entries = this.entries.map(item => item.id === id ? { ...item, ...patch } : item);
    this.publish();
  }
  private current(id: string, generation: number) {
    return this.active && generation === this.generation && this.entries.some(item => item.id === id);
  }
  private confirm(item: DraftAttachment, value: UploadedAsset) {
    if (value.id !== item.id || value.agent_key !== this.agentKey || value.size_bytes !== item.size
      || typeof value.filename !== "string" || !value.filename || value.deleted_at !== null) {
      throw new Error("The stored file did not match this upload. Check Workspace before trying again.");
    }
    this.files.delete(item.id);
    this.patch(item.id, { name: value.filename, status: "ready", error: undefined });
  }
  private async pump() {
    if (!this.active || this.running) return;
    const job = this.queue.shift();
    if (!job) return;
    const item = this.entries.find(item => item.id === job.id);
    if (!item) { void this.pump(); return; }
    const generation = this.generation;
    this.running = { id: item.id, generation };
    const path = `${agentPath(this.agentKey)}/assets`;
    try {
      if (job.check) {
        this.patch(item.id, { status: "checking", error: undefined });
        try {
          const saved = await this.request<UploadedAsset>(`${path}/${encodeURIComponent(item.id)}`);
          if (this.current(item.id, generation)) this.confirm(item, saved);
          return;
        } catch (error) {
          if (!this.current(item.id, generation)) return;
          if (!(error instanceof ApiError && error.status === 404)) throw error;
        }
      }
      if (!this.current(item.id, generation)) return;
      const file = this.files.get(item.id);
      if (!file) {
        this.patch(item.id, { status: "reselect", error: "Upload not found. Reselect the original file or remove this attachment." });
        return;
      }
      this.patch(item.id, { status: "uploading", error: undefined });
      const body = new FormData();
      body.append("asset_id", item.id);
      body.append("file", file);
      const saved = await this.request<UploadedAsset>(path, body);
      if (this.current(item.id, generation)) this.confirm(item, saved);
    } catch (error) {
      if (this.current(item.id, generation)) this.patch(item.id, {
        status: "error", error: `Upload unconfirmed. ${(error as Error).message} Retry checks the saved file first.`,
      });
    } finally {
      this.running = null;
      void this.pump();
    }
  }
}
