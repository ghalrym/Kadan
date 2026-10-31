import { Fragment, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { ArrowUp, ArrowDown, ArrowUpRight, RotateCw, Search, X } from "lucide-react";
import Markdown from "./Markdown";
import RequestReferences, { type SelectedReference } from "./RequestReferences";
import Composer, { readiness } from "./Composer";
import Pet, { useMood } from "./Pet";
import AgentCard from "./AgentCard";
import ComputerViewer from "./ComputerViewer";
import ChatExport from "./ChatExport";
import Activity from "./Activity";
import { useVoiceCall } from "./VoiceCallControl";
import { activeLabel } from "./presence";
import { useAgentPet } from "./PetContext";
import { agentPath, api } from "./api";
import type { PetState } from "./pet";
import type { AgentRun, Asset, ImageJob, Message, Resource, RunSnapshot, Status } from "./types";
import { activeRun, cancellableImage, currentResponseOutcome, imageStatus, isImageAsset, latestConversation, responseProgress } from "./agentRuns";
import { randomUuid } from "./uuid";
import { ChatAttachments } from "./attachments";
import { chatStorageKeys, confirmSend, copyPendingToDraft, discardUnassignedPending, queueSend, readPending, type PendingMessage } from "./outbox";
import SaveChatResult from "./SaveChatResult";

interface Props {
  agentKey: string;
  agentName: string;
  chain: Resource;
  personas: Resource[];
  status: Status | null;
  entrance: PetState;
  onChange: (value: Resource) => void;
  onBusy: (value: boolean) => void;
  onEditPet: () => void;
  blocked: boolean;
  refreshStatus: () => Promise<void>;
}
/** Messages further apart than this get a timestamp between them. */
const BREAK_MS = 4 * 60 * 60 * 1000;
/** Older messages render on request so a long chain opens quickly. */
const PAGE = 200;
const NEAR_LATEST = 80;
const ROLE = { user: "You", assistant: "Assistant", system: "System", tool: "Tool" } as const;

function breakLabel(value: string) {
  const date = new Date(value);
  const now = new Date();
  const time = date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  const days = Math.round(
    (new Date(now).setHours(0, 0, 0, 0) - new Date(date).setHours(0, 0, 0, 0)) / 86400000,
  );
  if (days === 0) return `Today ${time}`;
  if (days === 1) return `Yesterday ${time}`;
  if (days < 7) return `${date.toLocaleDateString(undefined, { weekday: "long" })} ${time}`;
  return `${date.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    ...(date.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  })} at ${time}`;
}
const stamp = (value: string) =>
  new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

function Attachment({ agentKey, id }: { agentKey: string; id: string }) {
  const [asset, setAsset] = useState<Asset | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const path = `${agentPath(agentKey)}/assets/${encodeURIComponent(id)}`;
  useEffect(() => {
    let closed = false;
    setAsset(null);
    setFailed(false);
    api<Asset>(path).then((value) => {
      if (!closed) setAsset(value);
    }).catch(() => { if (!closed) setFailed(true); });
    return () => { closed = true; };
  }, [path, attempt]);
  if (failed) return <button className="tag" onClick={() => setAttempt(attempt + 1)}>Reload attachment</button>;
  if (!asset) return <span className="tag">Loading attachment…</span>;
  const content = `/api${path}/content`;
  return isImageAsset(asset) ? (
    <a className="message-image" href={content} target="_blank" rel="noreferrer">
      <img src={content} alt={asset.title || asset.filename} loading="lazy" onError={() => setFailed(true)} />
    </a>
  ) : <a className="tag" href={`${content}?download=true`}>{asset.title || asset.filename}<ArrowUpRight size={12} /></a>;
}

export default function Chat({
  agentKey,
  agentName,
  chain,
  personas,
  status,
  entrance,
  onChange,
  onBusy,
  onEditPet,
  blocked,
  refreshStatus,
}: Props) {
  const petSheet = useAgentPet(agentKey);
  const { draft: draftKey, pending: pendingKey } = chatStorageKeys(agentKey, chain.id);
  const pending = useRef<PendingMessage | null>(
    confirmSend(readPending(sessionStorage.getItem(pendingKey)), chain.messages),
  );
  const [recovery, setRecovery] = useState(pending.current);
  const [unassigned, setUnassigned] = useState(() => readPending(sessionStorage.getItem(`${draftKey}.pending`)));
  const [draft, setDraft] = useState(() => sessionStorage.getItem(draftKey) || "");
  const [, redrawAttachments] = useState(0);
  const [attachments] = useState(() => new ChatAttachments(agentKey, chain.id, sessionStorage,
    () => redrawAttachments(value => value + 1)));
  useEffect(() => {
    attachments.start();
    return () => attachments.stop();
  }, [attachments]);
  const [persona, setPersona] = useState("");
  const [references, setReferences] = useState<SelectedReference[]>(pending.current?.references ?? []);
  const [sending, setSending] = useState(false);
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [images, setImages] = useState<ImageJob[]>([]);
  const [pollError, setPollError] = useState("");
  const [activityConversation, setActivityConversation] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const generating = runs.some(activeRun);
  const imagePending = images.some((job) => cancellableImage(job) || job.status === "cancelling" || (job.status === "completed" && !job.delivered_message_id));
  /** The message being sent, shown before Tsuchi confirms it. */
  const [outgoing, setOutgoing] = useState<Message | null>(null);
  const [error, setError] = useState("");
  const [shown, setShown] = useState(PAGE);
  const scope = `${agentKey}:${chain.id}`;
  const closedFind = { scope, open: false, query: "", messageId: "" };
  const [findState, setFind] = useState(closedFind);
  const find = findState.scope === scope ? findState : closedFind;
  const findButton = useRef<HTMLButtonElement>(null);
  const messageElements = useRef(new Map<string, HTMLElement>());
  const messages = useRef<HTMLDivElement>(null);
  const followLatest = useRef(true);
  const jumpRequested = useRef(false);
  const [away, setAway] = useState(false);
  const [card, setCard] = useState(false);
  const [computer, setComputer] = useState(false);
  const closeCard = useCallback(() => setCard(false), []);
  // Tool activity saved by the retired plugin system is history the model no longer sees.
  const saved = chain.messages.filter((message) => message.role !== "tool");
  const currentChain = useRef(chain);
  currentChain.current = latestConversation(currentChain.current, chain);
  const mutation = useRef(0);
  const sendLock = useRef(false);
  const callbacks = useRef({ onChange, refreshStatus });
  callbacks.current = { onChange, refreshStatus };
  const ready = readiness(status);
  const [dictating, setDictating] = useState(false);
  const voiceCall = useVoiceCall(
    agentKey, persona || null,
    sending || blocked || dictating || !ready.storage,
  );
  // Hikari loads the model on demand; the first reply waits for it without a deadline.
  const modelLoading = generating && status?.hikari.state === "ready" && status.hikari.model_state === "loading";
  const progress = responseProgress(runs, status?.hikari);
  const typing = runs.some((run) => run.status === "running") && !modelLoading;
  const [mood, react] = useMood(
    !status
      ? "idle"
      : modelLoading
        ? "waiting"
        : typing
          ? "running"
          : !ready.chat
            ? "waiting"
            : draft.trim()
              ? "review"
              : "idle",
  );
  const items = outgoing ? [...saved, outgoing] : saved;
  const previousItems = useRef({ scope, count: items.length });
  const query = find.open ? find.query.trim().toLowerCase() : "";
  const addedItems = previousItems.current.scope === scope ? Math.max(0, items.length - previousItems.current.count) : 0;
  const shownCount = shown + ((!followLatest.current || query) ? addedItems : 0);
  const matches = query ? saved.filter((message) => message.content.toLowerCase().includes(query)) : [];
  const match = matches.find((message) => message.id === find.messageId) ?? matches[0];
  const matchIndex = match ? matches.indexOf(match) : -1;
  const matchShown = match ? items.length - items.findIndex((message) => message.id === match.id) : 0;
  const visible = items.slice(-shownCount);
  const hidden = items.length - visible.length;
  let lastUser = visible.length - 1;
  while (lastUser >= 0 && visible[lastUser].role !== "user") lastUser--;
  const receipt =
    lastUser < 0
      ? null
      : visible[lastUser] === outgoing
        ? "Sending…"
        : "Delivered";

  useLayoutEffect(() => {
    followLatest.current = true;
    setAway(false);
    setFind({ scope, open: false, query: "", messageId: "" });
    setShown(PAGE);
    previousItems.current = { scope, count: items.length };
    scrollToLatest();
  }, [scope]);
  useEffect(() => {
    const observer = new ResizeObserver(() => {
      if (followLatest.current && !query) scrollToLatest();
    });
    if (messages.current) observer.observe(messages.current);
    return () => observer.disconnect();
  }, [scope, query]);
  useLayoutEffect(() => {
    const added = items.length - previousItems.current.count;
    if (previousItems.current.scope === scope && added > 0 && (!followLatest.current || query)) {
      // Keep the message being read inside the presentation window as replies arrive.
      setShown((count) => count + added);
    }
    previousItems.current = { scope, count: items.length };
    if (!query && followLatest.current) scrollToLatest();
  }, [scope, items.length, typing]);
  useLayoutEffect(() => {
    if (matchShown > shownCount) setShown((count) => Math.max(count, matchShown));
  }, [matchShown, shownCount]);
  const matchVisible = matchShown > 0 && matchShown <= shownCount;
  useLayoutEffect(() => {
    if (match && matchVisible) {
      followLatest.current = false;
      setAway(true);
      messageElements.current.get(match.id)?.scrollIntoView({ block: "center" });
    }
  }, [match?.id, query, matchVisible]);
  useLayoutEffect(() => {
    if (!jumpRequested.current) return;
    jumpRequested.current = false;
    scrollToLatest();
    messages.current?.focus({ preventScroll: true });
  });
  useEffect(() => {
    if (match && match.id !== find.messageId) setFind({ ...find, messageId: match.id });
  }, [match?.id, find.messageId]);
  useEffect(() => react(entrance, 2), [entrance, react]);
  useEffect(() => {
    let closed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      const revision = mutation.current;
      try {
        const snapshot = await api<RunSnapshot>(`${agentPath(agentKey)}/conversations/${chain.id}/runs`);
        if (!closed && revision === mutation.current) {
          accept(snapshot.conversation);
          setRuns(snapshot.runs);
          setImages(snapshot.images);
          setActivityConversation(chain.id);
          setPollError("");
        }
      } catch (e) {
        if (!closed) setPollError(`Updates unavailable: ${(e as Error).message}`);
      } finally {
        if (!closed) timer = setTimeout(poll, 2000);
      }
    }
    void poll();
    // Closing the view only stops polling; the agent and image jobs continue on the server.
    return () => { closed = true; clearTimeout(timer); };
  }, [agentKey, chain.id]);
  useEffect(() => {
    if (!generating) return;
    const timer = setInterval(() => void callbacks.current.refreshStatus(), 2000);
    void callbacks.current.refreshStatus();
    return () => clearInterval(timer);
  }, [generating]);
  useEffect(() => {
    if (draft) sessionStorage.setItem(draftKey, draft);
    else sessionStorage.removeItem(draftKey);
  }, [draft, draftKey]);
  useEffect(() => {
    syncPending(chain.messages);
  }, [chain.messages]);

  function syncPending(messages: Resource["messages"]) {
    const stored = readPending(sessionStorage.getItem(pendingKey));
    const confirmed = confirmSend(stored, messages);
    if (stored?.generate && !confirmed) setReferences([]);
    savePending(confirmed);
  }
  function savePending(value: PendingMessage | null) {
    pending.current = value;
    setRecovery(value);
    if (value) sessionStorage.setItem(pendingKey, JSON.stringify(value));
    else sessionStorage.removeItem(pendingKey);
  }
  function accept(value: Resource) {
    const latest = latestConversation(currentChain.current, value);
    syncPending(latest.messages);
    if (!pending.current) setOutgoing(null);
    if (latest !== currentChain.current) {
      currentChain.current = latest;
      callbacks.current.onChange(latest);
    }
  }
  async function send(generate: boolean, retry?: PendingMessage) {
    const text = retry?.content ?? draft.trim();
    if (!text || sendLock.current || sending || blocked) return;
    if (!retry && !attachments.ready) {
      setError("Finish uploading or remove the unresolved attachments before sending.");
      return;
    }
    syncPending(currentChain.current.messages);
    if (pending.current && (!retry || retry.id !== pending.current.id)) {
      setError("Retry or discard the unsent message before sending another.");
      return;
    }
    const message = queueSend(pending.current, retry ?? {
      id: randomUuid(), content: text, generate, persona, attachment_ids: attachments.ids(),
      references: generate ? references : [],
    });
    const messageId = message.id;
    const personaId = message.persona;
    if (!query) {
      followLatest.current = true;
      setAway(false);
    }
    savePending(message);
    sendLock.current = true;
    mutation.current++;
    setSending(true);
    onBusy(true);
    setError("");
    setOutgoing({
      id: messageId,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
      resource_ids: generate && personaId ? [personaId] : [],
      attachment_ids: message.attachment_ids ?? [],
      sources: [],
    });
    if (!retry) {
      setDraft("");
      attachments.removeMany(message.attachment_ids ?? []);
    }
    const path = `${agentPath(agentKey)}/conversations/${chain.id}`;
    try {
      if (!generate) {
        accept(
          await api<Resource>(`${path}/notes`, {
            content: text,
            expected_version: currentChain.current.version,
            message_id: messageId,
            attachment_ids: message.attachment_ids ?? [],
          }),
        );
        react("waving");
      } else {
        const result = await api<{ run: AgentRun; conversation: Resource }>(`${path}/runs`, {
          content: text,
          expected_version: currentChain.current.version,
          persona_id: personaId || null,
          references: (message.references ?? []).map(({ resource_id, expected_version }) => ({ resource_id, expected_version })),
          message_id: messageId,
          attachment_ids: message.attachment_ids ?? [],
        });
        accept(result.conversation);
        setRuns((previous) => [...previous.filter((run) => run.id !== result.run.id), result.run]);
      }
    } catch (e) {
      react("failed", 2);
      setError((e as Error).message);
      try {
        accept(await api<Resource>(`${agentPath(agentKey)}/resources/${chain.id}`));
      } catch {
        /* Keep the already confirmed messages visible. */
      }
      if (pending.current?.id === messageId) setOutgoing(null);
      void refreshStatus();
    } finally {
      setSending(false);
      onBusy(false);
      sendLock.current = false;
      mutation.current++;
    }
  }
  function reload() {
    setSending(true);
    onBusy(true);
    api<Resource>(`${agentPath(agentKey)}/resources/${chain.id}`)
      .then((value) => {
        accept(value);
        setError("");
      })
      .catch((e) => setError(e.message))
      .finally(() => {
        setSending(false);
        onBusy(false);
      });
  }
  async function cancelRun() {
    if (stopping) return;
    setStopping(true);
    mutation.current++;
    try {
      for (const run of runs.filter(activeRun)) {
        const updated = await api<AgentRun>(`${agentPath(agentKey)}/runs/${run.id}/cancel`, {});
        setRuns((previous) => previous.map((item) => item.id === updated.id ? updated : item));
      }
    } catch (e) { setError((e as Error).message); }
    finally { setStopping(false); mutation.current++; }
  }
  async function cancelImage(job: ImageJob) {
    if (stopping) return;
    setStopping(true);
    mutation.current++;
    try {
      const updated = await api<ImageJob>(`${agentPath(agentKey)}/image-jobs/${job.id}/cancel`, {});
      setImages((previous) => previous.map((item) => item.id === updated.id ? updated : item));
    } catch (e) { setError((e as Error).message); }
    finally { setStopping(false); mutation.current++; }
  }
  const conversationRuns = runs.filter((run) => run.conversation_id === chain.id);
  const outcome = !sending && currentResponseOutcome(chain, conversationRuns);
  const perchStatus = modelLoading
    ? "Loading model…"
    : typing ? "Response pending…"
      : generating ? "Waiting to respond…"
        : imagePending ? "Image pending…"
          : !ready.chat ? "Notes only" : agentName;
  const empty = items.length === 0;
  const working = modelLoading || typing || generating || imagePending;
  const agentCard = card && (
    <AgentCard
      agentName={agentName}
      sheet={petSheet}
      mood={mood}
      presence={working ? perchStatus : activeLabel(saved)}
      onClose={closeCard}
      callDisabled={!ready.storage || sending || blocked || dictating || voiceCall.active || !!voiceCall.unavailable}
      callTitle={voiceCall.active ? "Already in a call" : voiceCall.unavailable ?? (!ready.storage ? "Calls need storage for the agent’s persona" : "Start a voice call")}
      onCall={() => {
        setCard(false);
        voiceCall.start();
      }}
      onOpenComputer={() => {
        setCard(false);
        setComputer(true);
      }}
      activity={<Activity
        conversation={chain}
        runs={runs}
        images={images}
        loading={activityConversation !== chain.id}
        error={pollError}
      />}
    />
  );
  function toggleCard() {
    if (!card) react("waving");
    setCard(!card);
  }
  function closeFind() {
    setFind(closedFind);
    findButton.current?.focus();
  }
  function scrollToLatest() {
    const element = messages.current;
    if (element) element.scrollTop = element.scrollHeight;
  }
  function moveMatch(direction: number) {
    if (!matches.length) return;
    const index = (matchIndex + direction + matches.length) % matches.length;
    setFind({ ...find, messageId: matches[index].id });
  }
  function jumpToLatest() {
    setFind(closedFind);
    followLatest.current = true;
    setAway(false);
    jumpRequested.current = true;
  }
  return (
    <div className={`chat-split ${computer ? "with-computer" : ""}`}>
    <section className={`chat ${find.open ? "finding" : ""}`} aria-label={`Messages with ${agentName}`}>
      {!empty && !find.open && (
        <div className={`lookout ${(generating || imagePending) ? "working" : ""}`}>
          <button type="button" className="lookout-pet" aria-label={`Open ${agentName}’s card`}
            title={agentName} aria-expanded={card} aria-haspopup="dialog" onClick={toggleCard}>
            <Pet sheet={petSheet} state={mood} size={96} look className="hero-pet" />
          </button>
          <span className="lookout-ledge" role="status" aria-live="polite">
            {(generating || imagePending) && <i aria-hidden="true" />}
            {perchStatus}
          </span>
          {agentCard}
        </div>
      )}
      <ChatExport chain={chain} agentName={agentName} />
      <div className={`conversation-find ${find.open ? "is-open" : ""}`} onKeyDown={(event) => {
        if (find.open && event.key === "Escape") { event.preventDefault(); closeFind(); }
      }}>
        <button ref={findButton} type="button" disabled={!saved.length && !find.open} aria-expanded={find.open}
          aria-label={find.open ? "Close find" : "Find in conversation"} title={find.open ? "Close find" : "Find in conversation"}
          onClick={() => find.open ? closeFind() : setFind({ ...closedFind, open: true })}>
          {find.open ? <X size={16} /> : <Search size={16} />}
        </button>
        {find.open && (
          <form role="search" aria-label="Find in conversation" onSubmit={(event) => {
            event.preventDefault();
            moveMatch(1);
          }}>
            <label className="search-field">
              <span className="sr-only">Find in saved text</span>
              <input type="search" autoFocus placeholder="Find in saved text" value={find.query}
                onChange={(event) => setFind({ ...find, query: event.target.value, messageId: "" })}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && event.shiftKey) { event.preventDefault(); moveMatch(-1); }
                }} />
            </label>
            <span className="find-count" role="status" aria-live="polite">
              {query ? matches.length ? `${matchIndex + 1} of ${matches.length} messages` : "No matching messages" : "Saved text only"}
            </span>
            <button type="button" disabled={!matches.length} aria-label="Previous matching message" title="Previous (Shift+Enter)"
              onClick={() => moveMatch(-1)}><ArrowUp size={16} /></button>
            <button type="submit" disabled={!matches.length} aria-label="Next matching message" title="Next (Enter)"><ArrowDown size={16} /></button>
          </form>
        )}
      </div>
      <div className="messages" ref={messages} role="region" aria-label="Conversation messages" tabIndex={-1}
        onScroll={(event) => {
          const element = event.currentTarget;
          const near = element.scrollHeight - element.clientHeight - element.scrollTop <= NEAR_LATEST;
          followLatest.current = near;
          setAway(!near);
        }}>
        <div className={`messages-inner ${empty ? "is-empty" : ""}`}>
          {empty ? (
            <div className="home-stage">
              <div className="home-pet">
                <button type="button" className="lookout-pet" aria-label={`Open ${agentName}’s card`}
                  title={agentName} aria-expanded={card} aria-haspopup="dialog" onClick={toggleCard}>
                  <Pet sheet={petSheet} state={mood} size={192} look grounded className="hero-pet" />
                </button>
                {agentCard}
              </div>
              <h2>{agentName}</h2>
              <button className="pet-edit-link" onClick={onEditPet}>Edit pet</button>
            </div>
          ) : (
            hidden > 0 && (
              <button className="earlier" onClick={() => setShown((count) => count + PAGE)}>
                Show earlier messages
              </button>
            )
          )}
          {visible.map((message, index) => {
            const previous = visible[index - 1];
            const next = visible[index + 1];
            const gap = (a?: Message, b?: Message) =>
              !a || !b || new Date(b.created_at).getTime() - new Date(a.created_at).getTime() > BREAK_MS;
            const split = gap(previous, message);
            const first = split || previous?.role !== message.role;
            const last = gap(message, next) || next?.role !== message.role;
            const tags = message.resource_ids?.length > 0;
            const sources = message.sources?.length > 0;
            return (
              <Fragment key={message.id}>
                {split && (
                  <div className="time-break">
                    <time dateTime={message.created_at}>{breakLabel(message.created_at)}</time>
                  </div>
                )}
                <article
                  ref={(element) => {
                    if (element) messageElements.current.set(message.id, element);
                    else messageElements.current.delete(message.id);
                  }}
                  className={`message ${message.role} ${first ? "first" : ""} ${last ? "last" : ""} ${message === outgoing ? "sending" : ""} ${message.id === match?.id ? "find-match" : ""}`}
                  aria-label={message.id === match?.id ? `Matching message ${matchIndex + 1} of ${matches.length}` : undefined}
                  title={stamp(message.created_at)}
                >
                  <h3 className={message.role === "user" || message.role === "assistant" ? "sr-only" : "role"}>
                    {ROLE[message.role]}
                  </h3>
                  {message.content && <div className="body">
                    <Markdown>{message.content}</Markdown>
                  </div>}
                  {message.role === "assistant" && message.content.trim() &&
                    <SaveChatResult key={`${agentKey}:${chain.id}:${message.id}`} agentKey={agentKey} conversationId={chain.id}
                      message={message} disabled={blocked || !ready.storage} />}
                  {message.attachment_ids?.map((id) => <Attachment key={id} agentKey={agentKey} id={id} />)}
                  {(tags || sources) && (
                    <footer>
                      {message.resource_ids.map((id) => (
                        <span className="tag" key={id}>
                          {personas.find((p) => p.id === id)?.title || `Resource ${id.slice(0, 8)}`}
                        </span>
                      ))}
                      {message.sources?.length > 0 &&
                        (status?.tsuchi.browser_url ? (
                          <a className="tag" href={status.tsuchi.browser_url} target="_blank" rel="noreferrer">
                            Attachments in Tsuchi
                            <ArrowUpRight size={12} />
                          </a>
                        ) : (
                          <span className="tag">Attachments in Tsuchi</span>
                        ))}
                    </footer>
                  )}
                </article>
                {index === lastUser && receipt && (
                  <p className="receipt" aria-live="polite">
                    {receipt}
                  </p>
                )}
              </Fragment>
            );
          })}
          {typing && (
            <div className="typing" aria-hidden="true">
              <i />
              <i />
              <i />
            </div>
          )}
          {progress && (
            <p className="notice" role="status">
              {progress}
            </p>
          )}
          {outcome && <p className="notice" role="status">
            {outcome.status === "failed" ? `Response failed: ${outcome.error || "Please try again."}` : "Response stopped"}
          </p>}
          {images.filter((job) => imageStatus(job)).map((job) => (
            <div className="image-job notice" key={job.id} role="status">
              <span>{imageStatus(job)}</span>
              {cancellableImage(job) && <button disabled={stopping} onClick={() => void cancelImage(job)}>Stop image</button>}
            </div>
          ))}
        </div>
      </div>
      {(away || query) && <button type="button" className="jump-latest" onClick={jumpToLatest}>
        <ArrowDown size={14} />{find.open ? "Close find and jump to latest" : "Jump to latest"}
      </button>}
      {(error || pollError) && (
        <div className="banner inline" role="alert">
          {error || pollError}
          <button disabled={sending || blocked} onClick={reload}>
            <RotateCw size={14} />
            Reload
          </button>
        </div>
      )}
      {recovery && !sending && (
        <div className="banner inline" role="status">
          <div>
            <strong>Unsent message</strong>
            <p>{recovery.content}</p>
            {recovery.attachment_ids?.map(id => <Attachment key={id} agentKey={agentKey} id={id} />)}
          </div>
          <button disabled={blocked || !ready.storage || (recovery.generate && !ready.chat)}
            onClick={() => void send(recovery.generate, recovery)}>Retry</button>
          <button disabled={blocked} onClick={() => savePending(null)}>Discard</button>
        </div>
      )}
      {unassigned && (
        <div className="banner inline" role="status">
          <div>
            <strong>Earlier unsent message</strong>
            <p>{unassigned.content}</p>
            <p>Its original conversation wasn't saved. Check your history before copying it to the draft. It will send as a new message.</p>
          </div>
          <button disabled={blocked || sending || dictating || voiceCall.active} onClick={() => {
            try { setDraft(copyPendingToDraft(draft, unassigned)); }
            catch (error) { setError((error as Error).message); }
          }}>Copy to draft</button>
          <button disabled={blocked} onClick={() => {
            discardUnassignedPending(sessionStorage, agentKey);
            setUnassigned(null);
          }}>Discard</button>
        </div>
      )}
      <RequestReferences agentKey={agentKey} selected={references} onChange={setReferences}
        disabled={sending || blocked || !!recovery} />
      <Composer
        agentName={agentName}
        attachments={attachments}
        value={draft}
        onChange={setDraft}
        onSend={(generate) => void send(generate)}
        onStop={generating ? () => void cancelRun() : undefined}
        stopping={stopping}
        sending={sending}
        blocked={blocked}
        status={status}
        personas={personas}
        persona={persona}
        onPersona={setPersona}
        pet={null}
        autoFocus
        voiceCall={voiceCall}
        onDictating={setDictating}
      />
    </section>
    {computer && <ComputerViewer agentKey={agentKey} agentName={agentName} onClose={() => setComputer(false)} />}
    </div>
  );
}
