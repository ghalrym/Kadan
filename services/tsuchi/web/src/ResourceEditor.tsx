import { useId, useState } from "react";
import { randomUuid } from "./uuid";
import { Download, History, RotateCcw, Save, Trash2 } from "lucide-react";
import {
  agentPath,
  allResources,
  assetUrl,
  formatDate,
  json,
  request,
  useLoad,
} from "./api";
import {
  Dialog,
  ErrorNotice,
  Loading,
  Markdown,
  SelectItems,
  SourceLinks,
  SourcesField,
  TagsField,
} from "./components";
import { ConversationPanel, emptyMessageDraft } from "./ConversationPanel";
import { DeckEditor, SlidePreview } from "./DeckEditor";
import { newResource, singularLabels } from "./types";
import type {
  Asset,
  Kind,
  Resource,
  ResourceFields,
  ResourceInput,
  Revision,
} from "./types";

function resourceInput(resource: ResourceInput): ResourceInput {
  const { title, markdown, parent_id, tags, sources, attachment_ids } =
    resource;
  const fields = { title, markdown, parent_id, tags, sources, attachment_ids };
  if (resource.kind === "deck")
    return {
      ...fields,
      kind: resource.kind,
      deck: resource.deck,
      messages: [],
    };
  if (resource.kind === "conversation")
    return {
      ...fields,
      kind: resource.kind,
      deck: null,
      messages: resource.messages,
    };
  return { ...fields, kind: resource.kind, deck: null, messages: [] };
}
function parentChoices(resources: Resource[], currentId?: string) {
  const excluded = new Set(currentId ? [currentId] : []);
  let changed = true;
  while (changed) {
    changed = false;
    for (const resource of resources)
      if (
        resource.parent_id &&
        excluded.has(resource.parent_id) &&
        !excluded.has(resource.id)
      ) {
        excluded.add(resource.id);
        changed = true;
      }
  }
  return resources.filter(
    (resource) => resource.kind === "space" && !excluded.has(resource.id),
  );
}
export function ResourceEditor({
  agentKey,
  initial,
  onClose,
  onChanged,
}: {
  agentKey: string;
  initial: Resource | Kind;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [current, setCurrent] = useState<Resource | null>(
    typeof initial === "string" ? null : initial,
  );
  const [draft, setDraft] = useState<ResourceInput>(() =>
    typeof initial === "string" ? newResource(initial) : resourceInput(initial),
  );
  const [tab, setTab] = useState<"content" | "details" | "history">("content");
  const [markdownMode, setMarkdownMode] = useState<"edit" | "preview">("edit");
  const [busy, setBusy] = useState(false);
  const [messageDraft, setMessageDraft] = useState(emptyMessageDraft);
  const [messageBusy, setMessageBusy] = useState(false);
  const [messageError, setMessageError] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [formKey, setFormKey] = useState(0);
  const formId = useId();
  const {
    data,
    loading,
    error: loadError,
  } = useLoad(async () => {
    const [resources, assets] = await Promise.all([
      allResources(agentKey),
      request<Asset[]>(`${agentPath(agentKey)}/assets`),
    ]);
    return { resources, assets };
  }, [agentKey]);
  const history = useLoad(
    () =>
      current && tab === "history"
        ? request<Revision[]>(
            `${agentPath(agentKey)}/resources/${current.id}/versions`,
          )
        : Promise.resolve([]),
    [current?.id, current?.version, tab],
  );
  const messageDirty =
    JSON.stringify(messageDraft) !== JSON.stringify(emptyMessageDraft());
  const pending = busy || messageBusy;
  const dirty =
    JSON.stringify(draft) !==
    JSON.stringify(current ? resourceInput(current) : newResource(draft.kind));
  const update = (patch: Partial<ResourceFields>) => {
    setDraft((previous) => ({ ...previous, ...patch }));
    setNotice("");
  };
  const accept = (resource: Resource) => {
    setCurrent(resource);
    setDraft(resourceInput(resource));
    setFormKey((key) => key + 1);
    onChanged();
  };
  const mutate = async (
    action: () => Promise<Resource>,
    success: string,
    close = false,
  ) => {
    setBusy(true);
    setError("");
    try {
      accept(await action());
      setNotice(success);
      if (close) onClose();
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Unable to save resource.",
      );
    } finally {
      setBusy(false);
    }
  };
  const append = async () => {
    if (!current || pending) return;
    setMessageBusy(true);
    setMessageError("");
    try {
      const message = {
        ...messageDraft,
        id: randomUuid(),
        created_at: new Date().toISOString(),
      };
      const saved = await request<Resource>(
        `${agentPath(agentKey)}/resources/${current.id}/messages`,
        json("POST", { expected_version: current.version, message }),
      );
      setCurrent(saved);
      setDraft((previous) =>
        previous.kind === "conversation"
          ? { ...previous, messages: saved.messages }
          : previous,
      );
      setMessageDraft(emptyMessageDraft());
      onChanged();
    } catch (error) {
      setMessageError(
        error instanceof Error ? error.message : "Unable to append message.",
      );
    } finally {
      setMessageBusy(false);
    }
  };
  const close = () => {
    if (
      !pending &&
      (!(dirty || messageDirty) || window.confirm("Discard unsaved changes?"))
    )
      onClose();
  };
  const readOnly = Boolean(current?.deleted_at);
  return (
    <Dialog
      title={current ? current.title : `New ${singularLabels[draft.kind]}`}
      onClose={close}
      wide
    >
      <div className="dialog-scroll">
        <ErrorNotice message={error || loadError} />
        <div
          className="editor-tabs"
          role="tablist"
          aria-label="Resource editor"
        >
          {(["content", "details", "history"] as const).map((item) => (
            <button
              type="button"
              role="tab"
              aria-selected={tab === item}
              key={item}
              disabled={item === "history" && !current}
              onClick={() => setTab(item)}
            >
              {item === "history" && <History size={16} />}
              {
                { content: "Content", details: "Details", history: "History" }[
                  item
                ]
              }
            </button>
          ))}
        </div>
        {tab !== "history" && (
          <form
            id={formId}
            onSubmit={(event) => {
              event.preventDefault();
              if (pending) return;
              void mutate(
                () =>
                  request<Resource>(
                    `${agentPath(agentKey)}/resources${current ? `/${current.id}` : ""}`,
                    json(current ? "PUT" : "POST", {
                      ...draft,
                      ...(current ? { expected_version: current.version } : {}),
                    }),
                  ),
                "Saved.",
              );
            }}
          >
            <fieldset disabled={pending || readOnly} key={formKey}>
              <label>
                Title
                <input
                  required
                  maxLength={300}
                  value={draft.title}
                  onChange={(event) => update({ title: event.target.value })}
                />
              </label>
              {tab === "content" &&
                draft.kind !== "conversation" &&
                (draft.deck ? (
                  <DeckEditor
                    deck={draft.deck}
                    onChange={(deck) =>
                      setDraft((previous) =>
                        previous.kind === "deck"
                          ? { ...previous, deck }
                          : previous,
                      )
                    }
                    images={
                      data?.assets.filter((asset) => asset.is_image) || []
                    }
                    agentKey={agentKey}
                  />
                ) : (
                  <div className="markdown-editor">
                    <div className="segmented">
                      <button
                        type="button"
                        aria-pressed={markdownMode === "edit"}
                        onClick={() => setMarkdownMode("edit")}
                      >
                        Markdown
                      </button>
                      <button
                        type="button"
                        aria-pressed={markdownMode === "preview"}
                        onClick={() => setMarkdownMode("preview")}
                      >
                        Preview
                      </button>
                    </div>
                    {markdownMode === "edit" ? (
                      <textarea
                        className="document-textarea"
                        aria-label="Markdown content"
                        value={draft.markdown}
                        onChange={(event) =>
                          update({ markdown: event.target.value })
                        }
                      />
                    ) : (
                      <div className="document-preview">
                        {draft.markdown ? (
                          <Markdown text={draft.markdown} />
                        ) : (
                          <p className="muted">No Markdown content.</p>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              {tab === "details" && (
                <div className="metadata-fields">
                  {loading ? (
                    <Loading />
                  ) : (
                    <label>
                      Parent Space
                      <select
                        value={draft.parent_id || ""}
                        onChange={(event) =>
                          update({ parent_id: event.target.value || null })
                        }
                      >
                        <option value="">No parent</option>
                        {parentChoices(data?.resources || [], current?.id).map(
                          (resource) => (
                            <option key={resource.id} value={resource.id}>
                              {resource.title}
                            </option>
                          ),
                        )}
                      </select>
                    </label>
                  )}
                  <TagsField
                    value={draft.tags}
                    onChange={(tags) => update({ tags })}
                  />
                  <SourcesField
                    value={draft.sources}
                    onChange={(sources) => update({ sources })}
                  />
                  <SelectItems
                    label="Attachments"
                    items={data?.assets || []}
                    selected={draft.attachment_ids}
                    onChange={(attachment_ids) => update({ attachment_ids })}
                  />
                  <div className="link-list">
                    {draft.attachment_ids.map((id) => (
                      <a key={id} href={assetUrl(agentKey, id, true)}>
                        <Download size={14} />
                        {data?.assets.find((asset) => asset.id === id)?.title ||
                          id}
                      </a>
                    ))}
                  </div>
                  {(draft.kind === "deck" || draft.kind === "conversation") && (
                    <label>
                      Markdown notes
                      <textarea
                        rows={6}
                        value={draft.markdown}
                        onChange={(event) =>
                          update({ markdown: event.target.value })
                        }
                      />
                    </label>
                  )}
                  {current && (
                    <dl className="record-details">
                      <div>
                        <dt>ID</dt>
                        <dd className="mono">{current.id}</dd>
                      </div>
                      <div>
                        <dt>Version</dt>
                        <dd>{current.version}</dd>
                      </div>
                      <div>
                        <dt>Created</dt>
                        <dd>{formatDate(current.created_at)}</dd>
                      </div>
                      <div>
                        <dt>Updated</dt>
                        <dd>{formatDate(current.updated_at)}</dd>
                      </div>
                    </dl>
                  )}
                </div>
              )}
            </fieldset>
          </form>
        )}
        {tab === "content" && draft.kind === "conversation" && (
          <ConversationPanel
            agentKey={agentKey}
            messages={draft.messages}
            assets={data?.assets || []}
            resources={data?.resources || []}
            canAppend={Boolean(current) && !readOnly}
            draft={messageDraft}
            onChange={setMessageDraft}
            busy={pending}
            error={messageError}
            onAppend={append}
          />
        )}
        {tab === "history" && (
          <div className="revision-list">
            <ErrorNotice message={history.error} />
            {history.loading ? (
              <Loading />
            ) : (
              history.data?.map((revision) => (
                <details key={revision.version} className="revision">
                  <summary>
                    <span>Version {revision.version}</span>
                    <time>{formatDate(revision.created_at)}</time>
                  </summary>
                  <div className="revision-content">
                    <strong>{revision.snapshot.title}</strong>
                    <Markdown text={revision.snapshot.markdown} />
                    {revision.snapshot.deck?.slides.map((slide) => (
                      <SlidePreview
                        key={slide.id}
                        slide={slide}
                        agentKey={agentKey}
                      />
                    ))}
                    {revision.snapshot.messages.map((message) => (
                      <div className="message" key={message.id}>
                        <span className="muted">{message.role}</span>
                        <Markdown text={message.content} />
                      </div>
                    ))}
                    <SourceLinks sources={revision.snapshot.sources} />
                    <span className="muted">
                      {revision.snapshot.tags.join(", ")}
                    </span>
                    <button
                      type="button"
                      disabled={
                        busy ||
                        readOnly ||
                        revision.version === current?.version
                      }
                      onClick={() => {
                        if (
                          !current ||
                          (dirty &&
                            !window.confirm(
                              "Replace the unsaved draft with this version?",
                            ))
                        )
                          return;
                        void mutate(
                          () =>
                            request<Resource>(
                              `${agentPath(agentKey)}/resources/${current.id}/versions/${revision.version}/restore`,
                              json("POST", {
                                expected_version: current.version,
                              }),
                            ),
                          `Restored version ${revision.version}.`,
                        );
                      }}
                    >
                      <RotateCcw size={16} />
                      Restore this version
                    </button>
                  </div>
                </details>
              ))
            )}
          </div>
        )}
      </div>
      <div className="dialog-footer">
        <div className="footer-status" role="status">
          {pending
            ? messageBusy
              ? "Appending message…"
              : "Saving…"
            : messageDirty
              ? "Unsent message"
              : dirty
                ? "Unsaved changes"
                : readOnly
                  ? "In trash"
                  : notice || (current ? `Version ${current.version}` : "")}
        </div>
        {current && !readOnly && (
          <button
            className="danger"
            type="button"
            disabled={pending}
            onClick={() => {
              if (window.confirm(`Move “${current.title}” to trash?`))
                void mutate(
                  () =>
                    request<Resource>(
                      `${agentPath(agentKey)}/resources/${current.id}?expected_version=${current.version}`,
                      json("DELETE"),
                    ),
                  "",
                  true,
                );
            }}
          >
            <Trash2 size={16} />
            <span>Trash</span>
          </button>
        )}
        {current && readOnly ? (
          <button
            className="primary"
            disabled={pending}
            onClick={() =>
              void mutate(
                () =>
                  request<Resource>(
                    `${agentPath(agentKey)}/resources/${current.id}/restore`,
                    json("POST", { expected_version: current.version }),
                  ),
                "Restored.",
              )
            }
          >
            <RotateCcw size={16} />
            Restore
          </button>
        ) : (
          tab !== "history" && (
            <button
              className="primary"
              type="submit"
              form={formId}
              disabled={pending || !draft.title.trim()}
            >
              <Save size={16} />
              {busy ? "Saving…" : current ? "Save changes" : "Create"}
            </button>
          )
        )}
      </div>
    </Dialog>
  );
}
