import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  downloadModels,
  licenseNotices,
  llmModelId,
  llmVariantParts,
  modelsCanDownload,
  modelsComplete,
  modelsDownloading,
  modelsProgress,
  updateLlmModel,
  type CapabilityModel,
  type LlmQuant,
} from "../hikari/api/settings";
import { modelName, modelSize, STATE_LABEL, STATE_TONE, toAbility } from "./abilities";
import { formatBytes } from "./api";
import { PageHeader, StatusPill } from "./common";
import Subscriptions from "./Subscriptions";
import { useModels } from "./useModels";

function Dialog({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    return () => element?.close();
  }, []);
  return (
    <dialog ref={dialog} className="srv-dialog" aria-labelledby="srv-dialog-title"
      onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <h2 id="srv-dialog-title">{title}</h2>
      {children}
    </dialog>
  );
}

function Choice<T extends string | boolean>({ label, value, options, disabled, onChange }: {
  label: string;
  value: T;
  options: [T, string, string][];
  disabled: boolean;
  onChange: (value: T) => void;
}) {
  return (
    <fieldset className="srv-choice" disabled={disabled}>
      <legend>{label}</legend>
      {options.map(([option, title, hint]) => (
        <label key={String(option)} className={option === value ? "selected" : ""}>
          <input type="radio" name={label} checked={option === value} onChange={() => onChange(option)} />
          <span>
            <strong>{title}</strong>
            <small>{hint}</small>
          </span>
        </label>
      ))}
    </fieldset>
  );
}

export default function ModelsPage() {
  const models = useModels();
  const [confirm, setConfirm] = useState<CapabilityModel[] | null>(null);
  const [accepted, setAccepted] = useState(false);
  const rows = models.rows;
  const abilities = rows?.map(toAbility) ?? [];
  const ready = abilities.filter((ability) => ability.state === "ready").length;
  const downloading = !!rows && modelsDownloading(rows);
  const canDownload = !!rows && modelsCanDownload(rows);
  const complete = !!rows && modelsComplete(rows);
  const totalBytes = abilities.reduce((sum, ability) => sum + ability.sizeBytes, 0);
  const blocked = models.submitting || models.needsRefresh;

  const pending = confirm ?? [];
  const notices = licenseNotices(pending);
  const pendingBytes = pending.reduce((sum, row) => sum + modelSize(row.model), 0);

  function ask(selection: CapabilityModel[]) {
    setAccepted(false);
    setConfirm(selection.filter((row) => modelsCanDownload([row])));
  }

  const language = rows?.find((row) => row.capability.id === "inkling-2-mini")?.model ?? null;
  const variant = language ? llmVariantParts(language.id) : null;
  const languageAbility = abilities.find((ability) => ability.id === "inkling-2-mini");
  const languageBusy = languageAbility?.state === "downloading" || languageAbility?.state === "waiting";
  function switchLanguage(quant: LlmQuant, abliterated: boolean) {
    const next = llmModelId(quant, abliterated);
    if (!language || next === language.id) return;
    void models.run((signal) => updateLlmModel(next, signal), "Couldn’t change the chat model.");
  }

  const problem = (models.actionError || models.error) && (
    <div className="banner srv-banner" role="alert">
      <span>
        {models.actionError || models.error}
        {models.needsRefresh && " Some downloads may have started anyway. Check again before retrying."}
      </span>
      <button disabled={models.submitting} onClick={models.reload}>Check again</button>
    </div>
  );

  return (
    <div className="srv-page">
      <PageHeader
        title="AI models"
        description="Kadan runs these models on this computer. Download them once and they stay ready."
        actions={rows && !complete && (
          <button className="primary" disabled={blocked || !canDownload} onClick={() => ask(rows)}>
            {downloading ? `Downloading… ${modelsProgress(rows)}%` : canDownload ? "Download everything" : "Nothing to download"}
          </button>
        )}
      />
      {!confirm && problem}
      {!rows && !models.error && <p className="srv-loading">Loading AI models…</p>}

      {rows && (
        <section className="srv-card">
          <header className="srv-card-head">
            <h2>{complete ? "All abilities are ready" : `${ready} of ${abilities.length} abilities ready`}</h2>
            <p>{formatBytes(totalBytes)} of disk space when everything is downloaded.</p>
          </header>
          <ul className="srv-model-list">
            {abilities.map((ability, index) => {
              const row = rows[index];
              return (
                <li key={ability.id} className="srv-model">
                  <div className="srv-model-text">
                    <strong>{ability.name}</strong>
                    <span>{ability.description}</span>
                    {ability.modelName && (
                      <small title={ability.model?.license ? `License: ${ability.model.license}` : undefined}>
                        {ability.modelName} · {formatBytes(ability.sizeBytes)}
                      </small>
                    )}
                  </div>
                  <div className="srv-model-side">
                    {ability.state === "needs_download" || ability.state === "failed" ? (
                      <button className="button" disabled={blocked} onClick={() => ask([row])}>
                        {ability.state === "failed" ? "Try again" : "Download"}
                      </button>
                    ) : (
                      <StatusPill tone={STATE_TONE[ability.state]}>
                        {ability.state === "downloading" ? `${ability.progress}%` : STATE_LABEL[ability.state]}
                      </StatusPill>
                    )}
                  </div>
                  {(ability.state === "downloading" || ability.state === "waiting") && (
                    <progress className="srv-progress" aria-label={`${ability.name} download`}
                      max={100} value={ability.state === "waiting" ? undefined : ability.progress} />
                  )}
                  {ability.state === "not_running" && (
                    <p className="field-hint srv-model-note">
                      This part of Kadan Server isn’t running, so Kadan can’t check this model. Restart Kadan Server; if it
                      keeps happening, open Logs in the Kadan Server window.
                    </p>
                  )}
                  {ability.error && <p className="error srv-model-note" role="alert">{ability.error}</p>}
                </li>
              );
            })}
          </ul>
        </section>
      )}

      <Subscriptions />

      {variant && (
        <details className="srv-card srv-advanced">
          <summary>
            <span>
              <strong>Advanced</strong>
              <small>Change how the chat model trades size for quality.</small>
            </span>
          </summary>
          <div className="srv-advanced-body">
            <Choice<LlmQuant>
              label="Chat model size"
              value={variant.quant}
              disabled={blocked || languageBusy}
              options={[
                ["q4", "Smaller", "Fits on a 24 GB graphics card and answers quickly."],
                ["q8", "Higher quality", "Slightly better answers; needs more memory and disk space."],
              ]}
              onChange={(quant) => switchLanguage(quant, variant.abliterated)}
            />
            <Choice<boolean>
              label="Content filter"
              value={variant.abliterated}
              disabled={blocked || languageBusy}
              options={[
                [false, "Standard", "The model may decline some requests."],
                [true, "Unfiltered", "The model answers more freely. You’re responsible for how it’s used."],
              ]}
              onChange={(abliterated) => switchLanguage(variant.quant, abliterated)}
            />
            <p className="field-hint">A different choice may need its own download before Kadan can use it.</p>
          </div>
        </details>
      )}

      {confirm && (
        <Dialog title={pending.length === 1 ? `Download ${toAbility(pending[0]).name}?` : "Download AI models?"}
          onClose={() => !models.submitting && setConfirm(null)}>
          <p className="field-hint">
            This downloads {pending.length} {pending.length === 1 ? "model" : "models"}, about {formatBytes(pendingBytes)}.
            You can keep using Kadan while it runs.
          </p>
          {notices.map((model) => (
            <p key={model.id} className="srv-license">
              <strong>{modelName(model)}:</strong> {model.license_notice}{" "}
              {model.license_url && <a href={model.license_url} target="_blank" rel="noreferrer">Read the license</a>}
            </p>
          ))}
          {notices.length > 0 && (
            <label className="srv-check">
              <input type="checkbox" checked={accepted} disabled={models.submitting}
                onChange={(event) => setAccepted(event.target.checked)} />
              I’ve read and accept these licenses.
            </label>
          )}
          {problem}
          <div className="form-actions">
            <button autoFocus disabled={models.submitting} onClick={() => setConfirm(null)}>Cancel</button>
            <button className="primary"
              disabled={blocked || pending.length === 0 || (notices.length > 0 && !accepted)}
              onClick={() => void models.run((signal) => downloadModels(pending, true, signal), "Couldn’t start the download.")
                .then((ok) => ok && setConfirm(null))}>
              {models.submitting ? "Starting…" : notices.length ? "Accept and download" : "Download"}
            </button>
          </div>
        </Dialog>
      )}
    </div>
  );
}
