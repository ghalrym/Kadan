import { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Check, Search, Settings, Star } from "lucide-react";
import ProviderIcon from "./ProviderIcon";
import { usePopover } from "./popover";
import { harnessAuthLabel, type WorkHarness, type WorkHarnessModel } from "./work";

const FAVORITES_KEY = "kadan:work:favoriteModels";
const FAVORITES_RAIL = "favorites";

const EFFORT_LABELS: Record<string, string> = {
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "Extra High",
  max: "Max",
};

function effortLabel(level: string): string {
  return EFFORT_LABELS[level] ?? level.charAt(0).toUpperCase() + level.slice(1);
}

/** The model's reasoning setting, e.g. `effort` or `reasoning_effort`. */
function effortKey(model: WorkHarnessModel | null | undefined): string | null {
  if (!model?.options) return null;
  const key = Object.keys(model.options).find((k) => /effort|reasoning/i.test(k));
  return key && (model.options[key]?.length ?? 0) > 0 ? key : null;
}

function defaultEffort(levels: string[]): string | null {
  if (levels.includes("medium")) return "medium";
  return levels[0] ?? null;
}

function harnessModels(harness: WorkHarness): WorkHarnessModel[] {
  return harness.models.length > 0
    ? harness.models
    : [{ id: harness.default_model ?? "", label: harness.display_name, description: "", options: {} }];
}

function modelKey(harnessId: string, modelId: string | null | undefined): string {
  return `${harnessId}::${modelId ?? ""}`;
}

function readFavorites(): Set<string> {
  try {
    const raw = typeof localStorage !== "undefined" ? localStorage.getItem(FAVORITES_KEY) : null;
    const parsed = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(parsed) ? parsed.filter((v) => typeof v === "string") : []);
  } catch {
    return new Set();
  }
}

function writeFavorites(favorites: Set<string>): void {
  try {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites]));
  } catch {
    // Favorites are a convenience; ignore storage errors.
  }
}

interface Item {
  key: string;
  harness: WorkHarness;
  model: WorkHarnessModel;
  ready: boolean;
}

interface Props {
  harnesses: WorkHarness[];
  harnessId: string;
  modelId: string | null;
  options: Record<string, unknown>;
  disabled?: boolean;
  onChange: (harnessId: string, modelId: string | null, options: Record<string, unknown>) => void;
  onOpenSettings?: () => void;
}

export default function ModelPicker({
  harnesses,
  harnessId,
  modelId,
  options,
  disabled,
  onChange,
  onOpenSettings,
}: Props) {
  const [open, setOpen] = useState(false);
  const [rail, setRail] = useState(harnessId);
  const [query, setQuery] = useState("");
  const [highlight, setHighlight] = useState(0);
  const [favorites, setFavorites] = useState<Set<string>>(() => readFavorites());
  const [indicatorTop, setIndicatorTop] = useState<number | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const railRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const close = useCallback(() => setOpen(false), []);
  const placement = usePopover(open, close, rootRef, 420);

  const currentHarness = harnesses.find((h) => h.id === harnessId) ?? null;
  const currentModel =
    (currentHarness &&
      (harnessModels(currentHarness).find((m) => (m.id || null) === modelId) ??
        harnessModels(currentHarness)[0])) ||
    null;
  const currentKey = currentHarness ? modelKey(currentHarness.id, currentModel?.id || null) : "";
  const currentEffortKey = effortKey(currentModel);
  const currentEffort =
    currentEffortKey && typeof options[currentEffortKey] === "string"
      ? (options[currentEffortKey] as string)
      : null;

  const allItems = useMemo<Item[]>(
    () =>
      harnesses.flatMap((h) =>
        harnessModels(h).map((m) => ({
          key: modelKey(h.id, m.id || null),
          harness: h,
          model: m,
          ready: h.auth.status === "ready",
        })),
      ),
    [harnesses],
  );

  const needle = query.trim().toLowerCase();
  const searching = needle.length > 0;
  const items = searching
    ? allItems.filter(
        (item) =>
          item.model.label.toLowerCase().includes(needle) ||
          item.model.id.toLowerCase().includes(needle) ||
          item.harness.display_name.toLowerCase().includes(needle),
      )
    : rail === FAVORITES_RAIL
      ? allItems.filter((item) => favorites.has(item.key))
      : allItems.filter((item) => item.harness.id === rail);

  const railHarness = harnesses.find((h) => h.id === rail) ?? null;
  const activeItem = items[highlight] ?? allItems.find((i) => i.key === currentKey) ?? null;
  const activeEffortKey = effortKey(activeItem?.model);
  const activeLevels = activeEffortKey ? (activeItem?.model.options[activeEffortKey] ?? []) : [];
  const activeDefault = defaultEffort(activeLevels);
  const activeSelectedEffort =
    activeItem?.key === currentKey && currentEffort ? currentEffort : activeDefault;

  useLayoutEffect(() => {
    if (!open || !railRef.current) return;
    const target = railRef.current.querySelector<HTMLElement>(
      `[data-model-picker-provider="${searching ? "" : rail}"]`,
    );
    setIndicatorTop(target ? target.offsetTop + target.offsetHeight / 2 - 10 : null);
  }, [open, rail, searching, harnesses.length]);

  function openPicker() {
    if (disabled) return;
    const startRail = harnessId || harnesses[0]?.id || FAVORITES_RAIL;
    setRail(startRail);
    setQuery("");
    const startItems = allItems.filter((i) => i.harness.id === startRail);
    setHighlight(Math.max(0, startItems.findIndex((i) => i.key === currentKey)));
    setOpen(true);
    requestAnimationFrame(() => searchRef.current?.focus());
  }

  function selectRail(id: string) {
    setRail(id);
    setQuery("");
    const nextItems =
      id === FAVORITES_RAIL
        ? allItems.filter((i) => favorites.has(i.key))
        : allItems.filter((i) => i.harness.id === id);
    setHighlight(Math.max(0, nextItems.findIndex((i) => i.key === currentKey)));
    searchRef.current?.focus();
  }

  function choose(item: Item, effort?: string) {
    if (!item.ready) return;
    const key = effortKey(item.model);
    let nextOptions: Record<string, unknown> = {};
    if (key) {
      const levels = item.model.options[key] ?? [];
      const keep =
        effort ??
        (item.key === currentKey && currentEffort && levels.includes(currentEffort)
          ? currentEffort
          : null);
      if (keep && levels.includes(keep)) nextOptions = { [key]: keep };
    }
    onChange(item.harness.id, item.model.id || null, nextOptions);
    setOpen(false);
  }

  function toggleFavorite(key: string) {
    setFavorites((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      writeFavorites(next);
      return next;
    });
  }

  function onSearchKey(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setHighlight((i) => Math.min(items.length - 1, i + 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlight((i) => Math.max(0, i - 1));
    } else if (event.key === "Enter") {
      event.preventDefault();
      const item = items[highlight];
      if (item) choose(item);
    } else if ((event.ctrlKey || event.metaKey) && /^[1-9]$/.test(event.key)) {
      const item = items[Number(event.key) - 1];
      if (item) {
        event.preventDefault();
        choose(item);
      }
    }
  }

  if (harnesses.length === 0) return null;

  const triggerName = currentModel?.label ?? currentHarness?.display_name ?? "Choose model";
  const triggerEffort = currentEffortKey
    ? effortLabel(currentEffort ?? defaultEffort(currentModel?.options[currentEffortKey] ?? []) ?? "")
    : "";
  const statusHarness = activeItem?.harness ?? railHarness ?? currentHarness;
  const statusReady = statusHarness?.auth.status === "ready";

  return (
    <div className="model-picker" ref={rootRef}>
      <button
        type="button"
        className={`model-picker-trigger ${open ? "is-open" : ""}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`Model: ${triggerName}${triggerEffort ? ` ${triggerEffort}` : ""}`}
        disabled={disabled}
        onClick={() => (open ? setOpen(false) : openPicker())}
      >
        {currentHarness && <ProviderIcon harness={currentHarness.id} size={16} />}
        <span className="model-picker-trigger-name">{triggerName}</span>
        {triggerEffort && <span className="model-picker-trigger-effort">{triggerEffort}</span>}
      </button>

      {open && (
        <div
          className={`model-picker-pop is-${placement}`}
          role="dialog"
          aria-label="Choose a model"
          data-model-picker-content="true"
        >
          <div className="model-picker-body">
            <div className="model-picker-rail" ref={railRef} role="toolbar" aria-orientation="vertical" aria-label="Assistants">
              {indicatorTop !== null && (
                <span className="model-picker-indicator" style={{ top: indicatorTop }} aria-hidden="true" />
              )}
              <div data-model-picker-provider={FAVORITES_RAIL}>
                <button
                  type="button"
                  className="model-picker-rail-btn"
                  aria-label="Favorites"
                  title="Favorites"
                  aria-pressed={!searching && rail === FAVORITES_RAIL}
                  onClick={() => selectRail(FAVORITES_RAIL)}
                >
                  <Star size={18} fill="currentColor" />
                </button>
              </div>
              <span className="model-picker-rail-divider" aria-hidden="true" />
              {harnesses.map((h) => {
                const ready = h.auth.status === "ready";
                const label = ready ? h.display_name : `${h.display_name} · ${harnessAuthLabel(h.auth, h.installed)}`;
                return (
                  <div key={h.id} data-model-picker-provider={h.id}>
                    <button
                      type="button"
                      className={`model-picker-rail-btn ${ready ? "" : "is-unavailable"}`}
                      aria-label={label}
                      title={label}
                      aria-pressed={!searching && rail === h.id}
                      onClick={() => selectRail(h.id)}
                    >
                      <ProviderIcon harness={h.id} size={20} />
                    </button>
                  </div>
                );
              })}
            </div>

            <div className="model-picker-main">
              <label className="model-picker-search">
                <Search size={15} aria-hidden="true" />
                <input
                  ref={searchRef}
                  type="text"
                  placeholder="Search models..."
                  aria-label="Search models"
                  value={query}
                  onChange={(event) => {
                    setQuery(event.target.value);
                    setHighlight(0);
                  }}
                  onKeyDown={onSearchKey}
                />
              </label>
              <div className="model-picker-list" role="listbox" aria-label="Models">
                {!searching && railHarness && railHarness.auth.status !== "ready" && (
                  <div className="model-picker-notice">
                    <span>
                      {railHarness.display_name} isn't ready yet.{" "}
                      {railHarness.auth.message || harnessAuthLabel(railHarness.auth, railHarness.installed)}
                    </span>
                    {onOpenSettings && (
                      <button
                        type="button"
                        className="model-picker-link"
                        onClick={() => {
                          setOpen(false);
                          onOpenSettings();
                        }}
                      >
                        Open settings
                      </button>
                    )}
                  </div>
                )}
                {items.length === 0 ? (
                  <p className="model-picker-empty">
                    {!searching && rail === FAVORITES_RAIL ? "No favorites yet" : "No models found"}
                  </p>
                ) : (
                  items.map((item, index) => {
                    const selected = item.key === currentKey;
                    const favorite = favorites.has(item.key);
                    return (
                      <div
                        key={item.key}
                        role="option"
                        aria-selected={selected}
                        aria-disabled={!item.ready}
                        className={`model-picker-item ${index === highlight ? "is-highlighted" : ""} ${
                          item.ready ? "" : "is-disabled"
                        }`}
                        onMouseEnter={() => setHighlight(index)}
                        onClick={() => choose(item)}
                      >
                        <ProviderIcon harness={item.harness.id} size={16} />
                        <span className="model-picker-item-name">{item.model.label}</span>
                        {(searching || rail === FAVORITES_RAIL) && (
                          <span className="model-picker-item-provider">{item.harness.display_name}</span>
                        )}
                        {selected && <Check size={14} className="model-picker-check" aria-label="Selected" />}
                        <span className="model-picker-item-end">
                          <button
                            type="button"
                            className={`model-picker-star ${favorite ? "is-on" : ""}`}
                            aria-label={favorite ? `Remove ${item.model.label} from favorites` : `Add ${item.model.label} to favorites`}
                            onClick={(event) => {
                              event.stopPropagation();
                              toggleFavorite(item.key);
                            }}
                          >
                            <Star size={14} fill={favorite ? "currentColor" : "none"} />
                          </button>
                          {index < 9 && <kbd className="model-picker-kbd">Ctrl+{index + 1}</kbd>}
                        </span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            <div className="model-picker-effort">
              <div className="model-picker-effort-head">
                <span>Reasoning</span>
                {activeItem && <span className="model-picker-effort-model">{activeItem.model.label}</span>}
              </div>
              {activeLevels.length === 0 ? (
                <p className="model-picker-effort-empty">No reasoning options for this model.</p>
              ) : (
                <div className="model-picker-effort-list">
                  {activeLevels.map((level) => (
                    <button
                      key={level}
                      type="button"
                      className="model-picker-effort-item"
                      disabled={!activeItem?.ready}
                      aria-pressed={activeSelectedEffort === level}
                      onClick={() => activeItem && choose(activeItem, level)}
                    >
                      <span>{effortLabel(level)}</span>
                      {level === activeDefault && <span className="model-picker-default">Default</span>}
                      {activeSelectedEffort === level && <Check size={15} className="model-picker-check" />}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="model-picker-foot">
            <span className="model-picker-status">
              <span className={`dot ${statusReady ? "ready" : "warn"}`} aria-hidden="true" />
              {statusHarness
                ? `${statusHarness.display_name} · ${harnessAuthLabel(statusHarness.auth, statusHarness.installed)}`
                : ""}
              {statusHarness?.auth.account_label && (
                <span className="model-picker-account">{statusHarness.auth.account_label}</span>
              )}
            </span>
            {onOpenSettings && (
              <button
                type="button"
                className="model-picker-link"
                onClick={() => {
                  setOpen(false);
                  onOpenSettings();
                }}
              >
                <Settings size={13} />
                Manage assistants
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
