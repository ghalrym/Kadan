import { ChevronRight, CircleAlert, CircleCheck, CircleStop, LoaderCircle } from "lucide-react";
import { conversationActivity, type ActivityItem, type ActivityTone } from "./activity";
import type { AgentRun, ImageJob, Resource } from "./types";

interface Props {
  conversation: Resource;
  runs: AgentRun[];
  images: ImageJob[];
  loading: boolean;
  error: string;
}

const GROUP_LIMIT = 10;

const ICONS: Record<ActivityTone, typeof CircleCheck> = {
  failed: CircleAlert,
  stopped: CircleStop,
  active: LoaderCircle,
  done: CircleCheck,
};

export default function Activity({ conversation, runs, images, loading, error }: Props) {
  const [review, progressing, finished, earlier] = conversationActivity(conversation, runs, images);
  const current = [...review.items, ...progressing.items];
  const empty = current.length + finished.items.length + earlier.items.length === 0;
  return (
    <section className="agent-card-activity" key={conversation.id} aria-labelledby="agent-card-activity">
      <header>
        <h3 id="agent-card-activity">Activity</h3>
        {!loading && progressing.items.length > 0 && <span className="activity-pill active">{progressing.items.length} in progress</span>}
        {!loading && review.items.length > 0 && <span className="activity-pill review">{review.items.length} to review</span>}
      </header>
      {error && <p className="activity-error">{loading ? "Couldn’t load activity." : "Couldn’t refresh. Showing the last update."}</p>}
      {loading ? !error && <p className="activity-empty">Loading activity…</p> : empty ? <p className="activity-empty">Nothing yet.</p> : (
        <div className="activity-content" role="region" aria-label="Conversation activity" tabIndex={0}>
          {current.length > 0 && renderItems(current)}
          {[finished, earlier].map(group => group.items.length > 0 && (
            <details key={group.title} className="activity-group">
              <summary>
                <ChevronRight size={14} aria-hidden="true" />
                {group.title}
                <span>{group.items.length}</span>
              </summary>
              {renderItems(group.items)}
            </details>
          ))}
        </div>
      )}
    </section>
  );
}

function renderItems(items: ActivityItem[]) {
  return <>
    <ul className="activity-list">{items.slice(0, GROUP_LIMIT).map(item => {
      const Icon = ICONS[item.tone];
      return <li key={`${item.kind}-${item.id}`} className={`activity-item ${item.tone}`}>
        <Icon size={16} className={item.tone === "active" ? "spin" : ""} aria-hidden="true" />
        <div>
          <div className="activity-line">
            <strong>{item.title}</strong>
            <time dateTime={item.updatedAt} title={new Date(item.updatedAt).toLocaleString()}>{ago(item.updatedAt)}</time>
          </div>
          <span className="activity-state">{item.state}</span>
          {item.detail && <p className="activity-detail">{item.detail}</p>}
        </div>
      </li>;
    })}</ul>
    {items.length > GROUP_LIMIT && <p className="activity-more">Showing {GROUP_LIMIT} of {items.length}.</p>}
  </>;
}

/** Compact age for a row: "now", "5m", "3h", "2d", then the date. */
function ago(iso: string, now = Date.now()) {
  const minutes = Math.max(0, (now - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${Math.floor(minutes)}m`;
  if (minutes < 60 * 24) return `${Math.floor(minutes / 60)}h`;
  if (minutes < 60 * 24 * 7) return `${Math.floor(minutes / 60 / 24)}d`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
