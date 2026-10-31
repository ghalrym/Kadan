import { conversationActivity, type ActivityItem } from "./activity";
import type { AgentRun, ImageJob, Resource } from "./types";

interface Props {
  conversation: Resource;
  runs: AgentRun[];
  images: ImageJob[];
  loading: boolean;
  error: string;
}

const GROUP_LIMIT = 10;

export default function Activity({ conversation, runs, images, loading, error }: Props) {
  const groups = conversationActivity(conversation, runs, images);
  const review = groups[0].items.length;
  const progressing = groups[1].items.length;
  const total = groups.reduce((count, group) => count + group.items.length, 0);
  return (
    <details className="chat-activity" key={conversation.id}>
      <summary>
        Activity
        <span>{loading ? error ? "Unavailable" : "Loading…" : `${progressing} in progress · ${review} to review · ${groups[3].items.length} earlier failures/stops`}{!loading && error ? " · Last received" : ""}</span>
      </summary>
      <div className="activity-content" role="region" aria-label="Conversation activity" tabIndex={0}>
        <p className="quiet">Response and image work in this conversation. Codex job status is shown separately for the shared computer.</p>
        {error && <p className="error">Updates unavailable. {loading ? "Activity has not loaded." : "Showing the last received activity."}</p>}
        {loading ? (!error && <p className="quiet">Loading activity…</p>) : total === 0 ? <p className="quiet">No response or image activity yet.</p> : [groups[0], groups[1], groups[3], groups[2]].map(group => group.items.length > 0 && (
          <section key={group.title} aria-label={group.title}>
            {group.historical ? <details>
              <summary>{group.title} · {group.items.length}</summary>
              <p className="quiet">Earlier does not mean resolved. These records are kept without retrying them.</p>
              {renderItems(group.items)}
            </details> : <>
            <h3>{group.title} <span className="quiet">{group.items.length}</span></h3>
            {renderItems(group.items)}
            </>}
          </section>
        ))}
        <p className="quiet">Failures and stops are kept for review; they are not retried automatically. Use the conversation’s Stop controls for active work.</p>
      </div>
    </details>
  );
}

function renderItems(items: ActivityItem[]) {
  return <>
    <ul>{items.slice(0, GROUP_LIMIT).map(item => <li key={`${item.kind}-${item.id}`}>
      <strong>{item.title}</strong>
      <span className={`activity-state ${/failed|stopped|cancelled/.test(item.state) ? "review" : ""}`}>{item.state}</span>
      {item.detail && <p className="quiet">{item.detail}</p>}
      <time className="quiet" dateTime={item.updatedAt}>Updated {new Date(item.updatedAt).toLocaleString()}</time>
    </li>)}</ul>
    {items.length > GROUP_LIMIT && <p className="quiet">Showing the {GROUP_LIMIT} most recently updated of {items.length}.</p>}
  </>;
}
