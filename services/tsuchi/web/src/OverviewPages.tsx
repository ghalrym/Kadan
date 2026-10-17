import {
  ArrowUpRight,
  Download,
  FileText,
  HardDrive,
  History,
  Image,
  Trash2,
} from "lucide-react";
import { agentPath, formatBytes, formatDate, request, useLoad } from "./api";
import { Empty, ErrorNotice, Loading } from "./components";
import { kindLabels } from "./types";
import type { Activity, Kind, Stats, View } from "./types";

export function ActivityList({ activity }: { activity: Activity[] }) {
  return activity.length === 0 ? (
    <Empty>No activity yet.</Empty>
  ) : (
    <ol className="activity-list">
      {activity.map((event) => (
        <li key={event.id}>
          <span className="activity-marker" />
          <div className="row-content">
            <span className="row-title">{event.title}</span>
            <span className="row-metadata">
              {event.action.replaceAll("_", " ")} · {event.entity_type}
            </span>
          </div>
          <time dateTime={event.created_at}>
            {formatDate(event.created_at)}
          </time>
        </li>
      ))}
    </ol>
  );
}
export function OverviewPage({
  agentKey,
  refresh,
  navigate,
}: {
  agentKey: string;
  refresh: number;
  navigate: (view: View) => void;
}) {
  const { data, loading, error } = useLoad(async () => {
    const [stats, activity] = await Promise.all([
      request<Stats>(`${agentPath(agentKey)}/stats`),
      request<Activity[]>(`${agentPath(agentKey)}/activity?limit=8`),
    ]);
    return { stats, activity };
  }, [agentKey, refresh]);
  return (
    <>
      <div className="page-heading">
        <h1>Overview</h1>
        <a className="button" href={`${agentPath(agentKey)}/export`}>
          <Download size={18} />
          Export namespace
        </a>
      </div>
      <ErrorNotice message={error} />
      {loading ? (
        <Loading />
      ) : (
        data && (
          <>
            <div className="stat-grid">
              <div className="stat">
                <div className="stat-label">
                  <HardDrive size={18} />
                  File storage
                </div>
                <strong>{formatBytes(data.stats.storage_bytes)}</strong>
                <span className="muted">Includes trash</span>
              </div>
              <div className="stat">
                <div className="stat-label">
                  <FileText size={18} />
                  Database · all namespaces
                </div>
                <strong>{formatBytes(data.stats.database_bytes)}</strong>
              </div>
              <button className="stat" onClick={() => navigate("attachments")}>
                <div className="stat-label">
                  <Image size={18} />
                  Files
                </div>
                <strong>{data.stats.assets.toLocaleString()}</strong>
                <span className="muted">
                  {data.stats.images.toLocaleString()} images
                </span>
              </button>
              <button className="stat" onClick={() => navigate("trash")}>
                <div className="stat-label">
                  <Trash2 size={18} />
                  Trash
                </div>
                <strong>
                  {(
                    data.stats.trashed_resources + data.stats.trashed_assets
                  ).toLocaleString()}
                </strong>
                <span className="muted">
                  {data.stats.trashed_resources} resources ·{" "}
                  {data.stats.trashed_assets} files
                </span>
              </button>
            </div>
            <div className="collection-grid">
              {(Object.entries(data.stats.resources) as [Kind, number][]).map(
                ([kind, count]) => (
                  <button key={kind} onClick={() => navigate(kind)}>
                    <span>{kindLabels[kind]}</span>
                    <strong>{count.toLocaleString()}</strong>
                    <ArrowUpRight size={17} />
                  </button>
                ),
              )}
            </div>
            <div className="activity-heading">
              <button
                className="text-button"
                onClick={() => navigate("activity")}
              >
                Recent activity
                <ArrowUpRight size={16} />
              </button>
              <span className="muted">
                <History size={15} />
                {data.stats.revisions.toLocaleString()} revisions
              </span>
            </div>
            <ActivityList activity={data.activity} />
          </>
        )
      )}
    </>
  );
}
export function ActivityPage({
  agentKey,
  refresh,
}: {
  agentKey: string;
  refresh: number;
}) {
  const { data, loading, error } = useLoad(
    () => request<Activity[]>(`${agentPath(agentKey)}/activity?limit=200`),
    [agentKey, refresh],
  );
  return (
    <>
      <div className="page-heading">
        <h1>Activity</h1>
        <span className="muted">Latest 200 events</span>
      </div>
      <ErrorNotice message={error} />
      {loading ? <Loading /> : data && <ActivityList activity={data} />}
    </>
  );
}
