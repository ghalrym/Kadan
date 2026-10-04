import { Link, useParams } from 'react-router'
import { ActionButton } from './Controls'
import { AudioCard, MediaPlaceholder } from './Media'
import {
  requests,
  statusClass,
  statusLabels,
  type RequestRecord,
} from '../data/requests'
import { voiceDescription } from '../data/playground'

function DetailSection({ title, text }: { title: string; text: string }) {
  return (
    <section className="stack compact">
      <h3 className="eyebrow">{title}</h3>
      <div className="detail-text">{text}</div>
    </section>
  )
}

function DecisionResult() {
  return (
    <section className="stack compact">
      <h3 className="eyebrow">Questions</h3>
      <div className="detail-text stack compact">
        <div className="row">
          <strong className="mono">route</strong>
          <span className="badge">Choice</span>
          <span className="push-right faint mono">confidence 0.71</span>
        </div>
        <p>Which team should handle this</p>
        <strong>tier_1</strong>
        <div className="probability-row">
          <span>tier_1</span>
          <progress
            className="progress"
            value={84}
            max={100}
            aria-label="Tier 1 probability"
          />
          <span>84.0%</span>
        </div>
        <div className="probability-row">
          <span>tier_2</span>
          <progress
            className="progress progress--muted"
            value={16}
            max={100}
            aria-label="Tier 2 probability"
          />
          <span>16.0%</span>
        </div>
      </div>
      <div className="detail-text stack compact">
        <div className="row">
          <strong className="mono">urgency</strong>
          <span className="badge">Score</span>
          <span className="push-right faint mono">confidence 0.66</span>
        </div>
        <p>How urgent the request is</p>
        <strong>1.32 · Medium</strong>
      </div>
      <div className="detail-text stack compact">
        <div className="row">
          <strong className="mono">needs_refund</strong>
          <span className="badge">Noul</span>
        </div>
        <p>The customer is asking for a refund</p>
        <strong>12.0% likely true</strong>
        <progress
          className="progress"
          value={12}
          max={100}
          aria-label="Refund probability"
        />
      </div>
    </section>
  )
}

function RequestContent({ request }: { request: RequestRecord }) {
  const ok = request.status < 400
  switch (request.type) {
    case 'LLM':
      return (
        <section className="stack compact">
          <h3 className="eyebrow">Conversation</h3>
          <DetailSection title="System" text="You are a helpful assistant." />
          <DetailSection title="User" text={request.prompt} />
          {ok && (
            <DetailSection title="Assistant · output" text={request.output} />
          )}
        </section>
      )
    case 'STT':
      return (
        <>
          <h3 className="eyebrow">Recording</h3>
          <AudioCard
            voice={request.prompt}
            meta="WAV · 24 kHz"
            script="Recording"
            time="0:12"
          />
          {ok && <DetailSection title="Transcript" text={request.output} />}
        </>
      )
    case 'TTS':
      return (
        <>
          <DetailSection title="Voice · described" text={voiceDescription} />
          <DetailSection title="Script" text={request.prompt} />
          {ok && (
            <>
              <h3 className="eyebrow">Output audio</h3>
              <AudioCard
                voice={request.output}
                meta="F5-TTS · WAV · 24 kHz"
                script={request.prompt}
                time="0:03"
              />
            </>
          )}
        </>
      )
    case 'Image':
    case 'Video':
      return (
        <>
          <DetailSection
            title="Parameters"
            text={`${request.endpoint.includes('edits') ? 'Instruction' : 'Prompt'}: ${request.prompt}\nModel: ${request.model}\n${request.type === 'Image' ? 'Size: 1024×1024\nSeed: 48213' : 'Duration: 8s\nResolution: 720p\nFrame rate: 24 fps'}`}
          />
          {request.endpoint.includes('edits') && (
            <section className="stack compact">
              <h3 className="eyebrow">Source image</h3>
              <MediaPlaceholder aspect="square" label="source image" />
            </section>
          )}
          {ok && (
            <section className="stack compact">
              <h3 className="eyebrow">Output</h3>
              <MediaPlaceholder
                aspect={request.type === 'Image' ? 'square' : 'wide'}
                label={request.output}
              />
            </section>
          )}
        </>
      )
    case 'Decision':
      return (
        <>
          <DetailSection title="State" text={request.prompt} />
          {ok && <DecisionResult />}
        </>
      )
  }
}

export default function RequestDetails() {
  const { requestId } = useParams()
  const request = requests.find((item) => item.id === requestId)
  return (
    <div className="detail-overlay">
      <Link
        to="/requests"
        className="detail-backdrop"
        aria-label="Close request details"
        tabIndex={-1}
      />
      <aside className="request-drawer" aria-label="Request details">
        {request ? (
          <>
            <header className="drawer-header">
              <div className="row wrap">
                <span className={`badge type-${request.type}`}>
                  {request.type}
                </span>
                <span
                  className={`badge status-badge ${statusClass(request.status)}`}
                >
                  {request.status} {statusLabels[request.status]}
                </span>
                <Link to="/requests" className="button push-right">
                  Close
                </Link>
              </div>
              <p className="mono endpoint">
                <span className="muted">POST</span> {request.endpoint}
              </p>
            </header>
            <div className="drawer-content stack">
              <div className="detail-metrics mono muted">
                <span>
                  Model <strong>{request.model}</strong>
                </span>
                <span>
                  Latency <strong>{request.latency}</strong>
                </span>
                {request.ttft && (
                  <span>
                    TTFT <strong>{request.ttft}</strong>
                  </span>
                )}
                {request.tokensPerSecond && (
                  <span>
                    Tok/s <strong>{request.tokensPerSecond}</strong>
                  </span>
                )}
              </div>
              <RequestContent request={request} />
              {request.status >= 400 && (
                <div className="error-panel stack compact">
                  <h3>
                    {request.status} ·{' '}
                    {request.status === 500
                      ? 'internal_error'
                      : 'rate_limit_exceeded'}
                  </h3>
                  <p>{request.output}</p>
                </div>
              )}
              <details className="advanced-details">
                <summary>Advanced</summary>
                <div className="stack">
                  <h3 className="eyebrow">Request</h3>
                  <pre>
                    {JSON.stringify(
                      {
                        id: request.id,
                        model: request.model,
                        input: request.prompt,
                      },
                      null,
                      2,
                    )}
                  </pre>
                  <h3 className="eyebrow">Response</h3>
                  <pre>
                    {JSON.stringify(
                      { status: request.status, output: request.output },
                      null,
                      2,
                    )}
                  </pre>
                </div>
              </details>
            </div>
            <footer className="drawer-footer row wrap">
              <ActionButton>Copy as cURL</ActionButton>
              <ActionButton>Copy ID</ActionButton>
              <ActionButton>Replay request</ActionButton>
            </footer>
          </>
        ) : (
          <div className="empty-state stack">
            <h2>Request not found</h2>
            <Link to="/requests">Back to Requests</Link>
          </div>
        )}
      </aside>
    </div>
  )
}
