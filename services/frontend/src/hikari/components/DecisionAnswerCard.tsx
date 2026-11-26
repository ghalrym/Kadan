import type { Answer, Question } from '../api/decisions'

/** Whole-percent label that never rounds a nonzero probability down to 0%. */
const percent = (value: number) =>
  value > 0 && value < 0.005 ? '<1%' : `${Math.round(value * 100)}%`

/**
 * Horizontal probability bars. Rows keep the given order; the row whose key
 * matches `highlight` is accented as the model's answer.
 */
function Distribution({ rows, highlight }: { rows: { key: string; label: string; value: number }[]; highlight?: string }) {
  return (
    <ul className="answer-distribution" aria-label="Probabilities">
      {rows.map((row) => (
        <li key={row.key} className={row.key === highlight ? 'is-answer' : undefined}>
          <span className="answer-distribution-label" title={row.label}>{row.label}</span>
          <span className="answer-distribution-track" aria-hidden="true">
            <span style={{ width: `${Math.max(row.value * 100, row.value > 0 ? 1 : 0)}%` }} />
          </span>
          <span className="answer-distribution-value mono">{percent(row.value)}</span>
        </li>
      ))}
    </ul>
  )
}

/**
 * Render one typed answer next to the question it was evaluated with.
 * Choice: chosen option and its distribution. Score: expected (possibly
 * fractional) rubric index on a scale, nearest level and per-level odds.
 * Noul: probability of true with a verdict. Confidence shows when provided.
 */
export default function DecisionAnswerCard({ answer, question, stale }: { answer: Answer; question: Question; stale?: boolean }) {
  return (
    <article className={`answer-card${stale ? ' is-stale' : ''}`} aria-label={`Answer for ${answer.key}`}>
      <header className="answer-card-header">
        <span className="answer-key mono">{answer.key}</span>
        <span className={`type-badge type-badge--${answer.type}`}>{answer.type}</span>
        {answer.confidence != null && (
          <span className="answer-confidence mono" title="Model confidence in this answer">
            {percent(answer.confidence)} confidence
          </span>
        )}
      </header>
      {question.instructions && <p className="answer-instructions">{question.instructions}</p>}
      {answer.type === 'Choice' && question.type === 'Choice' && <ChoiceResult answer={answer} question={question} />}
      {answer.type === 'Score' && question.type === 'Score' && <ScoreResult answer={answer} question={question} />}
      {answer.type === 'Noul' && question.type === 'Noul' && <NoulResult answer={answer} question={question} />}
    </article>
  )
}

function ChoiceResult({ answer, question }: { answer: Extract<Answer, { type: 'Choice' }>; question: Extract<Question, { type: 'Choice' }> }) {
  const option = question.options.find((item) => item.key === answer.value)
  const probabilities = answer.probabilities
  return (
    <>
      <div className="answer-value">
        <strong>{answer.value}</strong>
        {option?.description && <span>{option.description}</span>}
      </div>
      {probabilities && (
        <Distribution
          highlight={answer.value}
          rows={question.options
            .map((item) => ({ key: item.key, label: item.key, value: probabilities[item.key] ?? 0 }))
            .sort((a, b) => b.value - a.value)}
        />
      )}
    </>
  )
}

function ScoreResult({ answer, question }: { answer: Extract<Answer, { type: 'Score' }>; question: Extract<Question, { type: 'Score' }> }) {
  const top = question.levels.length - 1
  const nearest = Math.min(Math.max(Math.round(answer.value), 0), top)
  const position = top > 0 ? (answer.value / top) * 100 : 0
  const probabilities = answer.probabilities
  return (
    <>
      <div className="answer-value">
        <strong>
          {Number.isInteger(answer.value) ? answer.value : answer.value.toFixed(2)}
          <small> / {top}</small>
        </strong>
        <span>Nearest level {nearest} · {question.levels[nearest]}</span>
      </div>
      <div className="answer-scale" role="img" aria-label={`Expected level ${answer.value.toFixed(2)} on a scale of 0 to ${top}`}>
        <div className="answer-scale-track">
          <span className="answer-scale-fill" style={{ width: `${position}%` }} />
          {question.levels.map((_, i) => (
            <span key={i} className="answer-scale-tick" style={{ left: `${top > 0 ? (i / top) * 100 : 0}%` }} />
          ))}
          <span className="answer-scale-marker" style={{ left: `${position}%` }} />
        </div>
        <div className="answer-scale-labels mono" aria-hidden="true">
          {question.levels.map((_, i) => (
            <span key={i} style={{ left: `${top > 0 ? (i / top) * 100 : 0}%` }}>{i}</span>
          ))}
        </div>
      </div>
      {probabilities && (
        <Distribution
          rows={question.levels.map((level, i) => ({ key: String(i), label: `${i} · ${level}`, value: probabilities[String(i)] ?? 0 }))}
        />
      )}
    </>
  )
}

function NoulResult({ answer, question }: { answer: Extract<Answer, { type: 'Noul' }>; question: Extract<Question, { type: 'Noul' }> }) {
  const verdict = answer.value >= 0.65 ? 'true' : answer.value <= 0.35 ? 'false' : 'uncertain'
  return (
    <>
      <div className="answer-value">
        <strong>{percent(answer.value)}<small> true</small></strong>
        <span className={`answer-verdict answer-verdict--${verdict}`}>
          {verdict === 'uncertain' ? 'Uncertain' : `Likely ${verdict}`}
        </span>
      </div>
      <div className="answer-meter" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={answer.value} aria-label="Probability true">
        <span style={{ width: `${answer.value * 100}%` }} />
      </div>
      {(question.trueWhen || question.falseWhen) && (
        <dl className="answer-criteria">
          {question.trueWhen && <><dt>True when</dt><dd>{question.trueWhen}</dd></>}
          {question.falseWhen && <><dt>False when</dt><dd>{question.falseWhen}</dd></>}
        </dl>
      )}
    </>
  )
}
