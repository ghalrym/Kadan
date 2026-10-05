import { useEffect, useRef, useState } from 'react'
import { Panel, SectionHeading } from '../components/Controls'
import DecisionQuestionCard from '../components/DecisionQuestionCard'
import DecisionAnswerCard from '../components/DecisionAnswerCard'
import {
  DecisionError,
  requestDecisions,
  validateDecisionRequest,
  type Answer,
  type DecisionIssue,
  type Question,
} from '../api/decisions'

const MAX_QUESTIONS = 8
const localFailure = new DecisionError('Check your inputs', 'Some fields need attention before evaluating.')

/** The inputs an answer set was produced from, so results stay readable while editing. */
type Evaluation = { state: string; questions: Question[]; answers: Answer[]; elapsedMs: number }

/**
 * Edit page-local decision state and display answers from the CPU Laya API.
 * Only the active request may update results; leaving the page aborts that request.
 * Answers render against the snapshot they were evaluated with and are marked
 * outdated, not discarded, when inputs change.
 */
export default function DecisionsPage() {
  const [state, setState] = useState('')
  const [questions, setQuestions] = useState<Question[]>([])
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [pending, setPending] = useState<{ questions: Question[]; startedAt: number } | null>(null)
  const [failure, setFailure] = useState<DecisionError | null>(null)
  const [checkInputs, setCheckInputs] = useState(false)
  const [notice, setNotice] = useState('')
  const [now, setNow] = useState(() => Date.now())
  const active = useRef<AbortController | null>(null)
  const answersPanel = useRef<HTMLDivElement | null>(null)
  useEffect(
    () => () => {
      active.current?.abort()
      active.current = null
    },
    [],
  )
  useEffect(() => {
    if (!pending) return
    const timer = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(timer)
  }, [pending])

  // After a rejected submit, recheck locally on every edit so fixed fields clear at once.
  const localIssues = checkInputs ? validateDecisionRequest(state, questions) : []
  const issues: DecisionIssue[] = localIssues.length ? localIssues : (failure?.issues ?? [])
  const shownFailure = localIssues.length ? localFailure : failure
  const pageIssues = issues.filter((issue) => issue.question === undefined)
  const stale = !!evaluation && (evaluation.state !== state || JSON.stringify(evaluation.questions) !== JSON.stringify(questions))

  /** Record an input edit: server field errors may no longer apply, runtime failures still do. */
  function edited() {
    if (failure && !failure.retryable) setFailure(null)
    setNotice('')
  }
  /** Append an empty question of the selected type. */
  function add(type: Question['type']) {
    const base = { key: '', instructions: '' }
    const question: Question =
      type === 'Choice'
        ? { ...base, type, options: [{ key: '', description: '' }] }
        : type === 'Score'
          ? { ...base, type, levels: ['', ''] }
          : { ...base, type, trueWhen: '', falseWhen: '' }
    setQuestions([...questions, question])
    edited()
  }
  /**
   * Submit the current state/questions once. Local validation failures never
   * reach the API; successful responses from abandoned requests are ignored and
   * pending state is released only for the active controller.
   */
  async function evaluate(event?: React.FormEvent) {
    event?.preventDefault()
    if (active.current) return
    setNotice('')
    if (validateDecisionRequest(state, questions).length) {
      setCheckInputs(true)
      setFailure(null)
      return
    }
    setCheckInputs(false)
    setFailure(null)
    const controller = new AbortController()
    active.current = controller
    const snapshot = { state, questions }
    const startedAt = Date.now()
    setNow(startedAt)
    setPending({ questions, startedAt })
    try {
      const answers = await requestDecisions(snapshot.state, snapshot.questions, controller.signal)
      if (active.current === controller && !controller.signal.aborted) {
        setEvaluation({ ...snapshot, answers, elapsedMs: Date.now() - startedAt })
        // In the single-column layout the answers sit below the form.
        answersPanel.current?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' })
      }
    } catch (error) {
      if (active.current === controller && !controller.signal.aborted)
        setFailure(error instanceof DecisionError ? error : new DecisionError('Evaluation failed', 'Something went wrong. Retry.', { retryable: true }))
    } finally {
      if (active.current === controller) {
        active.current = null
        setPending(null)
      }
    }
  }
  /**
   * Abort this browser request and invalidate its controller immediately.
   * Runtime cleanup may outlast the UI cancellation; previous answers are kept.
   */
  function cancel() {
    active.current?.abort()
    active.current = null
    setPending(null)
    setNotice('Evaluation cancelled. The CPU worker may take a moment to finish.')
  }

  const elapsed = pending ? Math.max(0, Math.floor((now - pending.startedAt) / 1000)) : 0
  return (
    <div className="scroll-page">
      <div className="decisions-layout">
        <form className="stack" onSubmit={evaluate} noValidate>
          <fieldset disabled={!!pending} className="decisions-fieldset stack">
            <Panel className="panel-body stack">
              <div className="decisions-section-heading">
                <SectionHeading number="01">State</SectionHeading>
                <span className={`mono ${state.length > 8000 ? 'error' : 'faint'}`}>{state.length.toLocaleString()} / 8,000</span>
              </div>
              <textarea
                aria-label="State"
                aria-invalid={pageIssues.some((issue) => /state/i.test(issue.message)) || undefined}
                className="input"
                rows={5}
                value={state}
                placeholder="Describe the situation to judge, e.g. a support message, an agent transcript or a document excerpt."
                onChange={(e) => {
                  setState(e.target.value)
                  edited()
                }}
              />
            </Panel>
            <Panel className="panel-body stack">
              <div className="decisions-section-heading">
                <SectionHeading number="02">Questions</SectionHeading>
                <span className="mono faint">{questions.length} / {MAX_QUESTIONS}</span>
              </div>
              {!questions.length && (
                <p className="decisions-hint">
                  <strong>Choice</strong> picks one option, <strong>Score</strong> rates against a rubric, and{' '}
                  <strong>Noul</strong> gives the probability a statement is true.
                </p>
              )}
              {questions.map((question, index) => (
                <DecisionQuestionCard
                  key={index}
                  question={question}
                  index={index}
                  issues={issues.filter((issue) => issue.question === index).map((issue) => issue.message)}
                  onChange={(value) => {
                    setQuestions(questions.map((q, i) => (i === index ? value : q)))
                    edited()
                  }}
                  onRemove={() => {
                    setQuestions(questions.filter((_, i) => i !== index))
                    edited()
                  }}
                />
              ))}
              <div className="row decisions-add">
                {(['Choice', 'Score', 'Noul'] as const).map((type) => (
                  <button
                    type="button"
                    className="button"
                    key={type}
                    onClick={() => add(type)}
                  >
                    + {type}
                  </button>
                ))}
              </div>
            </Panel>
          </fieldset>
          <div className="decisions-actions">
            <button className="button button--primary" type="submit" disabled={!!pending}>
              {pending ? 'Evaluating…' : 'Evaluate'}
            </button>
            {pending && (
              <button type="button" className="button button--secondary" onClick={cancel}>
                Cancel evaluation
              </button>
            )}
          </div>
          <p className="muted decisions-footnote">
            Uses Laya on CPU, independently of the chat model. Inputs must fit the
            specialist’s token budget. Questions and answers are not saved.
          </p>
        </form>
        <Panel className="answers-panel">
          <header className="panel-header">
            <SectionHeading>Answers</SectionHeading>
            {evaluation && !pending && (
              <span className="mono faint push-right">
                {evaluation.answers.length} · {(evaluation.elapsedMs / 1000).toFixed(1)}s
              </span>
            )}
          </header>
          <div ref={answersPanel} className="panel-body answers-body" aria-live="polite" aria-busy={!!pending}>
            {shownFailure && (
              <div role="alert" className="decision-error">
                <div className="decision-error-text">
                  <strong>{shownFailure.title}</strong>
                  <p>{shownFailure.message}</p>
                  {pageIssues.length > 0 && (
                    <ul>{pageIssues.map((issue) => <li key={issue.message}>{issue.message}</li>)}</ul>
                  )}
                  {issues.length > pageIssues.length && (
                    <p className="decision-error-hint">Problems are marked on the affected questions.</p>
                  )}
                </div>
                {shownFailure.retryable && (
                  <button type="button" className="button" onClick={() => void evaluate()}>Retry</button>
                )}
              </div>
            )}
            {notice && <p role="status" className="decision-notice">{notice}</p>}
            {pending ? (
              <>
                <p role="status" className="answers-progress">
                  <span className="answers-spinner" aria-hidden="true" />
                  Evaluating {pending.questions.length} question{pending.questions.length === 1 ? '' : 's'} on CPU… <span className="mono faint">{elapsed}s</span>
                </p>
                {pending.questions.map((question, i) => (
                  <div key={i} className="answer-card answer-card--loading" aria-hidden="true">
                    <header className="answer-card-header">
                      <span className="answer-key mono">{question.key}</span>
                      <span className={`type-badge type-badge--${question.type}`}>{question.type}</span>
                    </header>
                    <span className="skeleton" />
                    <span className="skeleton skeleton--short" />
                  </div>
                ))}
              </>
            ) : evaluation ? (
              <>
                {stale && (
                  <div className="answers-stale" role="status">
                    <span>Inputs changed since these answers.</span>
                    <button type="button" className="button" onClick={() => void evaluate()}>Re-evaluate</button>
                  </div>
                )}
                {evaluation.answers.map((answer, i) => (
                  <DecisionAnswerCard key={answer.key} answer={answer} question={evaluation.questions[i]} stale={stale} />
                ))}
              </>
            ) : (
              !shownFailure && (
                <div className="answers-empty">
                  <strong>No answers yet</strong>
                  <p>Describe a state, add up to {MAX_QUESTIONS} questions, then Evaluate. Each answer appears here with its probabilities.</p>
                </div>
              )
            )}
          </div>
        </Panel>
      </div>
    </div>
  )
}
