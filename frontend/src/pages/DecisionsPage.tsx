import { useEffect, useRef, useState } from 'react'
import { Panel, SectionHeading } from '../components/Controls'
import DecisionQuestionCard, {
  type Question,
} from '../components/DecisionQuestionCard'
import type { DecisionResponse } from '../api/generated'

/**
 * Edit page-local decision state and display answers from the CPU Laya API.
 * Only the active request may update results; leaving the page aborts that request.
 */
export default function DecisionsPage() {
  const [state, setState] = useState('')
  const [questions, setQuestions] = useState<Question[]>([])
  const [answers, setAnswers] = useState<DecisionResponse['answers']>([])
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const active = useRef<AbortController | null>(null)
  useEffect(
    () => () => {
      active.current?.abort()
      active.current = null
    },
    [],
  )
  /**
   * Append an empty question of the selected type and discard stale answers.
   */
  function add(type: Question['type']) {
    const base = { key: '', instructions: '' }
    const question: Question =
      type === 'Choice'
        ? { ...base, type, options: [{ key: '', description: '' }] }
        : type === 'Score'
          ? { ...base, type, levels: [''] }
          : { ...base, type, trueWhen: '', falseWhen: '' }
    setQuestions([...questions, question])
    setAnswers([])
  }
  /**
   * Submit the current state/questions once and display API answers or errors.
   * Clear previous results before evaluating; ignore successful responses from
   * abandoned requests and release pending state only for the active controller.
   */
  async function evaluate(event: React.FormEvent) {
    event.preventDefault()
    if (active.current) return
    const controller = new AbortController()
    active.current = controller
    setPending(true)
    setError('')
    setNotice('')
    setAnswers([])
    try {
      const response = await fetch('/v1/decisions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state, questions }),
        signal: controller.signal,
      })
      const data = await response.json()
      if (!response.ok)
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Check question keys, options and rubric levels, then retry.',
        )
      if (active.current === controller && !controller.signal.aborted)
        setAnswers(data.answers)
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(
          failure instanceof Error
            ? failure.message
            : 'Evaluation failed. Retry.',
        )
    } finally {
      if (active.current === controller) {
        active.current = null
        setPending(false)
      }
    }
  }
  /**
   * Abort this browser request and invalidate its controller immediately.
   * Runtime cleanup may outlast the UI cancellation; no replacement answer is created.
   */
  function cancel() {
    active.current?.abort()
    active.current = null
    setPending(false)
    setNotice(
      'Evaluation cancelled. The CPU worker may take a moment to finish.',
    )
  }
  return (
    <div className="scroll-page">
      <div className="decisions-layout">
        <form className="stack" onSubmit={evaluate}>
          <fieldset
            disabled={pending}
            style={{ border: 0, padding: 0, margin: 0 }}
            className="stack"
          >
            <Panel className="panel-body stack">
              <SectionHeading number="01">State</SectionHeading>
              <textarea
                aria-label="State"
                className="input"
                rows={5}
                maxLength={8000}
                required
                value={state}
                onChange={(e) => {
                  setState(e.target.value)
                  setAnswers([])
                }}
              />
            </Panel>
            <Panel className="panel-body stack">
              <SectionHeading number="02">
                Questions · {questions.length}/8
              </SectionHeading>
              {questions.map((question, index) => (
                <DecisionQuestionCard
                  key={index}
                  question={question}
                  index={index}
                  onChange={(value) => {
                    setQuestions(
                      questions.map((q, i) => (i === index ? value : q)),
                    )
                    setAnswers([])
                  }}
                  onRemove={() => {
                    setQuestions(questions.filter((_, i) => i !== index))
                    setAnswers([])
                  }}
                />
              ))}
              <div className="row">
                {(['Choice', 'Score', 'Noul'] as const).map((type) => (
                  <button
                    type="button"
                    className="button"
                    key={type}
                    disabled={questions.length >= 8}
                    onClick={() => add(type)}
                  >
                    + {type}
                  </button>
                ))}
              </div>
            </Panel>
            <button
              className="button button--primary"
              type="submit"
              disabled={!questions.length}
            >
              Evaluate
            </button>
          </fieldset>
          {pending && (
            <button type="button" className="button" onClick={cancel}>
              Cancel evaluation
            </button>
          )}
          {error && <p role="alert">{error}</p>}
          {notice && <p role="status">{notice}</p>}
          <p className="muted">
            Uses Laya on CPU, independently of the chat model. Score is an expected
            rubric index; Noul is the probability of true. Inputs must fit the
            specialist’s token budget. Questions and answers are not saved.
          </p>
        </form>
        <Panel className="answers-panel">
          <header className="panel-header">
            <SectionHeading>Answers</SectionHeading>
          </header>
          <div className="panel-body" aria-live="polite">
            {pending
              ? 'Evaluating…'
              : answers.length
                ? answers.map((answer) => (
                    <p key={answer.key}>
                      <strong>{answer.key}</strong>: {answer.type === 'Noul' ? `${(Number(answer.value) * 100).toFixed(1)}% true` : String(answer.value)}
                    </p>
                  ))
                : 'No answers yet'}
          </div>
        </Panel>
      </div>
    </div>
  )
}
