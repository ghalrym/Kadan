import { useId } from 'react'
import type { Question } from '../api/decisions'
export type { Question } from '../api/decisions'

const typeHelp: Record<Question['type'], string> = {
  Choice: 'Picks exactly one option.',
  Score: 'Expected level on the rubric, from 0 (lowest) upward.',
  Noul: 'Probability that the instructions hold true.',
}

/**
 * Render a controlled Choice, Score or Noul question editor.
 * Pass complete replacement questions to onChange and removal to onRemove;
 * validation and evaluation belong to the parent and API, which pass any
 * problems for this card back through `issues`.
 */
export default function DecisionQuestionCard({
  question,
  index,
  issues = [],
  onChange,
  onRemove,
}: {
  question: Question
  index: number
  issues?: string[]
  onChange: (value: Question) => void
  onRemove: () => void
}) {
  const id = useId()
  const invalid = issues.length > 0
  return (
    <article className={`question-card${invalid ? ' question-card--invalid' : ''}`} aria-describedby={invalid ? `${id}-issues` : undefined}>
      <header className="question-header">
        <span className="mono faint">{String(index + 1).padStart(2, '0')}</span>
        <span className={`type-badge type-badge--${question.type}`}>{question.type}</span>
        <h3 className="sr-only">Question {index + 1} · {question.type}</h3>
        <span className="question-help faint">{typeHelp[question.type]}</span>
        <button type="button" className="button button--text push-right" onClick={onRemove} aria-label={`Remove question ${index + 1}`}>
          Remove
        </button>
      </header>
      {invalid && (
        <ul className="question-issues" id={`${id}-issues`}>
          {issues.map((issue) => <li key={issue}>{issue}</li>)}
        </ul>
      )}
      <div className="question-body">
        <div className="question-fields">
          <div className="field">
            <label htmlFor={`${id}-key`}>Key</label>
            <input
              id={`${id}-key`}
              className="input mono"
              placeholder="e.g. intent"
              value={question.key}
              onChange={(e) => onChange({ ...question, key: e.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor={`${id}-instructions`}>Instructions</label>
            <input
              id={`${id}-instructions`}
              className="input"
              placeholder={question.type === 'Noul' ? 'e.g. The customer asks to speak to a manager' : 'e.g. What does the customer want?'}
              value={question.instructions}
              onChange={(e) => onChange({ ...question, instructions: e.target.value })}
            />
          </div>
        </div>
        {question.type === 'Choice' && (
          <div className="field">
            <span className="field-label">Options</span>
            {question.options.map((option, i) => (
              <div className="option-row" key={i}>
                <input
                  className="input mono"
                  aria-label={`Option ${i + 1} key`}
                  placeholder="key"
                  value={option.key}
                  onChange={(e) =>
                    onChange({ ...question, options: question.options.map((o, j) => (j === i ? { ...o, key: e.target.value } : o)) })
                  }
                />
                <input
                  className="input"
                  aria-label={`Option ${i + 1} description`}
                  placeholder="Description (optional)"
                  value={option.description}
                  onChange={(e) =>
                    onChange({ ...question, options: question.options.map((o, j) => (j === i ? { ...o, description: e.target.value } : o)) })
                  }
                />
                <button
                  type="button"
                  className="button button--text question-remove-item"
                  aria-label={`Remove option ${i + 1}`}
                  disabled={question.options.length === 1}
                  onClick={() => onChange({ ...question, options: question.options.filter((_, j) => j !== i) })}
                >
                  ×
                </button>
              </div>
            ))}
            <button
              type="button"
              className="button button--text question-add-item"
              onClick={() => onChange({ ...question, options: [...question.options, { key: '', description: '' }] })}
            >
              + Add option
            </button>
          </div>
        )}
        {question.type === 'Score' && (
          <div className="field">
            <span className="field-label">Rubric levels, lowest to highest</span>
            {question.levels.map((level, i) => (
              <div className="rubric-row" key={i}>
                <span className="level-number mono">{i}</span>
                <input
                  className="input"
                  aria-label={`Level ${i}`}
                  placeholder={i === 0 ? 'e.g. Not urgent' : 'Describe this level'}
                  value={level}
                  onChange={(e) => onChange({ ...question, levels: question.levels.map((v, j) => (j === i ? e.target.value : v)) })}
                />
                <button
                  type="button"
                  className="button button--text question-remove-item"
                  aria-label={`Remove level ${i}`}
                  disabled={question.levels.length === 1}
                  onClick={() => onChange({ ...question, levels: question.levels.filter((_, j) => j !== i) })}
                >
                  ×
                </button>
              </div>
            ))}
            <button
              type="button"
              className="button button--text question-add-item"
              onClick={() => onChange({ ...question, levels: [...question.levels, ''] })}
            >
              + Add level
            </button>
          </div>
        )}
        {question.type === 'Noul' && (
          <div className="question-fields question-fields--even">
            <div className="field">
              <label htmlFor={`${id}-true`}>True when <span className="faint">(optional)</span></label>
              <input
                id={`${id}-true`}
                className="input"
                value={question.trueWhen ?? ''}
                onChange={(e) => onChange({ ...question, trueWhen: e.target.value })}
              />
            </div>
            <div className="field">
              <label htmlFor={`${id}-false`}>False when <span className="faint">(optional)</span></label>
              <input
                id={`${id}-false`}
                className="input"
                value={question.falseWhen ?? ''}
                onChange={(e) => onChange({ ...question, falseWhen: e.target.value })}
              />
            </div>
          </div>
        )}
      </div>
    </article>
  )
}
