import type { DecisionRequest } from '../api/generated'
export type Question = DecisionRequest['questions'][number]

/**
 * Render a controlled Choice, Score or Noul question editor.
 * Pass complete replacement questions to onChange and removal to onRemove;
 * validation and evaluation belong to the parent and API, not this card.
 */
export default function DecisionQuestionCard({
  question,
  index,
  onChange,
  onRemove,
}: {
  question: Question
  index: number
  onChange: (value: Question) => void
  onRemove: () => void
}) {
  return (
    <article className="question-card">
      <header className="question-header">
        <h3>
          Question {index + 1} · {question.type}
        </h3>
        <button
          type="button"
          className="button button--text push-right"
          onClick={onRemove}
        >
          Remove
        </button>
      </header>
      <div className="question-body stack">
        <label>
          Key
          <input
            className="input"
            value={question.key}
            onChange={(e) => onChange({ ...question, key: e.target.value })}
          />
        </label>
        <label>
          Instructions
          <input
            className="input"
            value={question.instructions}
            onChange={(e) =>
              onChange({ ...question, instructions: e.target.value })
            }
          />
        </label>
        {question.type === 'Choice' && (
          <>
            <span>Options</span>
            {question.options.map((option, i) => (
              <div className="option-row" key={i}>
                <input
                  className="input"
                  aria-label={`Option ${i + 1} key`}
                  value={option.key}
                  onChange={(e) =>
                    onChange({
                      ...question,
                      options: question.options.map((o, j) =>
                        j === i ? { ...o, key: e.target.value } : o,
                      ),
                    })
                  }
                />
                <input
                  className="input"
                  aria-label={`Option ${i + 1} description`}
                  value={option.description}
                  onChange={(e) =>
                    onChange({
                      ...question,
                      options: question.options.map((o, j) =>
                        j === i ? { ...o, description: e.target.value } : o,
                      ),
                    })
                  }
                />
                <button
                  type="button"
                  className="button"
                  aria-label={`Remove option ${i + 1}`}
                  onClick={() =>
                    onChange({
                      ...question,
                      options: question.options.filter((_, j) => j !== i),
                    })
                  }
                >
                  ×
                </button>
              </div>
            ))}
            <button
              type="button"
              className="button"
              onClick={() =>
                onChange({
                  ...question,
                  options: [...question.options, { key: '', description: '' }],
                })
              }
            >
              + Option
            </button>
          </>
        )}
        {question.type === 'Score' && (
          <>
            <span>
              Rubric levels, lowest to highest. Answers use the zero-based level
              index.
            </span>
            {question.levels.map((level, i) => (
              <div className="rubric-row" key={i}>
                <span>{i}</span>
                <input
                  className="input"
                  aria-label={`Level ${i}`}
                  value={level}
                  onChange={(e) =>
                    onChange({
                      ...question,
                      levels: question.levels.map((v, j) =>
                        j === i ? e.target.value : v,
                      ),
                    })
                  }
                />
                <button
                  type="button"
                  className="button"
                  aria-label={`Remove level ${i}`}
                  onClick={() =>
                    onChange({
                      ...question,
                      levels: question.levels.filter((_, j) => j !== i),
                    })
                  }
                >
                  ×
                </button>
              </div>
            ))}
            <button
              type="button"
              className="button"
              onClick={() =>
                onChange({ ...question, levels: [...question.levels, ''] })
              }
            >
              + Level
            </button>
          </>
        )}
        {question.type === 'Noul' && (
          <>
            <label>
              True when (optional)
              <input
                className="input"
                value={question.trueWhen ?? ''}
                onChange={(e) =>
                  onChange({ ...question, trueWhen: e.target.value })
                }
              />
            </label>
            <label>
              False when (optional)
              <input
                className="input"
                value={question.falseWhen ?? ''}
                onChange={(e) =>
                  onChange({ ...question, falseWhen: e.target.value })
                }
              />
            </label>
          </>
        )}
      </div>
    </article>
  )
}
