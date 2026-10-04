import { ActionButton, Field, SegmentedControl } from './Controls'
import type { DecisionQuestion } from '../data/playground'

export default function DecisionQuestionCard({
  question,
  index,
}: {
  question: DecisionQuestion
  index: number
}) {
  return (
    <article className="question-card">
      <header className="question-header">
        <h3>Question {index + 1}</h3>
        <SegmentedControl
          label="Question type"
          options={['Choice', 'Score', 'Noul']}
          selected={question.type}
        />
        <ActionButton variant="text" className="push-right">
          Remove
        </ActionButton>
      </header>
      <div className="question-body">
        <div className="question-fields">
          <Field label="Key" value={question.key} mono />
          <Field label="Instructions" value={question.instructions} />
        </div>
        {question.type === 'Choice' && (
          <div className="stack compact">
            <span className="field-label">Options</span>
            <div className="option-row eyebrow">
              <span>Key</span>
              <span>Description</span>
              <span />
            </div>
            {question.options.map((option) => (
              <div className="option-row" key={option.key}>
                <input
                  className="input mono"
                  value={option.key}
                  readOnly
                  aria-label={`${option.key} key`}
                />
                <input
                  className="input"
                  value={option.description}
                  readOnly
                  aria-label={`${option.key} description`}
                />
                <ActionButton variant="text" label={`Remove ${option.key}`}>
                  ✕
                </ActionButton>
              </div>
            ))}
            <div>
              <ActionButton>+ Option</ActionButton>
            </div>
          </div>
        )}
        {question.type === 'Score' && (
          <div className="stack compact">
            <span className="field-label">Rubric levels</span>
            <div className="rubric-row eyebrow">
              <span>Level</span>
              <span>Description · lowest to highest</span>
              <span />
            </div>
            {question.levels.map((level, levelIndex) => (
              <div className="rubric-row" key={level}>
                <span className="level-number mono">{levelIndex}</span>
                <input
                  className="input"
                  value={level}
                  readOnly
                  aria-label={`Level ${levelIndex} description`}
                />
                <ActionButton
                  variant="text"
                  label={`Remove level ${levelIndex}`}
                >
                  ✕
                </ActionButton>
              </div>
            ))}
            <div>
              <ActionButton>+ Level</ActionButton>
            </div>
          </div>
        )}
        {question.type === 'Noul' && (
          <div className="two-columns">
            <Field
              label="True when · optional"
              placeholder="Describe the true case"
              value={question.trueWhen}
            />
            <Field
              label="False when · optional"
              placeholder="Describe the false case"
              value={question.falseWhen}
            />
          </div>
        )}
      </div>
    </article>
  )
}
