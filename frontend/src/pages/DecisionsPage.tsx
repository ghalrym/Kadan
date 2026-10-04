import { ActionButton, Panel, SectionHeading } from '../components/Controls'
import DecisionQuestionCard from '../components/DecisionQuestionCard'
import { decisionState, questions } from '../data/playground'

export default function DecisionsPage() {
  return (
    <div className="scroll-page">
      <div className="decisions-layout">
        <div className="stack">
          <Panel className="panel-body stack">
            <SectionHeading number="01">State</SectionHeading>
            <textarea
              aria-label="State"
              className="input"
              rows={5}
              value={decisionState}
              readOnly
            />
          </Panel>
          <Panel className="panel-body stack">
            <SectionHeading number="02">
              Questions <span className="mono faint">{questions.length}</span>
            </SectionHeading>
            {questions.map((question, index) => (
              <DecisionQuestionCard
                key={question.key}
                question={question}
                index={index}
              />
            ))}
            <div className="row">
              {['Choice', 'Score', 'Noul'].map((type) => (
                <ActionButton key={type}>+ {type}</ActionButton>
              ))}
            </div>
          </Panel>
          <div>
            <ActionButton variant="primary">Evaluate</ActionButton>
          </div>
        </div>
        <Panel className="answers-panel">
          <header className="panel-header">
            <SectionHeading>Answers</SectionHeading>
          </header>
          <div className="empty-state">No answers yet</div>
        </Panel>
      </div>
    </div>
  )
}
