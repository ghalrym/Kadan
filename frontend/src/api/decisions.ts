import { evaluateDecisions } from './generated/sdk.gen'
import type {
  DecisionRequest,
  DecisionResponse,
  ValidationError,
} from './generated/types.gen'

export type Question = DecisionRequest['questions'][number]
export type Answer = DecisionResponse['answers'][number]

/** One problem to show the user; `question` is a zero-based index when it belongs to a card. */
export type DecisionIssue = { message: string; question?: number }

/**
 * A failed evaluation, already phrased for display. `issues` are field-level
 * problems the user can fix; `retryable` marks failures where resending the
 * same input may succeed (runtime, network or malformed output).
 */
export class DecisionError extends Error {
  title: string
  issues: DecisionIssue[]
  retryable: boolean
  constructor(
    title: string,
    message: string,
    { issues = [], retryable = false }: { issues?: DecisionIssue[]; retryable?: boolean } = {},
  ) {
    super(message)
    this.title = title
    this.issues = issues
    this.retryable = retryable
  }
}

const MAX_QUESTIONS = 8
const MAX_STATE = 8000

/**
 * Mirror the API's structural request rules so obvious mistakes are shown on
 * the right card without a CPU round trip. The server stays authoritative and
 * additionally enforces total size and tokenizer budgets.
 */
export function validateDecisionRequest(state: string, questions: Question[]): DecisionIssue[] {
  const issues: DecisionIssue[] = []
  if (!state.trim()) issues.push({ message: 'Describe the state to evaluate.' })
  else if (state.length > MAX_STATE) issues.push({ message: `State must be at most ${MAX_STATE.toLocaleString()} characters.` })
  if (!questions.length) issues.push({ message: 'Add at least one question.' })
  if (questions.length > MAX_QUESTIONS) issues.push({ message: `Use at most ${MAX_QUESTIONS} questions.` })
  questions.forEach((question, index) => {
    const add = (message: string) => issues.push({ message, question: index })
    if (!question.key.trim()) add('Key is required.')
    else if (questions.findIndex((other) => other.key === question.key) < index)
      add(`Key “${question.key}” is already used by another question.`)
    if (!question.instructions.trim()) add('Instructions are required.')
    if (question.type === 'Choice') {
      if (!question.options.length) add('Add at least one option.')
      if (question.options.some((option) => !option.key.trim())) add('Every option needs a key.')
      const keys = question.options.map((option) => option.key).filter((key) => key.trim())
      const duplicate = keys.find((key, i) => keys.indexOf(key) < i)
      if (duplicate !== undefined) add(`Option key “${duplicate}” is used more than once.`)
    }
    if (question.type === 'Score') {
      if (!question.levels.length) add('Add at least one rubric level.')
      if (question.levels.some((level) => !level.trim())) add('Every rubric level needs a description.')
    }
  })
  return issues
}

/**
 * Turn FastAPI validation entries into card-scoped issues. Field locations
 * are made readable (`questions.1.Choice.options.0.key` → “Option 1 key”) and
 * request-wide messages naming a question key are attached to that card.
 */
function validationIssues(detail: ValidationError[], questions: Question[]): DecisionIssue[] {
  return detail.map((entry) => {
    const loc = entry.loc.filter((part) => part !== 'body')
    const message = entry.msg.replace(/^Value error, /, '')
    if (loc[0] === 'questions' && typeof loc[1] === 'number') {
      const field = loc
        .slice(2)
        .filter((part) => !['Choice', 'Score', 'Noul'].includes(String(part)))
        .map((part) => (typeof part === 'number' ? String(part + 1) : String(part)))
        .join(' ')
        .replace(/^options (\d+)/, 'Option $1')
        .replace(/^levels (\d+)/, 'Level $1')
      return { question: loc[1], message: field ? `${field[0].toUpperCase()}${field.slice(1)}: ${message}` : message }
    }
    const named = /question(?: key)? "([^"]+)"/i.exec(message)?.[1]
    const question = named === undefined ? -1 : questions.findIndex((item) => item.key === named)
    const field = loc.length ? `${String(loc[0])[0].toUpperCase()}${String(loc[0]).slice(1)}: ` : ''
    return question >= 0 ? { question, message } : { message: `${field}${message}` }
  })
}

/**
 * Evaluate one request snapshot and return answers in question order.
 * The signal cancels the browser request (the CPU worker may still finish).
 * Every failure, including network and malformed output, rejects with a
 * DecisionError; an abort rejects with the platform AbortError.
 */
export async function requestDecisions(
  state: string,
  questions: Question[],
  signal: AbortSignal,
): Promise<Answer[]> {
  const issues = validateDecisionRequest(state, questions)
  if (issues.length)
    throw new DecisionError('Check your inputs', 'Some fields need attention before evaluating.', { issues })
  let result: Awaited<ReturnType<typeof evaluateDecisions>>
  try {
    result = await evaluateDecisions({ body: { state, questions }, signal })
  } catch (failure) {
    if (signal.aborted) throw failure
    throw new DecisionError('Cannot reach Kadan', 'The decisions API did not respond. Check that the backend is running, then retry.', { retryable: true })
  }
  const status = result.response?.status
  if (!result.response?.ok) {
    if (signal.aborted) throw new DOMException('Evaluation cancelled', 'AbortError')
    const detail = (result.error as { detail?: unknown } | undefined)?.detail
    if (status === 422 && Array.isArray(detail))
      throw new DecisionError('Check your inputs', 'The API rejected part of this request.', {
        issues: validationIssues(detail as ValidationError[], questions),
      })
    const message = typeof detail === 'string' ? detail : undefined
    if (status === 422)
      throw new DecisionError('Input too long', message ?? 'The state or a question exceeds the decision model’s token budget. Shorten it, then retry.')
    if (status === 503)
      throw new DecisionError('Decision model unavailable', message ?? 'The CPU decision model could not be loaded. Retry in a moment.', { retryable: true })
    if (status === 409 || status === 429)
      throw new DecisionError('Decision model busy', message ?? 'Another evaluation is running. Wait for it to finish, then retry.', { retryable: true })
    if (status === 502)
      throw new DecisionError('Unexpected model output', message ?? 'The decision model returned an answer Kadan could not read. Retry.', { retryable: true })
    if (!status)
      throw new DecisionError('Cannot reach Kadan', 'The decisions API did not respond. Check that the backend is running, then retry.', { retryable: true })
    throw new DecisionError('Evaluation failed', message ?? `The decisions API returned HTTP ${status}. Retry.`, { retryable: true })
  }
  const answers = result.data?.answers
  if (!Array.isArray(answers) || answers.length !== questions.length || answers.some((answer, i) => answer?.key !== questions[i].key || answer.type !== questions[i].type))
    throw new DecisionError('Unexpected model output', 'The API returned answers that do not match the questions. Retry.', { retryable: true })
  return answers
}
