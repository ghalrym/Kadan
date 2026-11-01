import { useEffect, useRef, useState } from 'react'
import {
  accountNames,
  cancelLogin,
  chooseLanguage,
  disconnect,
  fetchSubscriptions,
  loginStatus,
  modelNames,
  providerNames,
  startLogin,
  submitLoginCode,
  type Provider,
  type SubscriptionLogin,
  type SubscriptionProvider,
  type SubscriptionsResponse,
} from '../api/subscriptions'

const providers: Provider[] = ['local', 'claude', 'codex']
const pending = (login: SubscriptionLogin | null) => login?.status === 'waiting' || login?.status === 'verifying'

/** The steps of one web sign-in: open the provider's page, then finish there or bring back its code. */
function LoginSteps({ login, submitting, onCode, onCancel }: {
  login: SubscriptionLogin
  submitting: boolean
  onCode: (code: string) => void
  onCancel: () => void
}) {
  const [code, setCode] = useState('')
  const account = accountNames[login.provider]
  if (login.status === 'starting') return <p role="status" className="muted subscription-login">Opening sign-in…</p>
  if (login.status === 'failed') return null
  return (
    <form
      className="subscription-login"
      onSubmit={(event) => {
        event.preventDefault()
        onCode(code)
      }}
    >
      <ol className="subscription-steps">
        <li>
          <a className="button button--primary" href={login.url ?? undefined} target="_blank" rel="noreferrer">
            Open {account} sign-in
          </a>
        </li>
        {login.user_code && (
          <li>
            When asked, enter this code: <strong className="mono subscription-code">{login.user_code}</strong>
          </li>
        )}
        {login.needs_code && (
          <li>
            <label htmlFor={`${login.provider}-code`}>Approve access, then copy the code it shows you and paste it here.</label>
            <input
              id={`${login.provider}-code`}
              className="input mono"
              autoComplete="off"
              value={code}
              disabled={submitting || login.status === 'verifying'}
              onChange={(event) => setCode(event.target.value)}
            />
          </li>
        )}
      </ol>
      <p role="status" className="muted">
        {login.status === 'verifying' || !login.needs_code ? 'Waiting for you to finish signing in…' : ''}
      </p>
      <div className="models-dialog-actions">
        <button type="button" className="button" onClick={onCancel}>Cancel</button>
        {login.needs_code && (
          <button type="submit" className="button button--primary"
            disabled={submitting || login.status === 'verifying' || !code.trim()}>
            {login.status === 'verifying' ? 'Finishing…' : 'Finish sign-in'}
          </button>
        )}
      </div>
    </form>
  )
}

/**
 * Sign in to the user's own Claude or Codex subscription and choose which provider
 * serves language. Local keeps Hikari's downloaded model.
 */
export default function SubscriptionsPanel() {
  const [state, setState] = useState<SubscriptionsResponse | null>(null)
  const [login, setLogin] = useState<SubscriptionLogin | null>(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const action = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    fetchSubscriptions(controller.signal)
      .then((next) => !controller.signal.aborted && setState(next))
      .catch((failure) => !controller.signal.aborted
        && setError(failure instanceof Error ? failure.message : 'Could not load subscriptions'))
    return () => {
      controller.abort()
      action.current?.abort()
    }
  }, [])

  // Follow a sign-in until the provider confirms it; the subscription is then connected.
  const polling = pending(login) ? login?.provider : undefined
  useEffect(() => {
    if (!polling) return
    const controller = new AbortController()
    const timer = setInterval(() => {
      void (async () => {
        try {
          const next = await loginStatus(polling, controller.signal)
          if (controller.signal.aborted) return
          if (next.status === 'complete') {
            setLogin(null)
            setState(await fetchSubscriptions(controller.signal))
          } else {
            setLogin(next)
            if (next.status === 'failed') setError(next.error ?? 'Sign-in did not finish. Please try again.')
          }
        } catch (failure) {
          if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Could not check sign-in')
        }
      })()
    }, 2000)
    return () => {
      controller.abort()
      clearInterval(timer)
    }
  }, [polling])

  async function run<T>(work: (signal: AbortSignal) => Promise<T>, done: (result: T) => void) {
    if (action.current) return
    const controller = new AbortController()
    action.current = controller
    setSubmitting(true)
    setError('')
    try {
      const result = await work(AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]))
      if (!controller.signal.aborted) done(result)
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Something went wrong. Please try again.')
    } finally {
      action.current = null
      if (!controller.signal.aborted) setSubmitting(false)
    }
  }

  function signIn(provider: SubscriptionProvider) {
    setLogin({ provider, status: 'starting', needs_code: provider === 'claude' })
    void run((signal) => startLogin(provider, signal), (next) => {
      setLogin(next)
      if (next.status === 'failed') setError(next.error ?? 'Sign-in could not start. Please try again.')
    })
  }

  function cancel(provider: SubscriptionProvider) {
    setLogin(null)
    void run((signal) => cancelLogin(provider, signal), () => {})
  }

  const connected = (provider: Provider) =>
    provider === 'local' || !!state?.subscriptions.find((row) => row.provider === provider)?.connected
  const language = (state?.capabilities.language ?? 'local') as Provider
  const languageModel = (provider: Provider) => {
    if (provider === 'local') return 'LFM3 1.7B on this computer'
    const model = state?.subscriptions.find((row) => row.provider === provider)?.model ?? ''
    return modelNames[model] ?? model
  }

  return (
    <section className="panel subscriptions-panel" aria-labelledby="subscriptions-title">
      <header className="models-summary">
        <div className="models-summary-text">
          <h2 id="subscriptions-title">Subscriptions</h2>
          <p className="muted">Use your own Claude or ChatGPT subscription instead of a model on this computer.</p>
        </div>
      </header>
      {error && <p role="alert" className="error subscriptions-error">{error}</p>}
      {!state && !error && <p role="status" className="muted subscriptions-error">Loading subscriptions…</p>}
      {state && (
        <>
          <div className="subscription-capability tone-llm">
            <span className="model-row-label">
              <span className="model-row-capability">Language</span>
              <span className="model-row-name">{languageModel(language)}</span>
            </span>
            <div className="segments" role="group" aria-label="Language provider">
              {providers.map((provider) => (
                <button
                  type="button"
                  key={provider}
                  aria-pressed={provider === language}
                  className={provider === language ? 'selected' : ''}
                  disabled={submitting || !connected(provider)}
                  title={connected(provider) ? undefined : `Sign in to ${providerNames[provider]} first`}
                  onClick={() => provider !== language
                    && void run((signal) => chooseLanguage(provider, signal), setState)}
                >
                  {providerNames[provider]}
                </button>
              ))}
            </div>
          </div>
          <ul className="subscription-list">
            {state.subscriptions.map(({ provider, connected: isConnected }) => (
              <li key={provider} className="subscription-row">
                <span className="subscription-name">{providerNames[provider]}</span>
                <span className={`mono subscription-status${isConnected ? ' subscription-status--connected' : ''}`}>
                  <span className="status-dot" aria-hidden="true" />
                  {isConnected ? 'Signed in' : 'Not signed in'}
                </span>
                <span className="subscription-actions">
                  {isConnected ? (
                    <button type="button" className="button" disabled={submitting}
                      onClick={() => void run((signal) => disconnect(provider, signal), setState)}>
                      Sign out
                    </button>
                  ) : (
                    <button type="button" className="button" disabled={submitting || pending(login)}
                      onClick={() => signIn(provider)}>
                      {login?.provider === provider && login.status === 'failed' ? 'Try again' : 'Sign in'}
                    </button>
                  )}
                </span>
                {login?.provider === provider && (
                  <LoginSteps
                    login={login}
                    submitting={submitting}
                    onCode={(code) => void run((signal) => submitLoginCode(provider, code, signal), setLogin)}
                    onCancel={() => cancel(provider)}
                  />
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
