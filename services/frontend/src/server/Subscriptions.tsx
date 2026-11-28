import { useEffect, useRef, useState } from "react";
import {
  accountNames,
  cancelLogin,
  chooseImages,
  chooseProvider,
  disconnect,
  fetchSubscriptions,
  loginStatus,
  modelNames,
  startLogin,
  submitLoginCode,
  type ImageProvider,
  type Provider,
  type SubscriptionLogin,
  type SubscriptionProvider,
  type SubscriptionsResponse,
} from "../hikari/api/subscriptions";
import { StatusPill } from "./common";

const waiting = (login: SubscriptionLogin | null) => login?.status === "waiting" || login?.status === "verifying";
const providerLabel = (provider: Provider) => (provider === "local" ? "This computer" : accountNames[provider]);

/** Open the provider's page, then finish there or paste back the code it shows. */
function SignInSteps({ login, busy, onCode, onCancel }: {
  login: SubscriptionLogin;
  busy: boolean;
  onCode: (code: string) => void;
  onCancel: () => void;
}) {
  const [code, setCode] = useState("");
  const account = accountNames[login.provider];
  if (login.status === "starting") return <p className="field-hint" role="status">Opening the {account} sign-in…</p>;
  if (login.status === "failed") return null;
  return (
    <form className="srv-signin" onSubmit={(event) => { event.preventDefault(); onCode(code); }}>
      <ol>
        <li>
          <a className="primary srv-link-button" href={login.url ?? undefined} target="_blank" rel="noreferrer">
            Open {account} sign-in
          </a>
        </li>
        {login.user_code && <li>When it asks, enter this code: <strong className="srv-code">{login.user_code}</strong></li>}
        {login.needs_code && (
          <li>
            <label>
              Approve access, then paste the code it shows you here.
              <input autoComplete="off" value={code} disabled={busy || login.status === "verifying"}
                onChange={(event) => setCode(event.target.value)} />
            </label>
          </li>
        )}
      </ol>
      <p className="field-hint" role="status">
        {login.status === "verifying" || !login.needs_code ? "Waiting for you to finish signing in…" : ""}
      </p>
      <div className="form-actions">
        <button type="button" onClick={onCancel}>Cancel</button>
        {login.needs_code && (
          <button type="submit" className="primary" disabled={busy || login.status === "verifying" || !code.trim()}>
            {login.status === "verifying" ? "Finishing…" : "Finish signing in"}
          </button>
        )}
      </div>
    </form>
  );
}

function Segments<T extends string>({ label, value, options, enabled, disabled, onChange }: {
  label: string;
  value: T;
  options: T[];
  enabled: (option: T) => boolean;
  disabled: boolean;
  onChange: (option: T) => void;
}) {
  return (
    <div className="srv-segments" role="group" aria-label={label}>
      {options.map((option) => (
        <button key={option} type="button" aria-pressed={option === value} className={option === value ? "selected" : ""}
          disabled={disabled || !enabled(option)}
          title={enabled(option) ? undefined : `Sign in to ${providerLabel(option as Provider)} first`}
          onClick={() => option !== value && onChange(option)}>
          {providerLabel(option as Provider)}
        </button>
      ))}
    </div>
  );
}

/** Use the person's own Claude or ChatGPT subscription instead of a model on this computer. */
export default function Subscriptions() {
  const [state, setState] = useState<SubscriptionsResponse | null>(null);
  const [login, setLogin] = useState<SubscriptionLogin | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const action = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchSubscriptions(controller.signal)
      .then((next) => !controller.signal.aborted && setState(next))
      .catch((failure) => !controller.signal.aborted && setError((failure as Error).message));
    return () => {
      controller.abort();
      action.current?.abort();
    };
  }, []);

  const following = waiting(login) ? login?.provider : undefined;
  useEffect(() => {
    if (!following) return;
    const controller = new AbortController();
    const timer = setInterval(() => {
      void (async () => {
        try {
          const next = await loginStatus(following, controller.signal);
          if (controller.signal.aborted) return;
          if (next.status === "complete") {
            const subscriptions = await fetchSubscriptions(controller.signal);
            if (controller.signal.aborted) return;
            setState(subscriptions);
            setLogin(null);
          } else {
            setLogin(next);
            if (next.status === "failed") setError(next.error ?? "Signing in didn’t finish. Try again.");
          }
        } catch (failure) {
          if (!controller.signal.aborted) setError((failure as Error).message);
        }
      })();
    }, 2000);
    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, [following]);

  async function run<T>(work: (signal: AbortSignal) => Promise<T>, done: (result: T) => void) {
    if (action.current) return;
    const controller = new AbortController();
    action.current = controller;
    setBusy(true);
    setError("");
    try {
      const result = await work(AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]));
      if (!controller.signal.aborted) done(result);
    } catch (failure) {
      if (!controller.signal.aborted) setError((failure as Error).message || "Something went wrong. Try again.");
    } finally {
      action.current = null;
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  function signIn(provider: SubscriptionProvider) {
    setLogin({ provider, status: "starting", needs_code: provider === "claude" });
    void run((signal) => startLogin(provider, signal), (next) => {
      setLogin(next);
      if (next.status === "failed") setError(next.error ?? "Signing in couldn’t start. Try again.");
    });
  }

  const connected = (provider: Provider) =>
    provider === "local" || !!state?.subscriptions.find((row) => row.provider === provider)?.connected;
  const subscriptionModel = (provider: SubscriptionProvider) => {
    const model = state?.subscriptions.find((row) => row.provider === provider)?.model ?? "";
    return modelNames[model] ?? model;
  };
  const language = (state?.capabilities.language ?? "local") as Provider;
  const image = (state?.capabilities.image ?? "local") as ImageProvider;

  return (
    <section className="srv-card" aria-labelledby="srv-subscriptions">
      <header className="srv-card-head">
        <h2 id="srv-subscriptions">Use a subscription instead</h2>
        <p>Already pay for Claude or ChatGPT? Kadan can use it in place of the models on this computer.</p>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      {!state && !error && <p className="field-hint" role="status">Checking your subscriptions…</p>}
      {state && (
        <>
          <div className="srv-sub-row">
            <div>
              <strong>Chat & thinking</strong>
              <span>{language === "local" ? "Using the model on this computer" : `Using ${subscriptionModel(language)}`}</span>
            </div>
            <Segments<Provider> label="Chat & thinking uses" value={language} options={["local", "claude", "codex"]}
              enabled={connected} disabled={busy}
              onChange={(provider) => void run((signal) => chooseProvider("language", provider, signal), setState)} />
          </div>
          <div className="srv-sub-row">
            <div>
              <strong>Images</strong>
              <span>
                {image === "local"
                  ? "Using the model on this computer. Claude can’t create images."
                  : "ChatGPT creates new pictures. Editing pictures and pets still uses this computer."}
              </span>
            </div>
            <Segments<ImageProvider> label="Images use" value={image} options={["local", "codex"]}
              enabled={connected} disabled={busy}
              onChange={(provider) => void run((signal) => chooseImages(provider, signal), setState)} />
          </div>
          <ul className="srv-accounts">
            {state.subscriptions.map(({ provider, connected: isConnected }) => (
              <li key={provider}>
                <div className="srv-account-row">
                  <strong>{accountNames[provider]}</strong>
                  <StatusPill tone={isConnected ? "ok" : "muted"}>{isConnected ? "Signed in" : "Not signed in"}</StatusPill>
                  {isConnected ? (
                    <button className="button" disabled={busy}
                      onClick={() => void run((signal) => disconnect(provider, signal), setState)}>Sign out</button>
                  ) : (
                    <button className="button" disabled={busy || waiting(login)} onClick={() => signIn(provider)}>
                      {login?.provider === provider && login.status === "failed" ? "Try again" : "Sign in"}
                    </button>
                  )}
                </div>
                {login?.provider === provider && (
                  <SignInSteps login={login} busy={busy}
                    onCode={(code) => void run((signal) => submitLoginCode(provider, code, signal), setLogin)}
                    onCancel={() => {
                      setLogin(null);
                      void run((signal) => cancelLogin(provider, signal), () => {});
                    }} />
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
