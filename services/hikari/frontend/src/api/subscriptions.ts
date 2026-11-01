import {
  cancelSubscriptionLogin,
  chooseCapabilityProvider,
  disconnectSubscription,
  getSubscriptionLogin,
  getSubscriptions,
  startSubscriptionLogin,
  submitSubscriptionLoginCode,
} from './generated/sdk.gen'
import type {
  CapabilityProviders,
  SubscriptionLogin,
  SubscriptionsResponse,
  SubscriptionStatus,
} from './generated/types.gen'

export type { SubscriptionLogin, SubscriptionsResponse, SubscriptionStatus }
export type SubscriptionProvider = SubscriptionStatus['provider']
export type Provider = 'local' | SubscriptionProvider
export type SubscriptionCapability = keyof CapabilityProviders

export const providerNames: Record<Provider, string> = { local: 'Local', claude: 'Claude', codex: 'Codex' }

/** The account each provider's sign-in page asks for. */
export const accountNames: Record<SubscriptionProvider, string> = { claude: 'Claude', codex: 'ChatGPT' }

export const modelNames: Record<string, string> = {
  'claude-haiku-5-5': 'Claude Haiku 5.5',
  'gpt-6.1-terra': 'GPT 6.1 Terra',
}

function subscriptionError(response: Response | undefined, error: unknown): Error {
  const detail = (error as { detail?: unknown } | undefined)?.detail
  if (typeof detail === 'string') return new Error(detail)
  return new Error(
    response
      ? `Something went wrong (error ${response.status}). Please try again.`
      : 'Hikari is not responding. Check that it is running, then try again.',
  )
}

async function settled<T>(pending: Promise<{ data?: T; error?: unknown; response?: Response }>): Promise<T> {
  const result = await pending
  if (!result.response?.ok || result.data === undefined) throw subscriptionError(result.response, result.error)
  return result.data
}

export const fetchSubscriptions = (signal: AbortSignal) => settled(getSubscriptions({ signal }))

export const disconnect = (provider: SubscriptionProvider, signal: AbortSignal) =>
  settled(disconnectSubscription({ path: { provider }, signal }))

export const chooseProvider = (capability: SubscriptionCapability, provider: Provider, signal: AbortSignal) =>
  settled(chooseCapabilityProvider({ path: { capability }, body: { provider }, signal }))

export const startLogin = (provider: SubscriptionProvider, signal: AbortSignal) =>
  settled(startSubscriptionLogin({ path: { provider }, signal }))

export const loginStatus = (provider: SubscriptionProvider, signal: AbortSignal) =>
  settled(getSubscriptionLogin({ path: { provider }, signal }))

export const submitLoginCode = (provider: SubscriptionProvider, code: string, signal: AbortSignal) =>
  settled(submitSubscriptionLoginCode({ path: { provider }, body: { code }, signal }))

export async function cancelLogin(provider: SubscriptionProvider, signal: AbortSignal): Promise<void> {
  const result = await cancelSubscriptionLogin({ path: { provider }, signal })
  if (!result.response?.ok) throw subscriptionError(result.response, result.error)
}
