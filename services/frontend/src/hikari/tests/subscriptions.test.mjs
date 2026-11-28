import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { createContext, SourceTextModule, SyntheticModule } from 'node:vm'
import ts from 'typescript'

const sdkNames = ['cancelSubscriptionLogin', 'chooseCapabilityProvider', 'disconnectSubscription', 'getSubscriptionLogin',
  'getSubscriptions', 'startSubscriptionLogin', 'submitSubscriptionLoginCode']

// Replace only the generated HTTP boundary; exercise the actual API adapter.
async function subscriptionApi(response) {
  const calls = []
  const sdk = new SyntheticModule(sdkNames, function () {
    for (const name of sdkNames) this.setExport(name, async args => { calls.push({ name, args }); return response })
  })
  const source = await readFile(new URL('../api/subscriptions.ts', import.meta.url), 'utf8')
  const module = new SourceTextModule(stripTypeScriptTypes(source))
  await module.link(specifier => {
    assert.equal(specifier, './generated/sdk.gen')
    return sdk
  })
  await module.evaluate()
  return { api: module.namespace, calls }
}

const snapshot = (image = 'local', codex = true) => ({
  subscriptions: [
    { provider: 'claude', connected: true, model: 'claude-haiku-5-5' },
    { provider: 'codex', connected: codex, model: 'gpt-6.1-terra' },
  ],
  capabilities: { language: 'local', image },
})

test('chooses the image provider separately from language', async () => {
  const { api, calls } = await subscriptionApi({ response: { ok: true }, data: snapshot('codex') })
  const signal = new AbortController().signal
  assert.deepEqual(await api.chooseImages('codex', signal), snapshot('codex'))
  assert.deepEqual(calls[0].args.path, { capability: 'image' })
  assert.deepEqual(calls[0].args.body, { provider: 'codex' })
  assert.equal(calls[0].args.signal, signal)
  await api.chooseLanguage('claude', signal)
  assert.deepEqual(calls[1].args.path, { capability: 'language' })
})

test('names ChatGPT as the only subscription image provider', async () => {
  const { api } = await subscriptionApi({})
  assert.deepEqual(api.imageProviderNames, { local: 'Local', codex: 'ChatGPT' })
})

test('shows the service explanation when an image provider is refused', async () => {
  const { api } = await subscriptionApi({ response: { ok: false, status: 409 }, error: { detail: 'Connect your Codex subscription first.' } })
  await assert.rejects(api.chooseImages('codex', new AbortController().signal), /Connect your Codex subscription first/)
})
