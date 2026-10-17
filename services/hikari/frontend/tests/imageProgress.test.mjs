import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
const source = stripTypeScriptTypes(await readFile(new URL('../src/api/imageProgress.ts', import.meta.url), 'utf8'))
const { describeImageProgress: describe, estimateRemaining, formatDuration } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

const describeImageProgress = (progress, previous, requestId = 'job') => describe(progress, previous, requestId)

const phases = { waiting_for_resources: 'queued', running: 'prompt', text_layer: 'prompt', step_completed: 'denoise', vae_up_block: 'decode', denoiser_block_loaded: 'loading' }
const snapshot = (stage, value = 0, image = { image_index: 1, image_count: 4, steps_completed: 0, steps_total: 50 }) =>
  ({ job_id: 'job', workload: 'image', stage, value, observed_unix_ns: 1, image: image ? { ...image, phase: phases[stage] ?? 'loading' } : null })

test('idle and unrelated jobs never claim request progress', () => {
  for (const progress of [null, { ...snapshot('running'), workload: 'llm' }, { ...snapshot('running'), job_id: 'other' }]) {
    const view = describeImageProgress(progress, null)
    assert.equal(view.fraction, null)
    assert.equal(view.title, 'Waiting for images')
  }
  assert.equal(describeImageProgress(snapshot('waiting_for_resources'), null).detail, 'Waiting for GPU memory')
})

test('reports model loading without an overall estimate', () => {
  const view = describeImageProgress(snapshot('denoiser_block_loaded', 12), null)
  assert.equal(view.milestone, 1)
  assert.equal(view.detail, 'Loading denoiser · block 12')
  assert.equal(view.fraction, null)
})

test('maps denoising steps to batch progress', () => {
  const view = describeImageProgress(snapshot('step_completed', 25, { image_index: 2, image_count: 4, steps_completed: 25, steps_total: 50 }), null)
  assert.equal(view.milestone, 2)
  assert.equal(view.title, 'Generating image 2 of 4')
  assert.equal(view.detail, 'Denoising · step 26 of 50')
  assert.ok(Math.abs(view.fraction - (1 + 0.03 + 0.46) / 4) < 1e-9)
  assert.equal(describeImageProgress(snapshot('vae_up_block', 1), null).phase, 'decode')
})

test('a first poll during qkv retains the API denoising phase', () => {
  const progress = snapshot('qkv', 3)
  progress.image.phase = 'denoise'
  progress.image.steps_completed = 1
  const view = describeImageProgress(progress, null)
  assert.equal(view.phase, 'denoise')
  assert.equal(view.detail, 'Denoising · step 2 of 50')
  assert.equal(Math.round(view.fraction * 100), 1)
  const decode = snapshot('gpu_0_weight_planned_bytes', 99)
  decode.image.phase = 'decode'
  assert.equal(describeImageProgress(decode, null).phase, 'decode')
})

test('failed polls preserve generation as stale until polling recovers', () => {
  const generating = describeImageProgress(snapshot('step_completed', 25, { image_index: 2, image_count: 4, steps_completed: 25, steps_total: 50 }), null)
  const failed = describeImageProgress(undefined, generating)
  assert.deepEqual(failed, { ...generating, stale: true })
  assert.notEqual(failed.fraction, 1)
  assert.deepEqual(describeImageProgress(undefined, failed), failed)
  const recovered = describeImageProgress(snapshot('step_completed', 26, { image_index: 2, image_count: 4, steps_completed: 26, steps_total: 50 }), failed)
  assert.equal(recovered.stale, false)
  assert.equal(recovered.step, 26)
})

test('an unavailable first poll does not claim progress', () => {
  const view = describeImageProgress(undefined, null)
  assert.equal(view.stale, true)
  assert.equal(view.fraction, null)
  assert.equal(view.title, 'Waiting for images')
})

test('idle and unavailable polls retain the last sample without claiming completion', () => {
  const decode = describeImageProgress(snapshot('vae_up_block', 3), null)
  const idle = describeImageProgress(null, decode)
  assert.equal(idle.fraction, decode.fraction)
  assert.equal(idle.stale, true)
  assert.notEqual(idle.fraction, 1)
})

test('new requests discard previous counters and unrelated global activity', () => {
  const old = describeImageProgress(snapshot('step_completed', 25), null)
  const next = describeImageProgress(snapshot('step_completed', 25), old, 'new')
  assert.equal(next.jobId, 'new')
  assert.equal(next.step, 0)
  assert.equal(next.fraction, null)
})

test('older successful polls cannot regress the retained phase or counters', () => {
  const current = describeImageProgress({ ...snapshot('step_completed', 25), observed_unix_ns: 10 }, null)
  assert.equal(describeImageProgress(snapshot('running'), current), current)
})

test('estimates remaining time once progress is measurable', () => {
  assert.equal(estimateRemaining(null, { ns: 1, fraction: 0.5 }), null)
  assert.equal(estimateRemaining({ ns: 0, fraction: 0.1 }, { ns: 1e9, fraction: 0.11 }), null)
  assert.equal(estimateRemaining({ ns: 0, fraction: 0 }, { ns: 10e9, fraction: 0.25 }), 30)
  assert.equal(formatDuration(75), '1:15')
  assert.equal(formatDuration(3725), '1:02:05')
})
