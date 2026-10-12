import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
const source = stripTypeScriptTypes(await readFile(new URL('../src/api/imageProgress.ts', import.meta.url), 'utf8'))
const { describeImageProgress, estimateRemaining, formatDuration } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

const snapshot = (stage, value = 0, image = { image_index: 1, image_count: 4, steps_completed: 0, steps_total: 50 }) =>
  ({ job_id: 'job', workload: 'image', stage, value, observed_unix_ns: 1, image })

test('describes idle and non-image worker activity', () => {
  const idle = describeImageProgress(null, null)
  assert.equal(idle.phase, 'queued')
  assert.equal(idle.fraction, null)
  const other = describeImageProgress({ ...snapshot('running'), workload: 'llm', image: null }, null)
  assert.equal(other.phase, 'queued')
  assert.match(other.detail, /chat/)
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
  assert.equal(view.detail, 'Denoising · step 25 of 50')
  assert.ok(Math.abs(view.fraction - (1 + 0.03 + 0.46) / 4) < 1e-9)
  assert.equal(describeImageProgress(snapshot('vae_up_block', 1), null).phase, 'decode')
})

test('keeps the previous phase for ambiguous weight placement stages', () => {
  const prompt = describeImageProgress(snapshot('text_layer', 3), null)
  assert.equal(describeImageProgress(snapshot('gpu_0_weight_planned_bytes', 9), prompt).phase, 'prompt')
  assert.equal(describeImageProgress(snapshot('gpu_0_weight_planned_bytes', 9), null).phase, 'loading')
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
  assert.equal(view.title, 'Worker status unavailable')
})

test('an idle worker after generation does not imply request completion', () => {
  const decode = describeImageProgress(snapshot('vae_up_block', 3), null)
  const idle = describeImageProgress(null, decode)
  assert.equal(idle.title, 'Worker idle')
  assert.equal(idle.fraction, null)
  assert.equal(idle.jobId, null)
  assert.equal(idle.stale, false)
})

test('overlapping image jobs reset worker counters and phase on job changes', () => {
  const other = describeImageProgress(snapshot('step_completed', 25, { image_index: 2, image_count: 4, steps_completed: 25, steps_total: 50 }), null)
  assert.equal(other.jobId, 'job')
  const next = describeImageProgress({ ...snapshot('gpu_0_weight_planned_bytes', 9, null), job_id: 'next-job' }, other)
  assert.equal(next.jobId, 'next-job')
  assert.equal(next.phase, 'loading')
  assert.equal(next.imageIndex, 1)
  assert.equal(next.imageCount, 1)
  assert.equal(next.step, 0)
  assert.equal(next.fraction, null)
  const chat = describeImageProgress({ ...snapshot('running'), workload: 'llm', job_id: 'chat-job', image: null }, other)
  assert.equal(chat.title, 'Worker busy')
  assert.equal(chat.fraction, null)
})

test('estimates remaining time once progress is measurable', () => {
  assert.equal(estimateRemaining(null, { ns: 1, fraction: 0.5 }), null)
  assert.equal(estimateRemaining({ ns: 0, fraction: 0.1 }, { ns: 1e9, fraction: 0.11 }), null)
  assert.equal(estimateRemaining({ ns: 0, fraction: 0 }, { ns: 10e9, fraction: 0.25 }), 30)
  assert.equal(formatDuration(75), '1:15')
  assert.equal(formatDuration(3725), '1:02:05')
})
