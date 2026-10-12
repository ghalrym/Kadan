import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
const source = stripTypeScriptTypes(await readFile(new URL('../src/api/imageProgress.ts', import.meta.url), 'utf8'))
const { describeImageProgress, estimateRemaining, formatDuration } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

const snapshot = (stage, value = 0, image = { image_index: 1, image_count: 4, steps_completed: 0, steps_total: 50 }) =>
  ({ job_id: 'job', workload: 'image', stage, value, observed_unix_ns: 1, image })

test('waits in the queue until the worker reports image work', () => {
  const idle = describeImageProgress(null, 4, null)
  assert.equal(idle.phase, 'queued')
  assert.equal(idle.fraction, null)
  const other = describeImageProgress({ ...snapshot('running'), workload: 'llm', image: null }, 4, null)
  assert.equal(other.phase, 'queued')
  assert.match(other.detail, /chat/)
  assert.equal(describeImageProgress(snapshot('waiting_for_resources'), 4, null).detail, 'Waiting for GPU memory')
})

test('reports model loading without an overall estimate', () => {
  const view = describeImageProgress(snapshot('denoiser_block_loaded', 12), 4, null)
  assert.equal(view.milestone, 1)
  assert.equal(view.detail, 'Loading denoiser · block 12')
  assert.equal(view.fraction, null)
})

test('maps denoising steps to batch progress', () => {
  const view = describeImageProgress(snapshot('step_completed', 25, { image_index: 2, image_count: 4, steps_completed: 25, steps_total: 50 }), 4, null)
  assert.equal(view.milestone, 2)
  assert.equal(view.title, 'Generating image 2 of 4')
  assert.equal(view.detail, 'Denoising · step 25 of 50')
  assert.ok(Math.abs(view.fraction - (1 + 0.03 + 0.46) / 4) < 1e-9)
  assert.equal(describeImageProgress(snapshot('vae_up_block', 1), 4, null).phase, 'decode')
})

test('keeps the previous phase for ambiguous weight placement stages', () => {
  const prompt = describeImageProgress(snapshot('text_layer', 3), 4, null)
  assert.equal(describeImageProgress(snapshot('gpu_0_weight_planned_bytes', 9), 4, prompt).phase, 'prompt')
  assert.equal(describeImageProgress(snapshot('gpu_0_weight_planned_bytes', 9), 4, null).phase, 'loading')
})

test('a cleared snapshot after generation reads as delivery', () => {
  const decode = describeImageProgress(snapshot('vae_up_block', 3, { image_index: 4, image_count: 4, steps_completed: 50, steps_total: 50 }), 4, null)
  const done = describeImageProgress(null, 4, decode)
  assert.equal(done.phase, 'finishing')
  assert.equal(done.milestone, 3)
  assert.equal(done.fraction, 1)
  const loading = describeImageProgress(snapshot('vae_loaded'), 4, null)
  assert.equal(describeImageProgress(null, 4, loading).phase, 'queued')
})

test('estimates remaining time once progress is measurable', () => {
  assert.equal(estimateRemaining(null, { ns: 1, fraction: 0.5 }), null)
  assert.equal(estimateRemaining({ ns: 0, fraction: 0.1 }, { ns: 1e9, fraction: 0.11 }), null)
  assert.equal(estimateRemaining({ ns: 0, fraction: 0 }, { ns: 10e9, fraction: 0.25 }), 30)
  assert.equal(formatDuration(75), '1:15')
  assert.equal(formatDuration(3725), '1:02:05')
})
