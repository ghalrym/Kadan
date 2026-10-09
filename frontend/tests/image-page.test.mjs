// Exercise the real React workspace with deferred HTTP responses; no inference.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { after, test } from 'node:test'
import { JSDOM } from 'jsdom'

const output = mkdtempSync(join(tmpdir(), 'kadan-image-page-'))
after(() => rmSync(output, { recursive: true, force: true }))
execFileSync(process.execPath, [resolve('node_modules/typescript/bin/tsc'),
  '--ignoreConfig', 'src/pages/ImagePage.tsx', '--outDir', output,
  '--module', 'commonjs', '--target', 'es2023', '--lib', 'es2023,dom',
  '--jsx', 'react-jsx', '--skipLibCheck', '--esModuleInterop'], { stdio: 'inherit' })
writeFileSync(join(output, 'package.json'), '{"type":"commonjs"}')
symlinkSync(resolve('node_modules'), join(output, 'node_modules'), 'dir')
const dom = new JSDOM('<!doctype html><div id="root"></div>', { url: 'http://kadan.test/image' })
globalThis.window = dom.window
globalThis.document = dom.window.document
globalThis.IS_REACT_ACT_ENVIRONMENT = true
after(() => dom.window.close())
const require = createRequire(import.meta.url)
const React = require('react')
const { createRoot } = require('react-dom/client')
const { MemoryRouter } = require('react-router')
const ImagePage = require(join(output, 'pages/ImagePage.js')).default
const { client } = require(join(output, 'api/generated/client.gen.js'))

const generated = { id: 'new', prompt: 'New apple', mode: 'Generate', aspect: 'square',
  meta: '', urls: ['/v1/images/new/files/0'] }
const older = { ...generated, id: 'older', prompt: 'Older apple', urls: ['/v1/images/older/files/0'] }

for (const retry of [false, true]) {
  test(`${retry ? 'retry' : 'initial'} history arriving after generation preserves the completed image`, async () => {
    const history = Promise.withResolvers()
    const generation = Promise.withResolvers()
    let reads = 0
    let submissions = 0
    client.setConfig({ baseUrl: 'http://kadan.test', fetch: async request => {
      if (request.method === 'GET') {
        reads++
        if (retry && reads === 1) return Response.json({}, { status: 503 })
        return history.promise
      }
      submissions++
      assert.equal(request.url, 'http://kadan.test/v1/images/generations')
      return generation.promise
    } })
    const root = createRoot(document.getElementById('root'))
    try {
      await React.act(async () => {
        root.render(React.createElement(MemoryRouter, { initialEntries: ['/image'] }, React.createElement(ImagePage)))
      })
      const textarea = document.querySelector('textarea')
      await React.act(async () => {
        Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, 'value').set.call(textarea, 'New apple')
        textarea.dispatchEvent(new dom.window.Event('input', { bubbles: true }))
      })
      await React.act(async () => {
        document.querySelector('form').dispatchEvent(new dom.window.Event('submit', { bubbles: true, cancelable: true }))
      })
      assert.equal(submissions, 1)
      if (retry) {
        await React.act(async () => {
          [...document.querySelectorAll('button')].find(button => button.textContent === 'Retry').click()
        })
      }
      await React.act(async () => { generation.resolve(Response.json({ image: generated })) })
      const urls = () => [...document.querySelectorAll('.image-gallery img')].map(image => image.getAttribute('src'))
      assert.deepEqual(urls(), generated.urls)
      await React.act(async () => {
        // Also cover deduplication: the older snapshot must not overwrite a
        // fresh generation's URLs when it contains the same identity.
        history.resolve(Response.json({ images: retry
          ? [{ ...generated, urls: ['/stale.png'] }, older] : [older] }))
      })
      assert.equal(reads, retry ? 2 : 1)
      assert.deepEqual(urls(), [...generated.urls, ...older.urls])
      assert.equal(document.querySelector('[role="alert"]'), null)
    } finally {
      await React.act(async () => root.unmount())
    }
  })
}
