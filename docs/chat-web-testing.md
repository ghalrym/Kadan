# Chat web testing

Run the API and frontend using the repository's development setup. Vite proxies
`/v1` to `http://127.0.0.1:8000`; `API_PROXY_TARGET` can override the backend
address for development. Open `/chat` and send a message.

The page posts the current conversation to `/v1/chat/completions` with
`{ "messages": [{ "role": "user", "text": "Hello" }] }` through the generated
SDK. It omits `model` so the backend controls model selection, and expects
`{ "message": { "role": "assistant", "text": "…", "meta": null } }`.
The chat wiring alone does not replace the backend's existing mock completion:
real inference requires the model runtime change. No canned chat history is
inserted into the UI.

Only the current page session retains messages. Navigating away or refreshing
clears them. Input is limited to 8,000 characters and a conversation to 12 turns;
start a new chat at the limit. Sending is blocked while a request is pending.
Failed or cancelled messages can be retried without duplicating the user turn,
or discarded by starting a new chat. Cancel aborts the browser request and
ignores late results; it does not guarantee that backend inference has stopped.

From `frontend/`, run:

```sh
node --test tests/chat.test.mjs
npm run lint
npm run build
```

The tests compile the real SDK adapter into a temporary directory and replace
fetch with controlled responses. They verify the request contract and history,
HTTP errors and recovery, network and malformed-response failures, cancellation,
and input/history bounds. They do not validate GPU inference or browser layout.
For a manual browser check, stop the API to trigger a failure, restart it and
retry; also cancel a pending request and verify that no late reply is appended.
