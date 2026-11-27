# Decisions worker

Hikari routes Decisions (`POST /v1/decisions`) directly to the native `llm`
worker (`ChatEngine` / `kadan::text::Engine`) running the active resident
Inkling 2 Mini checkpoint (`inkling-2-mini-*`). The public request has no model
selector or feature flag; `DecisionTask` targets `FEATURE_WORKERS["decisions"] = "llm"`
without requiring a separate checkpoint download or separate worker process. The
earlier `d1-2b-cpu-f32` checkpoint and `lfm3_decision` worker are retired. See
[Decisions](../../user-expectations/hikari/decisions.md).

## Single-forward-pass logit softmax

Decisions execute with zero autoregressive token generation:

1. Each question in a `DecisionRequest` (1 to 8 questions sharing `state`) is
   rendered with Inkling 2 Mini's native chat template using thinking effort
   `0` and ending directly at `<|message_model|><|content_text|>` (no thinking
   block). User-supplied `state`, instructions, options and rubric levels are
   passed as literal tokenizer ranges so control tokens cannot be injected.
2. `kadan::text::Engine::prefill_logits` runs the bounded prefill pass and
   returns the final token's full vocabulary logit vector without altering
   normal decode behavior or Strata MoE placement rules.
3. For requests with multiple questions sharing `state`, `ChatEngine` computes
   the exact shared leading token prefix across questions and checkpoints it via
   `checkpoint_at(shared_prefix)` on the first question, reusing the prefilled
   `state` KV/recurrent prefix for the remaining questions in the same request.
4. `score_decision_logits` computes log-softmax over the vocabulary, selects the
   highest log-probability among each candidate group's single-token variants
   (`yes`/`Yes`/`YES` and leading-space forms for `Noul`, resolved single-token
   option codes `A`, `B`, ... with optional leading space for `Choice`, and
   digits `0..K-1` with optional leading space for `Score`), applies softmax
   across groups, and formats the typed `DecisionAnswer` JSON in request order.
   `Score` computes the expected zero-based ordinal mean `sum(i * p_i)`.

## Active chat KV-cache preservation

Decisions share the resident `ChatEngine` with interactive chat, so preparing
`DecisionRequest` uses the exact same active model ID and saved context limit
configuration as `ChatRequest` to avoid unloading or reloading the engine.

Before running any decision prefill, `ChatEngine::execute_decisions` snapshots
the active chat conversation's live prefix via `model_->take_prefix()` and holds
it in an RAII guard. On scope exit — whether evaluation succeeds, fails, or is
cancelled — the guard erases the transient decision prefix checkpoint from
`CheckpointPool` and unconditionally restores the active chat prefix via
`model_->restore_prefix(...)` (or aborts transient state cleanly if no chat
prefix was active). Decisions therefore never evict or corrupt the live chat
conversation's KV cache.

## Validation and limits

Both the FastAPI schema (`DecisionRequest`) and native request parser enforce:

- `state`: non-blank, at most 8,000 characters
- `questions`: 1 to 8 questions with unique non-blank keys and non-blank
  instructions
- `Choice`: at least one option with unique non-blank keys
- `Score`: 2 to 10 non-blank rubric levels
- Total serialized request size: at most 16,000 characters
- Encoded tokens: each question must fit the active language model's context
  limit and the combined total across questions must not exceed 65,536 tokens,
  failing without truncation.

Cancellation and client disconnect stop prefill cleanly via `cancel`, restore
the chat prefix snapshot, and release request-local scratch reservations before
returning a terminal event. Automated validation uses pure model-free unit tests
only.
