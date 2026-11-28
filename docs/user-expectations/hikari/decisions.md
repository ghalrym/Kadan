# Decisions

Hikari's decisions capability evaluates supplied text against questions using Inkling 2 Mini in a single forward pass without autoregressive reasoning. Kadan uses it internally, and other apps can call it through Hikari's API; Kadan Server has no page for trying it. Hikari keeps no record of inputs or answers. Running decisions preserves any active chat conversation's cached state rather than evicting it.

Decisions share the active Inkling 2 Mini checkpoint downloaded from Kadan Server's AI models after reviewing and accepting its license; no separate decision model download is required. Evaluating a question does not start a model download. A connected language worker alone does not confirm that the checkpoint is available or that evaluation will succeed. Earlier d1 or Laya downloads may remain on disk, but neither model is offered or executed; there is no fallback model.

**Choice** selects one supplied option. **Score** evaluates an ordered rubric of two to ten levels and can return a value between levels. Rubrics outside that range are rejected. **Noul** estimates how likely a criterion is to be true. These are model judgments, not verified facts or measured real-world frequencies.

Questions and options must be complete and unambiguous. Invalid inputs produce an error. Text inputs that exceed the active language model's context limit or 65,536 encoded tokens across the request fail without truncation; existing text-size limits also apply. Requests wait for the language worker if it is busy or needs memory and fail cleanly if memory requirements cannot be met.

Each answer belongs to the request that produced it; retrying sends a new request. A client that cancels or disconnects stops waiting and ignores late results; cleanup may take a moment. Time spent in the queue counts toward how long a request takes.

See [Decision architecture](../../architecture/hikari/worker-decision.md) for prompt formatting, logit-softmax evaluation, KV-cache preservation and validation limits.
