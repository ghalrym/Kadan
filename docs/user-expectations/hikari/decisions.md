# Decisions

Hikari's Decisions page evaluates supplied text against questions using its specialist model. Users add questions and choose **Evaluate**. Inputs and answers are kept only in the open page and disappear on navigation or reload.

The fixed Laya checkpoint must be downloaded through Settings after reviewing and accepting its license. Evaluating a question does not start a model download. A connected worker alone does not confirm that the checkpoint is available or that evaluation will succeed.

**Choice** selects one supplied option. **Score** evaluates an ordered rubric and can return a value between levels. **Noul** estimates how likely a criterion is to be true. These are model judgments, not verified facts or measured real-world frequencies.

Questions and options must be complete and unambiguous. Invalid inputs produce an error, and unavailable specialist inference does not fall back to the chat model.

Answers stay paired with the inputs that produced them. Editing inputs marks old answers as outdated. **Re-evaluate** and retry use the current form, so later edits can change the request.

**Cancel evaluation** preserves previous answers and ignores late results. Cleanup may take longer, and leaving the page also stops its request. The waiting timer includes time spent in the queue.

See [Decision architecture](../../architecture/hikari/worker-decision.md) for validation and typed results.

An operator-enabled d1 CPU trial is separate from ordinary evaluation, which
continues to use Laya. The trial requires an explicitly selected checkpoint and
an acknowledged, verified download; it never downloads during evaluation or
falls back to Laya. It supports text only, at most 65,536 encoded tokens across
the entire input, and Score rubrics with two to ten levels. Existing text-size
limits still apply. Unsupported rubric sizes are rejected before
evaluation starts; ordinary Laya evaluation keeps its existing rubric support.
Inputs that exceed its limit fail without
truncation. It waits for sufficient RAM, and cancellation or a time-limit failure
keeps its resources reserved until its executor has stopped. The encoded ceiling
does not guarantee that a request fits memory or finishes in time; long inputs
have not been benchmarked at that ceiling. The regular
Decisions page does not select this trial.
