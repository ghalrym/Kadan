# Decisions

Hikari's Decisions page evaluates supplied text against questions using its fixed d1 specialist model on CPU. Users add questions and choose **Evaluate**. Inputs and answers are kept only in the open page and disappear on navigation or reload. This capability is separate from Kadan's language reasoning.

The d1 checkpoint must be downloaded through Settings after reviewing and accepting its license. Evaluating a question does not start a model download. A connected worker alone does not confirm that the checkpoint is available or that evaluation will succeed. Earlier Laya downloads remain on disk, but Laya is no longer offered or executed; there is no fallback model.

**Choice** selects one supplied option. **Score** evaluates an ordered rubric of two to ten levels and can return a value between levels. Rubrics outside that range are rejected; the previous broader Laya range is no longer supported. **Noul** estimates how likely a criterion is to be true. These are model judgments, not verified facts or measured real-world frequencies.

Questions and options must be complete and unambiguous. Invalid inputs produce an error. Text inputs exceeding 65,536 encoded tokens across the entire request fail without truncation; existing text-size limits also apply. This ceiling has not been benchmarked for memory fit or completion. The CPU worker waits for enough RAM and can fail on memory or time limits even below the ceiling. A measured 1,024-token CPU request did not finish within its three-minute execution limit, including model loading.

Answers stay paired with the inputs that produced them. Editing inputs marks old answers as outdated. **Re-evaluate** and retry use the current form, so later edits can change the request.

**Cancel evaluation** preserves previous answers and ignores late results. Cleanup may take longer, and leaving the page also stops its request. The waiting timer includes time spent in the queue. Resources remain reserved until the executor has stopped, including after a time-limit failure.

See [Decision architecture](../../architecture/hikari/worker-decision.md) for validation, resource limits and measured CPU results.
