# Decisions

Hikari's Decisions page evaluates supplied text against questions using its specialist model. Users add questions and choose **Evaluate**. Inputs and answers are kept only in the open page and disappear on navigation or reload.

**Choice** selects one supplied option. **Score** evaluates an ordered rubric and can return a value between levels. **Noul** estimates how likely a criterion is to be true. These are model judgments, not verified facts or measured real-world frequencies.

Questions and options must be complete and unambiguous. Invalid inputs produce an error, and unavailable specialist inference does not fall back to the chat model.

Answers stay paired with the inputs that produced them. Editing inputs marks old answers as outdated. **Re-evaluate** and retry use the current form, so later edits can change the request.

**Cancel evaluation** preserves previous answers and ignores late results. Cleanup may take longer, and leaving the page also stops its request. The waiting timer includes time spent in the queue.

See [Decision architecture](../../architecture/hikari/worker-decision.md) for validation and typed results.
