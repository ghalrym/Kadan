# Decisions

Hikari's Decisions page evaluates supplied text against questions using its specialist model. Users add questions and choose **Evaluate**. Inputs and answers are kept only in the open page and disappear on navigation or reload.

**Choice** selects one supplied option. **Score** evaluates an ordered rubric and can return a value between levels. **Noul** estimates how likely a criterion is to be true. These are model judgments, not verified facts or measured real-world frequencies.

Questions and options must be complete and unambiguous. Invalid inputs produce an error, and unavailable specialist inference does not fall back to the chat model.

Laya remains the default. Operators can enable a separate released BF16 d1-2B worker; its connected checkpoint can then be selected on Decisions. Review its license and explicitly download it in Settings first. Selecting d1 affects this page's evaluations, not chat, speech, hardware tiers or agent permissions. An unavailable worker or incomplete download produces an error without choosing another model.

This first d1 integration evaluates text with the existing three question types. Its Score rubric requires 2–10 levels and retains fractional expected scores. Other d1 capabilities, including images, video, extraction and multiple selections, are not exposed. Full input and questions must fit its supported token context and available memory; larger inputs fail without truncation. Long-context quality and physical fit require owner validation. A completed download or passing unit test does not establish successful inference.

Answers stay paired with the inputs that produced them. Editing inputs marks old answers as outdated. **Re-evaluate** and retry use the current form, so later edits can change the request.

**Cancel evaluation** preserves previous answers and ignores late results. Cleanup may take longer, and leaving the page also stops its request. The waiting timer includes time spent in the queue.

See [Decision architecture](../../architecture/hikari/worker-decision.md) for validation and typed results.
See [the opt-in d1 worker](../../architecture/hikari/worker-d1.md) for its distinct context and runtime constraints.
