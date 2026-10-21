# Decisions

Decisions evaluates supplied text with the Laya specialist on CPU, independently of the selected chat model. The page starts with empty state and no questions. Add questions and choose Evaluate explicitly. Inputs and answers remain page-local and are not saved when the page is left or reloaded.

State accepts up to 8,000 characters. Each request needs one to eight questions and must fit 16,000 serialized characters as well as the specialist's token budget. Question keys must be unique and nonblank, and instructions must not be blank. Choice option keys must also be unique within their question. Validation errors identify the affected inputs instead of evaluating an ambiguous request.

Choice returns one supplied option key, with probabilities when available. Score returns the expected zero-based index of an ordered rubric, so a result can be fractional rather than naming exactly one level. The page also shows the nearest level and per-level probabilities when available. Noul returns a probability from zero to one that the criterion is true, with optional True when and False when guidance. These are model judgments about the provided state, not verified facts or measured real-world frequencies.

The initial Choice question has one empty option; the initial Score question has two empty rubric levels. Fill these before evaluating. A Noul question starts without additional true/false guidance. A required checkpoint or worker that is unavailable produces an explicit error rather than using the chat model as a substitute.

Answers remain paired with the exact state and questions evaluated. Editing inputs marks existing answers as outdated rather than relabelling them as answers to the new inputs. Re-evaluate submits the current inputs. A retry after a runtime failure also evaluates the current form, so edits made after the failure can change the request.

Cancel evaluation stops the browser request, preserves previous answers and ignores late results. The CPU worker can take a moment to finish cleanup. Leaving the page also aborts the request. The elapsed time shown while waiting includes request and queue time and is not an isolated CPU inference benchmark.
