# Model-facing tool context

Tool definitions share the conversation model's context with instructions,
personas, recent delivered chat and this run's results. This baseline measures
the current definitions without inference or checkpoint downloads; it does not
recommend removing tools.

The LFM adapter sends structured OpenAI-style tools to Hikari, where the pinned
LFM tokenizer applies its chat template. The table measures the actual schema
array serialized as compact UTF-8 JSON, excluding the request envelope. All
current schema characters are ASCII, so byte and character counts match.
The runner's reservation instead uses `tool_characters`: the legacy native
tool section, including its instructions and whitespace. This is the existing
admission estimate, not the exact rendered LFM prompt length.

| Configuration | Tools | LFM schema JSON bytes | JSON token estimate | Reserved tool characters | Reservation token estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| No computer | 13 | 8,479 | 2,827 | 9,842 | 3,281 |
| Computer | 28 | 23,020 | 7,674 | 25,564 | 8,522 |
| Computer and vision | 29 | 23,505 | 7,835 | 26,068 | 8,690 |

Token columns are explicitly estimates: ceiling of characters divided by three,
matching Kadan's context-budget heuristic. No tokenizer was loaded. They are not
token counts from the pinned LFM tokenizer and exclude its final chat-template
overhead. The checked-in base `AGENT.md` adds 6,558 characters before personas,
history, event context and tool results. Those variable inputs must be measured
separately for a particular run. The runner also reserves 1,024 tokens for the
reply and template and includes at most 97 recent delivered messages.

The largest individual compact schema is `browser_execute_action` at 4,962
bytes; both history schemas total 2,461 bytes. The three Codex schemas total
1,793 bytes. Tool count alone therefore does not describe context cost.

Reproduce from the repository root with the Kadan dependencies installed:

```sh
PYTHONPATH=services/kadan .venv/bin/python - <<'PY'
import json
from math import ceil
from pathlib import Path
from langchain_core.utils.function_calling import convert_to_openai_tool
from kadan.runner.agent_inference import tool_characters
from kadan.runner.tools import (
    ALL_TOOLS, CONVERSATION_TOOLS, MEMORY_TOOLS, PERSONA_TOOLS, VISION_TOOLS,
)

groups = {
    "no computer": [*CONVERSATION_TOOLS, *MEMORY_TOOLS, *PERSONA_TOOLS],
    "computer": ALL_TOOLS,
    "computer and vision": [*ALL_TOOLS, *VISION_TOOLS],
}
for name, tools in groups.items():
    schemas = [convert_to_openai_tool(tool) for tool in tools]
    text = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
    reserved = tool_characters(tools)
    print(name, len(tools), len(text.encode("utf-8")), ceil(len(text) / 3),
          reserved, ceil(reserved / 3))
print("base instructions characters", len(Path("services/kadan/AGENT.md").read_text()))
PY
```

See [conversation context and delivery](agent-conversations.md) for admission
and durable message semantics, and [agent action expectations](../../user-expectations/kadan/agent-actions.md)
for user-visible limits.
