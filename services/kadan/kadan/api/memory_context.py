"""Core Memories selection/formatting and append-only archival memory recall."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from .config import ServiceError
from .pydantic_models.resources import Resource, ResourceKind

if TYPE_CHECKING:
    from .services import LocalServices

CORE_MEMORIES_TAG = "core"
CORE_MEMORIES_FALLBACK_LIMIT = 5
CORE_MEMORIES_MAX_ENTRIES = 12
CORE_MEMORIES_MAX_CHARS = 2500
CORE_MEMORIES_HEADER = (
    "Core memories (automatically loaded for this conversation; "
    "stay aware of these facts naturally without repeating them mechanically):\n"
)

RECALL_MAX_ENTRIES = 3
RECALL_MAX_CHARS = 1500
RECALL_MAX_SEARCH_TERMS = 3
RECALL_PER_TERM_LIMIT = 10

_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "always",
        "another",
        "any",
        "anything",
        "are",
        "around",
        "back",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "can",
        "cannot",
        "come",
        "could",
        "did",
        "didn",
        "does",
        "doesn",
        "doing",
        "done",
        "don",
        "dont",
        "down",
        "each",
        "else",
        "even",
        "ever",
        "every",
        "few",
        "for",
        "from",
        "get",
        "give",
        "going",
        "good",
        "got",
        "great",
        "had",
        "has",
        "have",
        "hello",
        "help",
        "her",
        "here",
        "hers",
        "hey",
        "him",
        "his",
        "how",
        "into",
        "its",
        "just",
        "keep",
        "know",
        "last",
        "let",
        "like",
        "little",
        "look",
        "make",
        "many",
        "may",
        "maybe",
        "more",
        "most",
        "much",
        "must",
        "need",
        "never",
        "new",
        "next",
        "nice",
        "not",
        "now",
        "off",
        "okay",
        "one",
        "only",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "please",
        "really",
        "right",
        "said",
        "same",
        "say",
        "see",
        "she",
        "should",
        "some",
        "something",
        "still",
        "such",
        "sure",
        "take",
        "tell",
        "than",
        "thank",
        "thanks",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "things",
        "think",
        "this",
        "those",
        "through",
        "time",
        "today",
        "too",
        "try",
        "two",
        "under",
        "until",
        "use",
        "using",
        "very",
        "want",
        "was",
        "way",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "without",
        "work",
        "would",
        "yeah",
        "yep",
        "yes",
        "yet",
        "you",
        "your",
        "yours",
    }
)


def is_core_memory(resource: Resource) -> bool:
    return any(tag.strip().lower() == CORE_MEMORIES_TAG for tag in resource.tags)


def _memory_sort_key(resource: Resource) -> tuple:
    return (
        0 if resource.pinned else 1,
        -resource.updated_at.timestamp(),
        resource.title.casefold(),
        str(resource.id),
    )


def select_core_memories(
    memories: Sequence[Resource],
    *,
    max_entries: int = CORE_MEMORIES_MAX_ENTRIES,
) -> list[Resource]:
    """Select active core memories (tagged 'core', or <=5 active fallback)."""
    active = [
        item
        for item in memories
        if item.kind == ResourceKind.MEMORY and item.deleted_at is None
    ]
    if not active:
        return []
    tagged = [item for item in active if is_core_memory(item)]
    if tagged:
        chosen = sorted(tagged, key=_memory_sort_key)
        return chosen[:max_entries]
    if len(active) <= CORE_MEMORIES_FALLBACK_LIMIT:
        chosen = sorted(active, key=_memory_sort_key)
        return chosen[: min(max_entries, CORE_MEMORIES_FALLBACK_LIMIT)]
    return []


def _collapse_body(text: str, max_len: int = 320) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max(1, max_len - 1)].rstrip() + "…"


def format_core_memories(
    memories: Sequence[Resource],
    *,
    max_chars: int = CORE_MEMORIES_MAX_CHARS,
    max_entries: int = CORE_MEMORIES_MAX_ENTRIES,
) -> str:
    """Format selected core memories concisely within max_chars."""
    selected = select_core_memories(memories, max_entries=max_entries)
    if not selected or max_chars <= len(CORE_MEMORIES_HEADER) + 16:
        return ""
    lines: list[str] = []
    used = len(CORE_MEMORIES_HEADER)
    for memory in selected:
        title = " ".join(memory.title.split())
        body = _collapse_body(memory.markdown, max_len=360)
        entry = f"- **{title}**: {body}" if body else f"- **{title}**"
        sep = 1 if lines else 0
        remaining = max_chars - used - sep
        if remaining < len(f"- **{title}**"):
            break
        if len(entry) > remaining:
            entry = entry[: remaining - 1].rstrip() + "…"
        lines.append(entry)
        used += sep + len(entry)
    if not lines:
        return ""
    return CORE_MEMORIES_HEADER + "\n".join(lines)


def format_core_memories_section(
    snapshot: str | None, max_chars: int | None = None
) -> str:
    """Return the system-prompt Core Memories block, preserving snapshot verbatim when it fits."""
    if not snapshot or not snapshot.strip():
        return ""
    section = "\n\n" + snapshot.strip()
    if max_chars is None or len(section) <= max_chars:
        return section
    if max_chars <= 3:
        return ""
    return section[: max_chars - 1].rstrip() + "…"


async def fetch_active_memories(
    services: LocalServices, agent_key: str, *, limit: int = 100
) -> list[Resource]:
    if not hasattr(services, "list_resource_kinds") or getattr(services, "client", object()) is None:
        return []
    try:
        return await services.list_resource_kinds(
            agent_key, (ResourceKind.MEMORY,), limit=limit
        )
    except ServiceError:
        return []


async def build_core_memories_snapshot(
    services: LocalServices,
    agent_key: str,
    *,
    max_chars: int = CORE_MEMORIES_MAX_CHARS,
) -> tuple[str, set[UUID]]:
    """Fetch active memories and build the Core Memories snapshot + included IDs."""
    memories = await fetch_active_memories(services, agent_key)
    selected = select_core_memories(memories)
    snapshot = format_core_memories(memories, max_chars=max_chars)
    return snapshot, {item.id for item in selected}


async def persist_memory_state(
    services: LocalServices,
    *,
    agent_key: str,
    conversation_id: UUID,
    expected_version: int | None = None,
    memory_extracted_through_message_id: UUID | None = None,
    core_memories_snapshot: str | None = None,
    core_memories_refreshed_at: datetime | None = None,
    only_if_snapshot_unset: bool = False,
    require_idle: bool = False,
    set_cursor: bool = False,
    set_snapshot: bool = False,
) -> Resource | None:
    """Persist memory cursor and/or epoch-frozen Core Memories snapshot to Tsuchi."""
    from .services import parse_resource

    body: dict[str, object] = {
        "only_if_snapshot_unset": only_if_snapshot_unset,
        "require_idle": require_idle,
    }
    if expected_version is not None:
        body["expected_version"] = expected_version
    if set_cursor and memory_extracted_through_message_id is not None:
        body["memory_extracted_through_message_id"] = str(
            memory_extracted_through_message_id
        )
    if set_snapshot:
        body["core_memories_snapshot"] = core_memories_snapshot
        if core_memories_snapshot is not None:
            refreshed = core_memories_refreshed_at or datetime.now(UTC)
            body["core_memories_refreshed_at"] = refreshed.astimezone(UTC).isoformat()
    value = await services.json_request(
        "Tsuchi",
        "PUT",
        f"/api/agents/{agent_key}/resources/{conversation_id}/memory-state",
        json=body,
    )
    return parse_resource(value)


def extract_recall_terms(text: str, *, max_terms: int = RECALL_MAX_SEARCH_TERMS) -> list[str]:
    """Extract distinctive search terms from a user message, skipping stop words/greetings."""
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]{2,}", text.casefold())
    candidates: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        cleaned = token.strip("'_-")
        if len(cleaned) < 3 or cleaned in _STOP_WORDS or cleaned.isdigit():
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)
    # Prefer longer, more specific terms while keeping deterministic ordering (Timsort is stable).
    candidates.sort(key=lambda term: -len(term))
    return candidates[:max_terms]


def _score_candidate(memory: Resource, terms: Sequence[str]) -> int:
    title_tags = " ".join([memory.title, *memory.tags]).casefold()
    body = memory.markdown.casefold()
    score = 0
    for term in terms:
        if term in title_tags:
            score += 3
        elif term in body:
            score += 1
    return score


async def recall_archival_memories(
    services: LocalServices,
    agent_key: str,
    user_text: str,
    *,
    exclude_ids: set[UUID] | None = None,
    max_entries: int = RECALL_MAX_ENTRIES,
    max_chars: int = RECALL_MAX_CHARS,
) -> list[dict[str, object]]:
    """Search non-core active memories relevant to the triggering user message."""
    if not hasattr(services, "list_resource_kinds") or getattr(services, "client", object()) is None:
        return []
    terms = extract_recall_terms(user_text)
    if not terms:
        return []
    excluded = set(exclude_ids or ())
    results = await asyncio.gather(
        *(
            services.list_resource_kinds(
                agent_key,
                (ResourceKind.MEMORY,),
                q=term,
                limit=RECALL_PER_TERM_LIMIT,
            )
            for term in terms
        ),
        return_exceptions=True,
    )
    candidates: dict[UUID, Resource] = {}
    for batch in results:
        if isinstance(batch, BaseException):
            continue
        for item in batch:
            if (
                item.kind != ResourceKind.MEMORY
                or item.deleted_at is not None
                or item.id in excluded
                or is_core_memory(item)
            ):
                continue
            candidates[item.id] = item
    if not candidates:
        return []
    scored = [
        (_score_candidate(item, terms), item)
        for item in candidates.values()
    ]
    scored = [(score, item) for score, item in scored if score > 0]
    scored.sort(
        key=lambda pair: (
            -pair[0],
            -pair[1].updated_at.timestamp(),
            pair[1].title.casefold(),
            str(pair[1].id),
        )
    )
    recalled: list[dict[str, object]] = []
    used_chars = 0
    for _, item in scored[:max_entries]:
        remaining = max_chars - used_chars
        if remaining < 60:
            break
        title = " ".join(item.title.split())
        per_entry_budget = max(80, min(480, remaining - len(title) - 48))
        excerpt = _collapse_body(item.markdown, max_len=per_entry_budget)
        entry: dict[str, object] = {
            "id": str(item.id),
            "title": title,
            "markdown": excerpt,
        }
        if item.tags:
            entry["tags"] = list(item.tags)
        used_chars += len(title) + len(excerpt) + 48
        recalled.append(entry)
    return recalled
