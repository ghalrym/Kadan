"""CPU-only replay of complete, ordered native device-weight cache traces."""

import json
from collections import Counter, OrderedDict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

UINT64_MAX = (1 << 64) - 1


def integer(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


@dataclass(frozen=True)
class TileKey:
    """The complete 96-byte native key, including exact expert/file identity."""

    words: tuple[int, ...]

    @classmethod
    def parse(cls, value: object) -> "TileKey":
        if not isinstance(value, list) or len(value) != 12:
            raise ValueError("key must contain all 12 native uint64 fields")
        return cls(tuple(integer(word, "key field") for word in value))

    @property
    def size(self) -> int:
        size = self.words[10] * self.words[11]
        if not 0 < size <= UINT64_MAX:
            raise ValueError("tile byte size must be positive and fit uint64")
        return size

    @property
    def source(self) -> tuple[int, ...]:
        return self.words[:9]


@dataclass
class Metrics:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    hit_bytes: int = 0
    transferred_bytes: int = 0
    admitted_bytes: int = 0
    replacement_bytes: int = 0
    retired_bytes: int = 0
    reloads: int = 0
    peak_bytes: int = 0


@dataclass
class Policy:
    name: Literal["first_admission", "lru", "decayed_frequency_lru"]
    capacity: int
    entries: OrderedDict[TileKey, int] = field(default_factory=OrderedDict)
    admitted: set[TileKey] = field(default_factory=set)
    allocated: int = 0
    metrics: Metrics = field(default_factory=Metrics)
    frequencies: Counter = field(default_factory=Counter)

    def access(self, key: TileKey, eligible: bool) -> None:
        size = key.size
        self.metrics.accesses += 1
        if self.name == "decayed_frequency_lru":
            if self.metrics.accesses % 1024 == 0:
                self.frequencies = Counter(
                    {
                        tile: count // 2
                        for tile, count in self.frequencies.items()
                        if count >= 2
                    }
                )
            self.frequencies[key] += 1
        if key in self.entries:
            self.metrics.hits += 1
            self.metrics.hit_bytes += size
            if self.name != "first_admission":
                self.entries.move_to_end(key)
            return
        self.metrics.misses += 1
        self.metrics.transferred_bytes += size
        self.metrics.reloads += key in self.admitted
        if not eligible or size > self.capacity:
            return
        if self.name == "decayed_frequency_lru":
            needed = self.allocated + size - self.capacity
            for victim, victim_size in self.entries.items():
                if needed <= 0:
                    break
                # Compare frequency per byte exactly, without floating-point
                # ranking or consulting any future access in the trace.
                if (
                    self.frequencies[key] * victim_size
                    <= self.frequencies[victim] * size
                ):
                    return
                needed -= victim_size
        if self.name != "first_admission":
            while self.allocated + size > self.capacity:
                _, removed = self.entries.popitem(last=False)
                self.allocated -= removed
                self.metrics.replacement_bytes += removed
        if self.allocated + size <= self.capacity:
            self.entries[key] = size
            self.admitted.add(key)
            self.allocated += size
            self.metrics.admitted_bytes += size
            self.metrics.peak_bytes = max(self.metrics.peak_bytes, self.allocated)

    def retire(self, source: tuple[int, ...] | None = None) -> None:
        for key in list(self.entries):
            if source is None or key.source == source:
                removed = self.entries.pop(key)
                self.allocated -= removed
                self.metrics.retired_bytes += removed

    def report(self) -> dict:
        return {**asdict(self.metrics), "allocated_bytes": self.allocated}


@dataclass
class Bank:
    device: int
    maximum: int
    capacity: int
    seq: int = 0
    emitted: int = 0
    ended: bool = False
    allocated: int = 0
    actual: dict[TileKey, int] = field(default_factory=dict)
    observed: Counter = field(default_factory=Counter)
    refusals: Counter = field(default_factory=Counter)
    policies: list[Policy] = field(default_factory=list)

    def apply(self, event: dict) -> None:
        seq = integer(event.get("seq"), "seq")
        if seq != self.seq + 1:
            raise ValueError("bank sequence must be contiguous")
        self.seq = seq
        if integer(event.get("device"), "device") != self.device:
            raise ValueError("bank device changed")
        kind = event.get("event")
        self.ended = False
        if kind == "truncated":
            raise ValueError("truncated cache traces cannot be replayed")
        if kind != "summary":
            self.emitted += 1
        capacity = integer(event.get("capacity"), "capacity")
        allocated = integer(event.get("allocated"), "allocated")
        key = TileKey.parse(event.get("key")) if kind != "summary" else None
        if key is not None and integer(event.get("bytes"), "bytes") != integer(
            key.words[10] * key.words[11], "key byte size"
        ):
            raise ValueError("event bytes disagree with native key count and width")
        if kind == "bound":
            if self.actual or capacity > self.maximum:
                raise ValueError("capacity change requires an empty admitted bank")
            self.capacity = capacity
            for policy in self.policies:
                if policy.allocated:
                    raise ValueError("capacity change requires an empty simulated bank")
                policy.capacity = capacity
        elif capacity != self.capacity:
            raise ValueError("capacity changed without a bound event")
        if kind == "open":
            if seq != 1 or allocated:
                raise ValueError("open must establish a known empty bank")
        elif kind == "access":
            hit, eligible = event.get("hit"), event.get("eligible")
            if type(hit) is not bool or type(eligible) is not bool:
                raise ValueError("access hit and eligible must be booleans")
            if hit != (key in self.actual):
                raise ValueError("observed hit disagrees with admitted tile state")
            self.observed["accesses"] += 1
            self.observed["hits" if hit else "misses"] += 1
            self.observed["hit_bytes" if hit else "miss_bytes"] += key.size
            for policy in self.policies:
                policy.access(key, eligible)
        elif kind == "admit":
            if key in self.actual:
                raise ValueError("duplicate observed admission")
            self.actual[key] = key.size
            self.allocated += key.size
            self.observed["admitted_bytes"] += key.size
        elif kind == "refuse":
            if not isinstance(event.get("reason"), str) or not event["reason"]:
                raise ValueError("refusal requires a reason")
            self.observed["refused"] += 1
            self.refusals[event["reason"]] += 1
        elif kind == "retire_source":
            if any(key.words[9:]):
                raise ValueError("retire_source must contain only a source identity")
            for policy in self.policies:
                policy.retire(key.source)
        elif kind == "drop":
            removed = self.actual.pop(key, None)
            if removed is None and event.get("reason") == "rollback":
                removed = 0
            elif removed != key.size:
                raise ValueError("drop must acknowledge an observed allocation")
            self.allocated -= removed
            if event.get("reason") != "rollback":
                self.observed["retired_bytes"] += key.size
        elif kind == "clear":
            if self.actual:
                raise ValueError("clear requires acknowledged physical drops")
            for policy in self.policies:
                policy.retire()
        elif kind == "summary":
            if event.get("quarantined") is not False:
                raise ValueError(
                    "quarantined or unspecified final state cannot be replayed"
                )
            if event.get("output_failed") is not False:
                raise ValueError("trace output failed or is unspecified")
            if integer(event.get("dropped"), "dropped"):
                raise ValueError("trace dropped records")
            if integer(event.get("emitted"), "emitted") != self.emitted:
                raise ValueError("summary emitted count disagrees with trace records")
            for name in (
                "accesses",
                "hits",
                "misses",
                "hit_bytes",
                "miss_bytes",
                "admitted_bytes",
                "refused",
                "retired_bytes",
            ):
                if integer(event.get(name), name) != self.observed[name]:
                    raise ValueError(f"summary {name} disagrees with trace events")
            self.ended = True
        elif kind != "bound":
            raise ValueError(f"unknown cache trace event: {kind!r}")
        if allocated != self.allocated or allocated > self.capacity:
            raise ValueError(
                "observed allocated bytes disagree with acknowledged tiles"
            )
        for policy in self.policies:
            if not 0 <= policy.allocated <= policy.capacity:
                raise ValueError("simulated occupancy exceeded the byte budget")


def replay_events(events: Iterable[dict]) -> dict:
    """Compare online tile policies; observed physical refusals are not simulated."""
    banks: dict[int, Bank] = {}
    for event in events:
        if not isinstance(event, dict):
            raise TypeError("trace record must be a JSON object")
        if "weight_cache_trace" not in event:
            continue
        if (
            type(event["weight_cache_trace"]) is not int
            or event["weight_cache_trace"] != 1
        ):
            raise ValueError("unsupported weight cache trace version")
        bank_id = integer(event.get("bank"), "bank")
        if bank_id not in banks:
            if event.get("event") != "open":
                raise ValueError("every bank must start with an empty open event")
            device = integer(event.get("device"), "device")
            capacity = integer(event.get("capacity"), "capacity")
            banks[bank_id] = Bank(
                device,
                capacity,
                capacity,
                policies=[
                    Policy("first_admission", capacity),
                    Policy("lru", capacity),
                    Policy("decayed_frequency_lru", capacity),
                ],
            )
        banks[bank_id].apply(event)
    if not banks or any(not bank.ended for bank in banks.values()):
        raise ValueError("trace requires a final summary for every opened bank")
    return {
        "scope": "Ordered native tiles; ideal byte budgets, not model timing or output quality.",
        "policy_unit": "Complete 96-byte tile key; no expert/window grouping or future accesses.",
        "decayed_frequency_lru": "Online tile access counts halve (integer floor) every 1024 bank accesses. Admit over LRU victims only when candidate frequency/byte strictly exceeds every required victim; ties retain residents. No training prefix or Strata-style verify window.",
        "reload_definition": "A miss on a tile previously admitted by that simulated policy.",
        "observed_vs_simulated": "Observed refusals include physical failures; ideal policies use eligibility and capacity only.",
        "banks": [
            {
                "bank": bank_id,
                "device": bank.device,
                "capacity_bytes": bank.capacity,
                "observed": {
                    **dict(bank.observed),
                    "refusal_reasons": dict(bank.refusals),
                },
                "policies": {policy.name: policy.report() for policy in bank.policies},
            }
            for bank_id, bank in sorted(banks.items())
        ],
    }


def replay_trace(path: Path) -> dict:
    """Read NDJSON, permitting unrelated worker JSON records between trace records."""
    with path.open() as stream:
        return replay_events(json.loads(line) for line in stream if line.strip())
