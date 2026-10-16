"""Small manual traces exercise placement and capture validation without models."""

import unittest
from copy import deepcopy

from bin.cache_replay import Policy, TileKey, replay_events


def key(expert: int, size: int = 4, first: int = 0) -> TileKey:
    return TileKey(
        (1, 77, 1000, 1, 2, 3, 4, expert * 100, expert * 100 + 100, first, size, 1)
    )


def capture(records: list[dict], capacity: int = 8, **summary: int) -> list[dict]:
    totals = {
        "accesses": 0,
        "hits": 0,
        "misses": 0,
        "hit_bytes": 0,
        "miss_bytes": 0,
        "admitted_bytes": 0,
        "refused": 0,
        "retired_bytes": 0,
    }
    totals.update(summary)
    events = [
        {"event": "open"},
        *records,
        {
            "event": "summary",
            "quarantined": False,
            "output_failed": False,
            "dropped": 0,
            "emitted": len(records) + 1,
            **totals,
        },
    ]
    result = []
    for seq, event in enumerate(events, 1):
        raw_key = event.get("key", [0] * 12)
        result.append(
            {
                "weight_cache_trace": 1,
                "bank": 1,
                "device": 0,
                "seq": seq,
                "key": raw_key,
                "bytes": raw_key[10] * raw_key[11],
                "hit": False,
                "eligible": True,
                "capacity": capacity,
                "allocated": 0,
                "reason": "",
                **event,
            }
        )
    return result


def access(tile: TileKey, **fields: object) -> dict:
    return {"event": "access", "key": list(tile.words), **fields}


class CacheReplayTests(unittest.TestCase):
    def test_byte_bounded_lru_replaces_multiple_variable_sized_tiles(self) -> None:
        policy = Policy("lru", 10)
        a, b, c = key(1, 3), key(2, 4), key(3, 8)
        for tile in (a, b, a, c, c, a):
            policy.access(tile, True)
        self.assertEqual(policy.metrics.hits, 2)
        self.assertEqual(policy.metrics.transferred_bytes, 18)
        self.assertEqual(policy.metrics.replacement_bytes, 15)
        self.assertEqual(policy.metrics.reloads, 1)
        self.assertEqual(policy.allocated, 3)
        self.assertEqual(policy.metrics.peak_bytes, 8)

    def test_first_admission_keeps_prefix_while_dense_scan_thrashes_lru(self) -> None:
        prefix, lru = Policy("first_admission", 8), Policy("lru", 8)
        for tile in (key(1), key(2), key(3), key(4), key(1), key(2)):
            prefix.access(tile, True)
            lru.access(tile, True)
        self.assertEqual(prefix.metrics.hits, 2)
        self.assertEqual(lru.metrics.hits, 0)
        self.assertEqual(lru.metrics.replacement_bytes, 16)
        self.assertEqual(lru.metrics.reloads, 2)
        self.assertEqual(prefix.metrics.replacement_bytes, 0)

    def test_oversize_and_streaming_misses_do_not_evict_retained_tiles(self) -> None:
        for name in ("first_admission", "lru"):
            policy = Policy(name, 8)
            policy.access(key(1), True)
            policy.access(key(2, 9), True)
            policy.access(key(3), False)
            policy.access(key(1), False)
            self.assertEqual(policy.metrics.hits, 1)
            self.assertEqual(policy.metrics.transferred_bytes, 17)
            self.assertEqual(policy.metrics.replacement_bytes, 0)
            self.assertEqual(policy.allocated, 4)

    def test_online_frequency_filter_protects_hot_tiles_then_adapts(self) -> None:
        policy = Policy("decayed_frequency_lru", 8)
        a, b, c = key(1), key(2), key(3)
        for tile in (a, b, a, b, c, key(4)):
            policy.access(tile, True)
        self.assertEqual(list(policy.entries), [a, b])
        self.assertEqual(policy.metrics.replacement_bytes, 0)
        # Only the observed prefix trains the filter. A third visit to C can
        # replace the older twice-accessed resident; no future popularity leaks.
        policy.access(c, True)
        self.assertEqual(list(policy.entries), [a, b])
        policy.access(c, True)
        self.assertEqual(list(policy.entries), [b, c])
        for _ in range(1024 - policy.metrics.accesses):
            policy.access(c, True)
        self.assertEqual(policy.frequencies[a], 1)
        self.assertEqual(policy.frequencies[b], 1)
        self.assertEqual(policy.frequencies[c], 510)

    def test_frequency_rejection_does_not_partially_evict_multiple_victims(
        self,
    ) -> None:
        policy = Policy("decayed_frequency_lru", 8)
        a, b, large = key(1), key(2), key(3, 8)
        for tile in (a, b, b, b, large, large, large):
            policy.access(tile, True)
        self.assertEqual(list(policy.entries), [a, b])
        self.assertEqual(policy.metrics.replacement_bytes, 0)

    def test_retire_exact_expert_including_all_block_ranges(self) -> None:
        policy = Policy("lru", 12)
        a, tail, other = key(1), key(1, first=4), key(2)
        for tile in (a, tail, other):
            policy.access(tile, True)
        policy.retire(a.source)
        self.assertEqual(list(policy.entries), [other])
        self.assertEqual(policy.metrics.retired_bytes, 8)
        policy.access(a, True)
        self.assertEqual(policy.metrics.reloads, 1)
        policy.retire()
        self.assertEqual(policy.allocated, 0)

    def test_full_key_fields_distinguish_tiles(self) -> None:
        policy = Policy("lru", 100)
        original = key(1)
        policy.access(original, True)
        for index in range(12):
            words = list(original.words)
            words[index] += 1
            policy.access(TileKey(tuple(words)), True)
        self.assertEqual(policy.metrics.hits, 0)
        self.assertEqual(len(policy.entries), 13)

    def test_observed_physical_refusal_differs_from_ideal_capacity(self) -> None:
        a = key(1)
        events = capture(
            [
                access(a),
                {"event": "refuse", "key": list(a.words), "reason": "physical"},
                access(a),
                {"event": "admit", "key": list(a.words), "allocated": 4},
                access(a, hit=True, allocated=4),
                {"event": "drop", "key": list(a.words), "reason": "release"},
                {"event": "clear"},
            ],
            accesses=3,
            hits=1,
            misses=2,
            hit_bytes=4,
            miss_bytes=8,
            admitted_bytes=4,
            refused=1,
            retired_bytes=4,
        )
        bank = replay_events(events)["banks"][0]
        self.assertEqual(bank["observed"]["refusal_reasons"], {"physical": 1})
        self.assertEqual(bank["policies"]["first_admission"]["hits"], 2)
        self.assertEqual(bank["policies"]["lru"]["transferred_bytes"], 4)
        self.assertEqual(bank["policies"]["lru"]["allocated_bytes"], 0)

    def test_retire_directive_applies_even_without_observed_retention(self) -> None:
        a = key(1)
        source = list(a.source) + [0, 0, 0]
        events = capture(
            [access(a), {"event": "retire_source", "key": source}, access(a)],
            accesses=2,
            misses=2,
            miss_bytes=8,
        )
        result = replay_events(events)["banks"][0]["policies"]["lru"]
        self.assertEqual(result["hits"], 0)
        self.assertEqual(result["retired_bytes"], 4)
        self.assertEqual(result["reloads"], 1)

    def test_banks_have_independent_sequence_and_budgets(self) -> None:
        first = capture([access(key(1))], accesses=1, misses=1, miss_bytes=4)
        second = deepcopy(first)
        for event in second:
            event.update(bank=2, device=1, capacity=2)
        mixed = [event for pair in zip(first, second) for event in pair]
        banks = replay_events(mixed)["banks"]
        self.assertEqual(banks[0]["policies"]["lru"]["allocated_bytes"], 4)
        self.assertEqual(banks[1]["policies"]["lru"]["allocated_bytes"], 0)

    def test_bound_changes_apply_after_clear_without_inflating_budget(self) -> None:
        events = capture(
            [
                {"event": "bound", "capacity": 4},
                access(key(1), capacity=4),
                {"event": "clear", "capacity": 4},
                {"event": "bound", "capacity": 2},
                access(key(1), capacity=2),
            ],
            accesses=2,
            misses=2,
            miss_bytes=8,
        )
        events[-1]["capacity"] = 2
        bank = replay_events(events)["banks"][0]
        self.assertEqual(bank["capacity_bytes"], 2)
        for result in bank["policies"].values():
            self.assertEqual(result["allocated_bytes"], 0)
            self.assertEqual(result["peak_bytes"], 4)
        events[1]["capacity"] = 9
        with self.assertRaises(ValueError):
            replay_events(events)

    def test_short_interim_summary_then_later_release_summary(self) -> None:
        events = capture([access(key(1))], accesses=1, misses=1, miss_bytes=4)
        short = events[-1]
        for field in ("key", "bytes", "eligible", "hit", "reason"):
            short.pop(field)
        events.extend(
            [
                {**events[0], "seq": 4, "event": "clear"},
                {**short, "seq": 5, "emitted": 3},
            ]
        )
        result = replay_events(events)["banks"][0]["policies"]["lru"]
        self.assertEqual(result["allocated_bytes"], 0)
        self.assertEqual(result["retired_bytes"], 4)

    def test_rollback_drop_does_not_invent_admitted_or_retired_bytes(self) -> None:
        events = capture(
            [
                access(key(1)),
                {"event": "drop", "key": list(key(1).words), "reason": "rollback"},
            ],
            accesses=1,
            misses=1,
            miss_bytes=4,
        )
        self.assertEqual(replay_events(events)["banks"][0]["observed"]["misses"], 1)

    def test_reject_incomplete_or_corrupt_traces(self) -> None:
        with self.assertRaises(TypeError):
            replay_events([[]])
        valid = capture([access(key(1))], accesses=1, misses=1, miss_bytes=4)
        bad = [[], valid[1:], valid[:-1]]
        for field, value in (
            ("seq", 99),
            ("key", [1]),
            ("allocated", 1),
            ("capacity", 9),
            ("bytes", 3),
            ("hit", True),
            ("eligible", 1),
        ):
            altered = deepcopy(valid)
            altered[1][field] = value
            bad.append(altered)
        for field, value in (
            ("dropped", 1),
            ("quarantined", True),
            ("output_failed", True),
            ("misses", 2),
            ("emitted", 0),
            ("event", "truncated"),
        ):
            altered = deepcopy(valid)
            altered[-1][field] = value
            bad.append(altered)
        for events in bad:
            with self.subTest(events=events), self.assertRaises(ValueError):
                replay_events(events)


if __name__ == "__main__":
    unittest.main()
