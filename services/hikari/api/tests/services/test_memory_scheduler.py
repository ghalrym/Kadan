import unittest

from api.services.memory_scheduler import (
    GIB,
    Action,
    MachineLedger,
    Requirement,
    plan_growth,
    plan_load,
)


def ledger(ram=64, vram=(24,)):
    return MachineLedger.from_machine(
        {
            "ram_total": 2 * ram * GIB,
            "ram_free": ram * GIB,
            "devices": [
                {"vram_total": 2 * v * GIB, "vram_free": v * GIB} for v in vram
            ],
        },
        ["llm", "decision", "tts", "speech", "formatting", "image", "video"],
    )


def requirement(host_min, device_min, host_pref=None, device_pref=None):
    return Requirement(
        model="m",
        minimum=(host_min * GIB, device_min * GIB),
        preferred=(
            (host_pref if host_pref is not None else host_min) * GIB,
            (device_pref if device_pref is not None else device_min) * GIB,
        ),
    )


def load(machine, name, budget, *, pending=0, busy=False, used=0.0, need=None):
    slot = machine.workers[name]
    slot.loaded, slot.busy, slot.pending, slot.last_used = True, busy, pending, used
    slot.budget = [value * GIB for value in budget]
    slot.requirement = need


class CapacityTests(unittest.TestCase):
    def test_capacity_is_free_memory_at_connection_not_totals(self):
        machine = ledger(ram=64, vram=(24, 12))
        self.assertEqual(machine.capacity, [64 * GIB, 24 * GIB, 12 * GIB])
        self.assertNotIn("store", machine.workers)

    def test_budget_is_capped_at_preferred_and_placed_on_the_roomiest_device(self):
        machine = ledger(vram=(8, 20))
        machine.workers["image"].requirement = requirement(2, 6, 4, 10)
        self.assertEqual(
            plan_load(machine, "image"),
            [Action("load", "image", [4 * GIB, 0, 10 * GIB])],
        )

    def test_flexible_language_model_takes_what_is_free_above_its_minimum(self):
        machine = ledger(ram=64, vram=(24,))
        load(machine, "image", [4, 14])
        machine.workers["llm"].requirement = requirement(10, 6, 60, 40)
        self.assertEqual(
            plan_load(machine, "llm"),
            [Action("load", "llm", [60 * GIB, 10 * GIB])],
        )

    def test_host_only_worker_needs_no_device(self):
        machine = ledger(vram=(1,))
        load(machine, "image", [0, 1])
        machine.workers["decision"].requirement = requirement(4, 0)
        self.assertEqual(
            plan_load(machine, "decision"), [Action("load", "decision", [4 * GIB, 0])]
        )

    def test_unknown_requirement_waits(self):
        self.assertIsNone(plan_load(ledger(), "image"))


class DisplacementTests(unittest.TestCase):
    def test_lowest_priority_then_least_recent_idle_worker_is_offloaded_first(self):
        machine = ledger(vram=(24,))
        load(machine, "image", [4, 8], used=1.0)
        load(machine, "video", [4, 8], used=5.0)
        load(machine, "tts", [4, 8], used=0.0)
        machine.workers["speech"].requirement = requirement(1, 8)
        self.assertEqual(
            plan_load(machine, "speech"),
            [Action("offload", "video"), Action("load", "speech", [GIB, 8 * GIB])],
        )

    def test_language_model_shrinks_before_it_is_offloaded(self):
        machine = ledger(vram=(24,))
        load(machine, "llm", [40, 24], need=requirement(10, 6, 60, 40))
        machine.workers["image"].requirement = requirement(4, 14)
        # Only VRAM is short, so the language model keeps its RAM.
        self.assertEqual(
            plan_load(machine, "image"),
            [
                Action("load", "llm", [40 * GIB, 6 * GIB]),
                Action("load", "image", [4 * GIB, 14 * GIB]),
            ],
        )

    def test_ram_only_shortage_shrinks_only_ram(self):
        machine = ledger(ram=48, vram=(24,))
        load(machine, "llm", [40, 6], need=requirement(10, 6, 60, 40))
        machine.workers["image"].requirement = requirement(16, 4)
        self.assertEqual(
            plan_load(machine, "image"),
            [
                Action("load", "llm", [10 * GIB, 6 * GIB]),
                Action("load", "image", [16 * GIB, 4 * GIB]),
            ],
        )

    def test_shrinking_that_is_not_enough_becomes_an_offload(self):
        machine = ledger(vram=(24,))
        load(machine, "llm", [40, 24], need=requirement(10, 12, 60, 40))
        machine.workers["video"].requirement = requirement(4, 20)
        self.assertEqual(
            plan_load(machine, "video"),
            [Action("offload", "llm"), Action("load", "video", [4 * GIB, 20 * GIB])],
        )

    def test_busy_workers_are_never_displaced(self):
        machine = ledger(vram=(24,))
        load(machine, "video", [4, 24], busy=True)
        machine.workers["llm"].requirement = requirement(4, 8)
        self.assertIsNone(plan_load(machine, "llm"))

    def test_queued_work_is_displaced_only_by_higher_priority(self):
        machine = ledger(vram=(24,))
        load(machine, "image", [4, 24], pending=2)
        machine.workers["video"].requirement = requirement(4, 8)
        self.assertIsNone(plan_load(machine, "video"))
        machine.workers["tts"].requirement = requirement(4, 8)
        self.assertEqual(
            plan_load(machine, "tts"),
            [Action("offload", "image"), Action("load", "tts", [4 * GIB, 8 * GIB])],
        )

    def test_queued_language_model_keeps_memory_against_lower_priority(self):
        machine = ledger(vram=(24,))
        load(machine, "llm", [40, 24], pending=1, need=requirement(10, 6, 60, 40))
        machine.workers["image"].requirement = requirement(4, 14)
        self.assertIsNone(plan_load(machine, "image"))

    def test_several_offloads_combine_until_the_worker_fits(self):
        machine = ledger(vram=(24,))
        load(machine, "image", [2, 10], used=1.0)
        load(machine, "video", [2, 10], used=2.0)
        machine.workers["llm"].requirement = requirement(4, 20)
        self.assertEqual(
            plan_load(machine, "llm"),
            [
                Action("offload", "video"),
                Action("offload", "image"),
                Action("load", "llm", [4 * GIB, 20 * GIB]),
            ],
        )


class GrowthTests(unittest.TestCase):
    def test_idle_flexible_worker_grows_into_free_memory(self):
        machine = ledger(ram=64, vram=(24,))
        load(machine, "llm", [10, 6], need=requirement(10, 6, 60, 40))
        self.assertEqual(
            plan_growth(machine), [Action("load", "llm", [60 * GIB, 24 * GIB])]
        )
        # Planning never changes the ledger; the hub applies accepted actions.
        self.assertEqual(machine.workers["llm"].budget, [10 * GIB, 6 * GIB])

    def test_no_growth_while_another_worker_waits_or_below_threshold(self):
        machine = ledger(ram=64, vram=(24,))
        load(machine, "llm", [10, 6], need=requirement(10, 6, 60, 40))
        machine.workers["image"].pending = 1
        self.assertEqual(plan_growth(machine), [])
        machine.workers["image"].pending = 0
        load(machine, "llm", [60, 23.5], need=requirement(10, 6, 60, 40))
        self.assertEqual(plan_growth(machine), [])

    def test_busy_worker_does_not_grow(self):
        machine = ledger()
        load(machine, "llm", [10, 6], busy=True, need=requirement(10, 6, 60, 40))
        self.assertEqual(plan_growth(machine), [])


class LanguagePriorityTests(unittest.TestCase):
    def test_other_workers_leave_room_for_the_language_model(self):
        machine = ledger(ram=64, vram=(24,))
        machine.workers["llm"].requirement = requirement(10, 6, 30, 22)
        machine.workers["image"].requirement = requirement(16, 2, 40, 33)
        self.assertEqual(
            plan_load(machine, "image"),
            [Action("load", "image", [40 * GIB, 18 * GIB])],
        )

    def test_growth_keeps_the_language_reserve(self):
        machine = ledger(ram=64, vram=(24,))
        machine.workers["llm"].requirement = requirement(10, 6, 30, 22)
        load(machine, "image", [40, 18], need=requirement(16, 2, 40, 33))
        self.assertEqual(plan_growth(machine), [])

    def test_only_the_language_tier_is_reserved(self):
        machine = ledger(ram=64, vram=(24,))
        machine.workers["tts"].requirement = requirement(2, 6)
        machine.workers["video"].requirement = requirement(4, 3, 40, 33)
        self.assertEqual(
            plan_load(machine, "video"),
            [Action("load", "video", [40 * GIB, 24 * GIB])],
        )

    def test_worker_still_runs_when_the_reserve_does_not_leave_room(self):
        machine = ledger(ram=64, vram=(8,))
        machine.workers["llm"].requirement = requirement(10, 6, 30, 22)
        machine.workers["image"].requirement = requirement(16, 4, 40, 33)
        self.assertEqual(
            plan_load(machine, "image"),
            [Action("load", "image", [40 * GIB, 8 * GIB])],
        )

    def test_language_model_reclaims_only_the_vram_it_lacks_from_an_idle_image(self):
        machine = ledger(ram=60, vram=(24,))
        load(machine, "llm", [30, 6], need=requirement(10, 6, 30, 22))
        load(machine, "image", [30, 18], need=requirement(16, 2, 40, 33))
        self.assertEqual(
            plan_growth(machine),
            [
                Action("load", "image", [30 * GIB, 2 * GIB]),
                Action("load", "llm", [30 * GIB, 22 * GIB]),
            ],
        )

    def test_busy_or_queued_workers_are_not_reclaimed(self):
        for state in ({"busy": True}, {"pending": 1}):
            with self.subTest(state=state):
                machine = ledger(ram=60, vram=(24,))
                load(machine, "llm", [30, 6], need=requirement(10, 6, 30, 22))
                load(
                    machine, "image", [30, 18], need=requirement(16, 2, 40, 33), **state
                )
                self.assertEqual(plan_growth(machine), [])


def spread(host_min, device_min, host_pref, device_pref, floor):
    need = requirement(host_min, device_min, host_pref, device_pref)
    need.device_floor = floor * GIB
    return need


class MultiGpuTests(unittest.TestCase):
    def test_lone_workload_spreads_across_both_gpus(self):
        machine = ledger(ram=64, vram=(24, 24))
        machine.workers["llm"].requirement = spread(10, 6, 30, 40, 2)
        # 40 GiB preferred on one GPU, plus one extra floor for the second.
        self.assertEqual(
            plan_load(machine, "llm"),
            [Action("load", "llm", [30 * GIB, 24 * GIB, 18 * GIB])],
        )

    def test_minimum_may_be_split_when_no_single_gpu_holds_it(self):
        machine = ledger(ram=64, vram=(8, 8))
        machine.workers["llm"].requirement = spread(10, 12, 30, 40, 2)
        budget = plan_load(machine, "llm")[0].budget
        self.assertEqual(budget, [30 * GIB, 8 * GIB, 8 * GIB])

    def test_every_gpu_used_needs_its_own_floor(self):
        machine = ledger(ram=64, vram=(10, 1))
        machine.workers["llm"].requirement = spread(10, 10, 30, 40, 2)
        self.assertEqual(
            plan_load(machine, "llm"),
            [Action("load", "llm", [30 * GIB, 10 * GIB, 0])],
        )

    def test_competing_single_gpu_workers_take_separate_gpus(self):
        machine = ledger(ram=64, vram=(24, 24))
        load(machine, "llm", [10, 24, 0], need=spread(10, 6, 10, 20, 2))
        machine.workers["image"].requirement = spread(4, 2, 4, 20, 2)
        self.assertEqual(
            plan_load(machine, "image"),
            [Action("load", "image", [4 * GIB, 0, 20 * GIB])],
        )

    def test_shrink_keeps_a_floor_on_every_gpu_it_uses(self):
        machine = ledger(ram=64, vram=(24, 24))
        load(machine, "llm", [30, 24, 18], need=spread(10, 6, 30, 40, 2))
        machine.workers["video"].requirement = spread(4, 20, 4, 20, 3)
        actions = plan_load(machine, "video")
        self.assertEqual(
            actions[0], Action("load", "llm", [30 * GIB, 6 * GIB, 2 * GIB])
        )
        self.assertEqual(actions[1].worker, "video")

    def test_growth_can_add_a_gpu(self):
        machine = ledger(ram=64, vram=(24, 24))
        load(machine, "llm", [30, 24, 0], need=spread(10, 6, 30, 40, 2))
        self.assertEqual(
            plan_growth(machine),
            [Action("load", "llm", [30 * GIB, 24 * GIB, 18 * GIB])],
        )

    def test_growth_never_exceeds_the_preference_across_gpus(self):
        # A roomier second GPU must not add a copy beside the current share.
        machine = ledger(ram=64, vram=(21, 24))
        load(machine, "llm", [30, 20, 0], need=spread(10, 6, 30, 20, 2))
        self.assertEqual(plan_growth(machine), [])

    def test_growth_in_place_stays_within_the_spread_preference(self):
        machine = ledger(ram=64, vram=(24, 24))
        load(machine, "llm", [30, 10, 10], need=spread(10, 6, 30, 20, 2))
        machine.workers["image"].loaded = True
        machine.workers["image"].budget = [0, 0, 4 * GIB]
        grown = plan_growth(machine)[0].budget
        self.assertLessEqual(sum(grown[1:]), 22 * GIB)

    def test_language_reserve_spans_gpus_like_a_real_grant(self):
        machine = ledger(ram=64, vram=(24, 24))
        machine.workers["llm"].requirement = spread(10, 30, 10, 30, 2)
        machine.workers["image"].requirement = spread(4, 2, 4, 40, 2)
        # Language would start as [24, 8], so the image gets the other 16.
        self.assertEqual(
            plan_load(machine, "image"),
            [Action("load", "image", [4 * GIB, 0, 16 * GIB])],
        )
