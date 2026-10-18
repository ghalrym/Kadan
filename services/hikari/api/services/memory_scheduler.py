"""Decide which workers hold memory on a machine; the API's only global view of it.

Budgets are footprints of [host RAM, device 0, device 1, ...] bytes. A worker's
engine never exceeds its budget, so the ledger is the sum of granted budgets.
"""

from copy import deepcopy
from dataclasses import dataclass, field

GIB = 1024**3

# Lower numbers keep memory longer and are never displaced by higher numbers
# while they still have work queued.
PRIORITY = {
    "llm": 1,
    "decision": 1,
    "tts": 2,
    "speech": 3,
    "formatting": 3,
    "image": 4,
    "video": 5,
}
TOP_PRIORITY = min(PRIORITY.values())

# Growing a resident engine reloads it, so only grow for a worthwhile gain.
GROWTH_THRESHOLD = GIB


@dataclass
class Requirement:
    model: str
    minimum: tuple[int, int]  # (host, device)
    preferred: tuple[int, int]

    @classmethod
    def parse(cls, message: dict) -> "Requirement":
        return cls(
            model=message["model"],
            minimum=tuple(message["minimum"]),
            preferred=tuple(message["preferred"]),
        )


@dataclass
class WorkerSlot:
    name: str
    loaded: bool = False
    busy: bool = False
    pending: int = 0
    budget: list[int] = field(default_factory=list)
    last_used: float = 0.0
    requirement: Requirement | None = None

    @property
    def priority(self) -> int:
        return PRIORITY[self.name]


@dataclass
class Action:
    kind: str  # "load" grants a budget; "offload" releases one.
    worker: str
    budget: list[int] | None = None


@dataclass
class MachineLedger:
    capacity: list[int]
    workers: dict[str, WorkerSlot]

    @classmethod
    def from_machine(cls, machine: dict, workers: list[str]) -> "MachineLedger":
        # A machine connects with nothing loaded, so its free memory then (after
        # the native probe's margins) is what its workers can share. Totals
        # would count memory other processes, like a desktop, already hold.
        capacity = [
            machine["ram_free"],
            *(device["vram_free"] for device in machine["devices"]),
        ]
        return cls(
            capacity=capacity,
            workers={
                name: WorkerSlot(name=name, budget=[0] * len(capacity))
                for name in workers
                if name in PRIORITY
            },
        )

    def free(self) -> list[int]:
        free = list(self.capacity)
        for slot in self.workers.values():
            for index, value in enumerate(slot.budget):
                free[index] -= value
        return free

    def fit(self, requirement: Requirement, free: list[int]) -> list[int] | None:
        """The largest budget up to preferred that meets the minimum, on one device."""
        (host_min, device_min), (host_pref, device_pref) = (
            requirement.minimum,
            requirement.preferred,
        )
        host = min(host_pref, free[0])
        if host < host_min:
            return None
        best = None
        for device in range(1, len(free)):
            amount = min(device_pref, free[device])
            if amount >= device_min and (best is None or amount > best[1]):
                best = (device, amount)
        budget = [host] + [0] * (len(free) - 1)
        if device_min == 0 and device_pref == 0:
            return budget
        if best is None:
            return None
        budget[best[0]] = best[1]
        return budget

    def apply(self, action: Action) -> None:
        slot = self.workers[action.worker]
        if action.kind == "offload":
            slot.loaded = False
            slot.budget = [0] * len(self.capacity)
        else:
            slot.loaded = True
            slot.budget = list(action.budget)


def _shrunk(slot: WorkerSlot) -> list[int] | None:
    """A smaller budget at the worker's minimum, on the device it already uses."""
    if slot.requirement is None:
        return None
    host_min, device_min = slot.requirement.minimum
    budget = [min(slot.budget[0], host_min)]
    for value in slot.budget[1:]:
        budget.append(min(value, device_min) if value else 0)
    return budget if budget != slot.budget else None


def _reserve(ledger: MachineLedger, target: WorkerSlot) -> list[int]:
    """Room kept for the language tier while it is not loaded.

    Other workers never take its minimum, so the language model can start while
    an image or video runs instead of queueing behind it.
    """
    free = ledger.free()
    reserve = [0] * len(free)
    for slot in ledger.workers.values():
        if (
            slot.priority != TOP_PRIORITY
            or slot.priority >= target.priority
            or slot.loaded
            or slot.requirement is None
        ):
            continue
        host_min, device_min = slot.requirement.minimum
        reserve[0] += host_min
        if device_min and len(free) > 1:
            device = max(
                range(1, len(free)), key=lambda index: free[index] - reserve[index]
            )
            reserve[device] += device_min
    return reserve


def _plan(
    ledger: MachineLedger, target: WorkerSlot, reserve: list[int]
) -> list[Action] | None:
    free = [value - kept for value, kept in zip(ledger.free(), reserve)]
    budget = ledger.fit(target.requirement, free)
    if budget is not None:
        return [Action("load", target.name, budget)]
    candidates = sorted(
        (
            slot
            for slot in ledger.workers.values()
            if slot.name != target.name
            and slot.loaded
            and not slot.busy
            and (slot.pending == 0 or slot.priority > target.priority)
        ),
        key=lambda slot: (-slot.priority, slot.last_used),
    )
    actions: list[Action] = []
    for slot in candidates:
        shrunk = _shrunk(slot)
        for option in ([shrunk] if shrunk is not None else []) + [None]:
            released = slot.budget if option is None else option
            trial = list(free)
            for index, value in enumerate(slot.budget):
                trial[index] += value - (0 if option is None else released[index])
            kind = "offload" if option is None else "load"
            step = Action(kind, slot.name, option)
            budget = ledger.fit(target.requirement, trial)
            if budget is not None:
                return [*actions, step, Action("load", target.name, budget)]
            # A shrink that is not enough is superseded by the offload.
            if option is None:
                actions.append(step)
                free = trial
    return None


def plan_load(ledger: MachineLedger, name: str) -> list[Action] | None:
    """Actions that make room for `name` and grant it a budget, or None to wait.

    Busy workers are never touched. A worker with queued jobs is displaced only
    by a strictly higher priority. Room comes from the lowest priority first,
    least recently used first, shrinking a flexible worker before offloading it.
    Higher-priority workers keep room for their minimum when the machine has it;
    without it the worker still runs rather than waiting forever.
    """
    target = ledger.workers[name]
    if target.requirement is None:
        return None
    reserve = _reserve(ledger, target)
    planned = _plan(ledger, target, reserve)
    if planned is None and any(reserve):
        planned = _plan(ledger, target, [0] * len(reserve))
    return planned


def _grown(slot: WorkerSlot, free: list[float]) -> list[int]:
    """The worker's budget grown into `free`, capped at its preference."""
    host_pref, device_pref = slot.requirement.preferred
    budget = list(slot.budget)
    budget[0] = min(host_pref, slot.budget[0] + free[0])
    for index, value in enumerate(slot.budget[1:], 1):
        if value:
            budget[index] = min(device_pref, value + free[index])
    return budget


def plan_growth(ledger: MachineLedger) -> list[Action]:
    """Let idle workers grow toward their preference once nothing is waiting.

    A worker may take free memory, and memory from idle lower-priority workers,
    which shrink to their minimum (they stay loaded). Plans on a copy; the
    caller applies each action once the worker accepts it.
    """
    ledger = deepcopy(ledger)
    if any(slot.pending and not slot.loaded for slot in ledger.workers.values()):
        return []
    actions = []
    for slot in sorted(ledger.workers.values(), key=lambda slot: slot.priority):
        if not slot.loaded or slot.busy or slot.requirement is None:
            continue
        # Growth never takes the room kept for the language tier.
        free = [
            value - kept for value, kept in zip(ledger.free(), _reserve(ledger, slot))
        ]
        preferred = _grown(slot, [float("inf")] * len(free))
        target = _grown(slot, free)
        donors = []
        for donor in sorted(
            ledger.workers.values(),
            key=lambda other: (-other.priority, other.last_used),
        ):
            if target == preferred:
                break
            shrunk = _shrunk(donor)
            if (
                donor.priority <= slot.priority
                or not donor.loaded
                or donor.busy
                or donor.pending
                or shrunk is None
            ):
                continue
            # Give up only what the grower still lacks, down to the minimum.
            short = [want - have for want, have in zip(preferred, target)]
            given = [
                max(0, min(old - floor, need))
                for old, floor, need in zip(donor.budget, shrunk, short)
            ]
            if not any(given):
                continue
            free = [value + amount for value, amount in zip(free, given)]
            donors.append(
                Action(
                    "load",
                    donor.name,
                    [old - amount for old, amount in zip(donor.budget, given)],
                )
            )
            target = _grown(slot, free)
        if not any(
            new - old >= GROWTH_THRESHOLD for new, old in zip(target, slot.budget)
        ):
            continue
        for action in [*donors, Action("load", slot.name, target)]:
            ledger.apply(action)
            actions.append(action)
    return actions
