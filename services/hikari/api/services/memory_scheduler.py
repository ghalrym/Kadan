"""Decide which workers hold memory on a machine; the API's only global view of it.

Budgets are footprints of [host RAM, device 0, device 1, ...] bytes. A worker's
engine never exceeds its budget, so the ledger is the sum of granted budgets.
VRAM is never pooled: every GPU a worker uses must hold that worker's own
per-GPU floor, and only the rest of its device memory is divided between them.
"""

from copy import deepcopy
from dataclasses import dataclass, field, replace

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
    # (host, device) for the whole workload on one GPU.
    minimum: tuple[int, int]
    preferred: tuple[int, int]
    # Each further GPU needs this much of its own (context and headroom).
    device_floor: int = 0
    # The head job's model and context this was computed for.
    key: tuple | None = None

    @classmethod
    def parse(cls, message: dict) -> "Requirement":
        return cls(
            model=message["model"],
            minimum=tuple(message["minimum"]),
            preferred=tuple(message["preferred"]),
            device_floor=message.get("device_floor", 0),
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
    # A shrink or offload was sent; its old charge stays until the worker
    # confirms the memory is physically free.
    releasing: bool = False
    releasing_since: float = 0.0

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

    def fit(self, requirement: Requirement, free: list[float]) -> list[int] | None:
        """The largest budget up to preferred that meets the minimum.

        The workload may spread over several GPUs, roomiest first. Each GPU it
        uses must have the per-GPU floor free; each extra GPU adds that floor to
        the total. Among placements, the one holding the most of the workload
        wins, and fewer GPUs win ties.
        """
        (host_min, device_min), (host_pref, device_pref) = (
            requirement.minimum,
            requirement.preferred,
        )
        floor = requirement.device_floor
        host = min(host_pref, free[0])
        if host < host_min:
            return None
        budget = [host] + [0] * (len(free) - 1)
        if device_min == 0 and device_pref == 0:
            return budget
        order = sorted(range(1, len(free)), key=lambda index: -free[index])
        best = None
        for count in range(1, len(order) + 1):
            devices = order[:count]
            if free[devices[-1]] <= 0 or free[devices[-1]] < floor:
                break
            extra = floor * (count - 1)
            available = sum(free[index] for index in devices)
            if available < device_min + extra:
                continue
            amount = min(device_pref + extra, available)
            if best is None or amount - extra > best[1] - floor * (len(best[0]) - 1):
                best = (devices, amount)
        if best is None:
            return None
        devices, amount = best
        shares = {index: min(floor, free[index]) for index in devices}
        remaining = amount - sum(shares.values())
        for index in devices:
            given = min(free[index] - shares[index], remaining)
            shares[index] += given
            remaining -= given
        for index, share in shares.items():
            budget[index] = int(share)
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
    """A smaller budget at the worker's minimum, on the GPUs it already uses."""
    if slot.requirement is None:
        return None
    host_min, device_min = slot.requirement.minimum
    floor = slot.requirement.device_floor
    budget = [min(slot.budget[0], host_min)] + [0] * (len(slot.budget) - 1)
    devices = sorted(
        (index for index, value in enumerate(slot.budget[1:], 1) if value),
        key=lambda index: -slot.budget[index],
    )
    remaining = device_min + floor * max(0, len(devices) - 1)
    for index in devices:
        budget[index] = min(slot.budget[index], floor)
        remaining -= budget[index]
    for index in devices:
        given = max(0, min(slot.budget[index] - budget[index], remaining))
        budget[index] += given
        remaining -= given
    return budget if budget != slot.budget else None


def _shrink_options(slot: WorkerSlot) -> list[list[int]]:
    """Smaller budgets to try, least disruptive first: VRAM only, RAM only, both."""
    shrunk = _shrunk(slot)
    if shrunk is None:
        return []
    options = [[slot.budget[0], *shrunk[1:]], [shrunk[0], *slot.budget[1:]], shrunk]
    unique = []
    for option in options:
        if option != slot.budget and option not in unique:
            unique.append(option)
    return unique


def _reserve(ledger: MachineLedger, target: WorkerSlot) -> list[int]:
    """Room kept for the language tier while it is not loaded.

    Other workers never take its minimum, so the language model can start while
    an image or video runs instead of queueing behind it. The minimum is placed
    by the same per-GPU rules as a real grant, so it may span GPUs.
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
        minimum = replace(slot.requirement, preferred=slot.requirement.minimum)
        placed = ledger.fit(minimum, [room - kept for room, kept in zip(free, reserve)])
        if placed is None:
            # It cannot start here anyway; keeping nothing lets others run.
            continue
        reserve = [kept + amount for kept, amount in zip(reserve, placed)]
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
            and not slot.releasing
            and (slot.pending == 0 or slot.priority > target.priority)
        ),
        key=lambda slot: (-slot.priority, slot.last_used),
    )
    actions: list[Action] = []
    for slot in candidates:
        for option in [*_shrink_options(slot), None]:
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
    without it the worker still runs rather than waiting forever. A loaded
    worker is re-granted as if its own budget were free.
    """
    target = ledger.workers[name]
    if target.requirement is None:
        return None
    if target.loaded:
        ledger = deepcopy(ledger)
        target = ledger.workers[name]
        target.budget = [0] * len(ledger.capacity)
        target.loaded = False
    reserve = _reserve(ledger, target)
    planned = _plan(ledger, target, reserve)
    if planned is None and any(reserve):
        planned = _plan(ledger, target, [0] * len(reserve))
    return planned


def needs_grant(ledger: MachineLedger, slot: WorkerSlot) -> bool:
    """Whether a worker with queued jobs lacks a budget meeting its minimum."""
    if not slot.pending or slot.busy or slot.releasing:
        return False
    if not slot.loaded:
        return True
    return slot.requirement is not None and (
        ledger.fit(slot.requirement, slot.budget) is None
    )


def _grown(ledger: MachineLedger, slot: WorkerSlot, free: list[float]) -> list[int]:
    """The worker's budget grown into `free`, capped at its preference.

    It may move onto more GPUs when that placement holds at least its current
    share everywhere; otherwise it grows only on the GPUs it already uses. It
    never shrinks anywhere, so growth needs no release before it is granted.
    """
    requirement = slot.requirement
    available = [room + held for room, held in zip(free, slot.budget)]
    candidate = ledger.fit(requirement, available)
    if candidate is not None and all(
        new >= old for new, old in zip(candidate, slot.budget)
    ):
        return candidate
    host_pref, device_pref = requirement.preferred
    budget = list(slot.budget)
    budget[0] = max(slot.budget[0], min(host_pref, available[0]))
    devices = sorted(
        (index for index, held in enumerate(slot.budget[1:], 1) if held),
        key=lambda index: -free[index],
    )
    cap = device_pref + requirement.device_floor * max(0, len(devices) - 1)
    remaining = max(0, cap - sum(slot.budget[index] for index in devices))
    for index in devices:
        given = max(0, min(free[index], remaining))
        budget[index] += int(given)
        remaining -= given
    return budget


def plan_growth(ledger: MachineLedger) -> list[Action]:
    """Let idle workers grow toward their preference once nothing is waiting.

    A worker may take free memory, and memory from idle lower-priority workers,
    which shrink toward their minimum (they stay loaded). Plans on a copy; the
    caller applies each action once the worker accepts it.
    """
    ledger = deepcopy(ledger)
    if any(needs_grant(ledger, slot) for slot in ledger.workers.values()):
        return []
    actions = []
    for slot in sorted(ledger.workers.values(), key=lambda slot: slot.priority):
        if not slot.loaded or slot.busy or slot.releasing or slot.requirement is None:
            continue
        # Growth never takes the room kept for the language tier.
        free = [
            value - kept for value, kept in zip(ledger.free(), _reserve(ledger, slot))
        ]
        target = _grown(ledger, slot, free)
        donors = []
        for donor in sorted(
            ledger.workers.values(),
            key=lambda other: (-other.priority, other.last_used),
        ):
            shrunk = _shrunk(donor)
            if (
                donor.priority <= slot.priority
                or not donor.loaded
                or donor.busy
                or donor.pending
                or donor.releasing
                or shrunk is None
            ):
                continue
            # Give up only what the grower would actually take, down to the
            # donor's minimum.
            releasable = [old - floor for old, floor in zip(donor.budget, shrunk)]
            ideal = _grown(
                ledger, slot, [room + spare for room, spare in zip(free, releasable)]
            )
            given = [
                max(0, min(spare, want - have))
                for spare, want, have in zip(releasable, ideal, target)
            ]
            if not any(given):
                continue
            free = [room + amount for room, amount in zip(free, given)]
            donors.append(
                Action(
                    "load",
                    donor.name,
                    [old - amount for old, amount in zip(donor.budget, given)],
                )
            )
            target = _grown(ledger, slot, free)
        if not any(
            new - old >= GROWTH_THRESHOLD for new, old in zip(target, slot.budget)
        ):
            continue
        for action in [*donors, Action("load", slot.name, target)]:
            ledger.apply(action)
            actions.append(action)
    return actions


def releases(slot: WorkerSlot, action: Action) -> bool:
    """Whether an action frees memory the worker holds, so it must be confirmed."""
    if action.kind == "offload":
        return True
    return any(new < old for new, old in zip(action.budget, slot.budget))
