from uuid import uuid4

import pytest
from test_agent_runs import (  # noqa: F401
    claim,
    enqueue,
    expire_run,
    repository,
    run_state,
)
from tsuchi.config import StorageError
from tsuchi.models import AgentRunState
from tsuchi.repository import Repository


def test_progress_decision_survives_restart_and_cannot_change(repository):  # noqa: F811
    run = claim(repository, enqueue(repository))
    action_id = uuid4()
    saved = repository.update_run(
        "alpha",
        run.id,
        run_state(run, progress_update_decisions={action_id: "suppress"}),
        run.version,
        run.owner_id,
    )
    restarted = Repository(repository.data_dir)
    restarted.initialize()
    assert restarted.get_run("alpha", run.id).progress_update_decisions == {
        action_id: "suppress"
    }
    for decisions in [{}, {action_id: "send"}]:
        with pytest.raises(StorageError, match="immutable"):
            repository.update_run(
                "alpha",
                run.id,
                run_state(saved, progress_update_decisions=decisions),
                saved.version,
                run.owner_id,
            )


def test_expired_owner_cannot_add_a_progress_decision_while_claiming(repository):  # noqa: F811
    run = claim(repository, enqueue(repository))
    expired = expire_run(repository, run)
    with pytest.raises(StorageError, match="Recovering a run"):
        repository.update_run(
            "alpha",
            run.id,
            run_state(expired, progress_update_decisions={uuid4(): "send"}),
            expired.version,
            expired.owner_id,
        )


def test_progress_decision_map_is_bounded_and_typed():
    with pytest.raises(ValueError):
        AgentRunState(progress_update_decisions={uuid4(): "unknown"})
    with pytest.raises(ValueError):
        AgentRunState(progress_update_decisions={uuid4(): "send" for _ in range(25)})
