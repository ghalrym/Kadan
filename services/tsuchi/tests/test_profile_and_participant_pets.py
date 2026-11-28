"""Unit tests for the user profile and Group Chat participant pets."""

from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_pets import ready_pet
from tsuchi.config import StorageError
from tsuchi.models import (
    AgentInput,
    GroupChatParticipantInput,
    GroupChatParticipantUpdate,
    PetRef,
    UserProfileUpdate,
)
from tsuchi.repository import Repository


@pytest.fixture
def repository(tmp_path):
    repository = Repository(tmp_path)
    repository.initialize()
    repository.create_agent(AgentInput(key="alpha", name="Alpha"))
    return repository


def external_participant(repository):
    room = repository.get_or_create_default_group_chat_room()
    return repository.create_group_chat_participant(
        room.id,
        GroupChatParticipantInput(
            display_name="Dot", kind="external_agent", token_hash="a" * 64
        ),
    )


def test_profile_defaults_then_saves_name_and_ready_pet(repository):
    profile = repository.get_user_profile()
    assert profile.display_name == "You"
    assert profile.pet is None
    assert profile.version == 0

    pet = ready_pet(repository)
    saved = repository.update_user_profile(
        UserProfileUpdate(
            display_name="  Andrew  ",
            pet=PetRef(agent_key="alpha", pet_id=pet.id),
            expected_version=0,
        )
    )
    assert saved.display_name == "Andrew"
    assert saved.pet == PetRef(agent_key="alpha", pet_id=pet.id)
    assert repository.get_user_profile() == saved

    renamed = repository.update_user_profile(
        UserProfileUpdate(display_name="Drew", expected_version=saved.version)
    )
    assert renamed.pet == saved.pet
    cleared = repository.update_user_profile(
        UserProfileUpdate(clear_pet=True, expected_version=renamed.version)
    )
    assert cleared.display_name == "Drew"
    assert cleared.pet is None


def test_profile_rejects_stale_versions_and_unfinished_or_missing_pets(repository):
    with pytest.raises(StorageError, match="Profile changed"):
        repository.update_user_profile(
            UserProfileUpdate(display_name="Drew", expected_version=3)
        )
    unfinished = repository.create_pet("alpha", uuid4(), "An owl")
    with pytest.raises(StorageError, match="finished pets"):
        repository.update_user_profile(
            UserProfileUpdate(
                pet=PetRef(agent_key="alpha", pet_id=unfinished.id), expected_version=0
            )
        )
    with pytest.raises(StorageError, match="Pet not found"):
        repository.update_user_profile(
            UserProfileUpdate(
                pet=PetRef(agent_key="alpha", pet_id=uuid4()), expected_version=0
            )
        )
    with pytest.raises(ValidationError):
        UserProfileUpdate(display_name="   ", expected_version=0)
    with pytest.raises(ValidationError):
        UserProfileUpdate(
            pet=PetRef(agent_key="alpha", pet_id=uuid4()),
            clear_pet=True,
            expected_version=0,
        )


def test_participant_pet_is_assigned_kept_and_cleared(repository):
    participant = external_participant(repository)
    assert participant.pet is None
    pet = ready_pet(repository)

    assigned = repository.update_group_chat_participant(
        participant.room_id,
        participant.id,
        GroupChatParticipantUpdate(
            pet=PetRef(agent_key="alpha", pet_id=pet.id),
            expected_version=participant.version,
        ),
    )
    assert assigned.pet == PetRef(agent_key="alpha", pet_id=pet.id)

    renamed = repository.update_group_chat_participant(
        participant.room_id,
        participant.id,
        GroupChatParticipantUpdate(display_name="Dottie", expected_version=assigned.version),
    )
    assert renamed.pet == assigned.pet

    cleared = repository.update_group_chat_participant(
        participant.room_id,
        participant.id,
        GroupChatParticipantUpdate(clear_pet=True, expected_version=renamed.version),
    )
    assert cleared.pet is None


def test_participant_pet_must_be_finished(repository):
    participant = external_participant(repository)
    unfinished = repository.create_pet("alpha", uuid4(), "An owl")
    with pytest.raises(StorageError, match="finished pets"):
        repository.update_group_chat_participant(
            participant.room_id,
            participant.id,
            GroupChatParticipantUpdate(
                pet=PetRef(agent_key="alpha", pet_id=unfinished.id),
                expected_version=participant.version,
            ),
        )
