"""Saved subscription sign-ins and which provider serves each capability."""

import json

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from api.database import get_engine
from api.orm_models.subscriptions import CapabilityProvider, Subscription
from api.pydantic_models.subscriptions import (
    SUBSCRIPTION_MODELS,
    ActiveSubscription,
    Capability,
    CapabilityProviders,
    Provider,
    SubscriptionProvider,
    SubscriptionsResponse,
    SubscriptionStatus,
)


class SubscriptionError(ValueError):
    pass


def snapshot(session: Session) -> SubscriptionsResponse:
    connected = set(session.scalars(select(Subscription.provider)))
    providers = {
        row.capability: row.provider
        for row in session.scalars(select(CapabilityProvider))
    }
    return SubscriptionsResponse(
        subscriptions=[
            SubscriptionStatus(
                provider=provider, connected=provider in connected, model=model
            )
            for provider, model in SUBSCRIPTION_MODELS.items()
        ],
        capabilities=CapabilityProviders(**providers),
    )


def validate_credential(provider: SubscriptionProvider, credential: str) -> str:
    credential = credential.strip()
    if provider == "codex":
        try:
            auth = json.loads(credential)
        except ValueError as error:
            raise SubscriptionError(
                "Paste the full contents of ~/.codex/auth.json."
            ) from error
        if not isinstance(auth, dict) or not isinstance(auth.get("tokens"), dict):
            raise SubscriptionError(
                "Codex sign-in did not include a ChatGPT account. Please try again."
            )
    elif any(character.isspace() for character in credential):
        raise SubscriptionError(
            "Claude sign-in returned an unreadable response. Please try again."
        )
    return credential


def connect(session: Session, provider: SubscriptionProvider, credential: str) -> None:
    credential = validate_credential(provider, credential)
    row = session.get(Subscription, provider)
    if row is None:
        session.add(Subscription(provider=provider, credential=credential))
    else:
        row.credential = credential
    session.commit()


def store(provider: SubscriptionProvider, credential: str) -> None:
    with Session(get_engine()) as session:
        connect(session, provider, credential)


def disconnect(session: Session, provider: SubscriptionProvider) -> None:
    """Forget the sign-in; capabilities it served return to their local model."""
    row = session.get(Subscription, provider)
    if row is not None:
        session.delete(row)
    for capability in session.scalars(
        select(CapabilityProvider).where(CapabilityProvider.provider == provider)
    ):
        capability.provider = "local"
    session.commit()


def choose(session: Session, capability: Capability, provider: Provider) -> None:
    if provider != "local" and session.get(Subscription, provider) is None:
        raise SubscriptionError(f"Connect your {provider.title()} subscription first.")
    row = session.get(CapabilityProvider, capability)
    if row is None:
        session.add(CapabilityProvider(capability=capability, provider=provider))
    else:
        row.provider = provider
    session.commit()


def active(capability: Capability) -> ActiveSubscription | None:
    """The subscription serving a capability, or None while it stays local."""
    with Session(get_engine()) as session:
        row = session.get(CapabilityProvider, capability)
        if row is None or row.provider == "local":
            return None
        subscription = session.get(Subscription, row.provider)
        if subscription is None:
            raise SubscriptionError(
                f"The {row.provider.title()} subscription is not connected."
            )
        return ActiveSubscription(
            provider=row.provider, credential=subscription.credential
        )


def refresh(provider: SubscriptionProvider, previous: str, credential: str) -> None:
    """Keep a CLI-refreshed sign-in unless the user replaced or removed it meanwhile."""
    with Session(get_engine()) as session:
        session.execute(
            update(Subscription)
            .where(
                Subscription.provider == provider, Subscription.credential == previous
            )
            .values(credential=credential)
        )
        session.commit()
