"""Sign in to Claude or Codex subscriptions and choose which provider serves a capability."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_session
from api.pydantic_models.subscriptions import (
    Capability,
    CapabilityProviderRequest,
    SubscriptionLogin,
    SubscriptionLoginCode,
    SubscriptionProvider,
    SubscriptionsResponse,
)
from api.repositories import subscriptions
from api.services.subscription_login import LoginFailure, logins

router = APIRouter(prefix="/v1/subscriptions", tags=["Subscriptions"])
Database = Annotated[Session, Depends(get_session)]


@router.get("", operation_id="getSubscriptions")
def get_subscriptions(session: Database) -> SubscriptionsResponse:
    return subscriptions.snapshot(session)


@router.put("/capabilities/{capability}", operation_id="chooseCapabilityProvider")
def choose_provider(
    capability: Capability,
    body: CapabilityProviderRequest,
    session: Database,
) -> SubscriptionsResponse:
    try:
        subscriptions.choose(session, capability, body.provider)
    except subscriptions.SubscriptionError as exc:
        raise HTTPException(409, str(exc)) from exc
    return subscriptions.snapshot(session)


@router.post("/{provider}/login", operation_id="startSubscriptionLogin")
async def start_login(provider: SubscriptionProvider) -> SubscriptionLogin:
    return await logins.start(provider)


@router.get("/{provider}/login", operation_id="getSubscriptionLogin")
def get_login(provider: SubscriptionProvider) -> SubscriptionLogin:
    login = logins.status(provider)
    if login is None:
        raise HTTPException(404, "No sign-in has been started.")
    return login


@router.post("/{provider}/login/code", operation_id="submitSubscriptionLoginCode")
async def submit_login_code(
    provider: SubscriptionProvider, body: SubscriptionLoginCode
) -> SubscriptionLogin:
    try:
        return await logins.submit_code(provider, body.code)
    except LoginFailure as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete(
    "/{provider}/login", operation_id="cancelSubscriptionLogin", status_code=204
)
async def cancel_login(provider: SubscriptionProvider) -> None:
    await logins.cancel(provider)


@router.delete("/{provider}", operation_id="disconnectSubscription")
def disconnect_subscription(
    provider: SubscriptionProvider, session: Database
) -> SubscriptionsResponse:
    subscriptions.disconnect(session, provider)
    return subscriptions.snapshot(session)
