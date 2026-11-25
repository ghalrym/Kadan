"""Bring-your-own Claude and Codex subscriptions that can replace a local capability."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SubscriptionProvider = Literal["claude", "codex"]
Provider = Literal["local", "claude", "codex"]
Capability = Literal["language", "image"]

SUBSCRIPTION_MODELS: dict[SubscriptionProvider, str] = {
    "claude": "claude-haiku-5-5",
    "codex": "gpt-6.1-terra",
}


class SubscriptionStatus(BaseModel):
    provider: SubscriptionProvider
    connected: bool
    model: str


class CapabilityProviders(BaseModel):
    """Capabilities a subscription may serve; each defaults to its local model."""

    language: Provider = "local"
    image: Literal["local", "codex"] = "local"


class SubscriptionsResponse(BaseModel):
    subscriptions: list[SubscriptionStatus]
    capabilities: CapabilityProviders


class SubscriptionLogin(BaseModel):
    """A web sign-in in progress; `url` opens the provider's sign-in page."""

    provider: SubscriptionProvider
    status: Literal["starting", "waiting", "verifying", "complete", "failed"]
    url: str | None = None
    user_code: str | None = Field(
        default=None,
        description="Codex: the one-time code to enter on its sign-in page.",
    )
    needs_code: bool = Field(
        description="Claude: its sign-in page shows a code to send back to Hikari."
    )
    error: str | None = None


class SubscriptionLoginCode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=1, max_length=4096)


class CapabilityProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Provider


class ActiveSubscription(BaseModel):
    provider: SubscriptionProvider
    credential: str
