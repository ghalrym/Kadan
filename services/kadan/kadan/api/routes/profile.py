"""The user's profile: the name and pet shown on their Group Chat messages."""

from fastapi import APIRouter, Request

from ..pydantic_models.profile import UserProfile, UserProfileUpdate

router = APIRouter(prefix="/api/profile")


@router.get("", response_model=UserProfile)
async def get_profile(request: Request) -> UserProfile:
    data = await request.app.state.services.json_request("Tsuchi", "GET", "/api/profile")
    return UserProfile.model_validate(data)


@router.put("", response_model=UserProfile)
async def update_profile(request: Request, value: UserProfileUpdate) -> UserProfile:
    data = await request.app.state.services.json_request(
        "Tsuchi", "PUT", "/api/profile", json=value.model_dump(mode="json", exclude_none=True)
    )
    return UserProfile.model_validate(data)
