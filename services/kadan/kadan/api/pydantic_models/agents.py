from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from .resources import AgentKey, Title


class Agent(BaseModel):
    key: AgentKey
    name: Title
    created_at: datetime
    avatar_asset_id: UUID | None = None
