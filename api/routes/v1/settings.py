from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, model_validator
from api.pydantic_models.settings import ModelSetting, ModelType

router = APIRouter(prefix="/v1/settings", tags=["Settings"])

MOCK_MODEL_SETTINGS = [
    ModelSetting(
        label='Language model',
        type='LLM',
        selected='Qwen3 32B Instruct',
        options=['Qwen3 32B Instruct',
                 'Llama 3.3 70B Instruct',
                 'Mistral Small 3.1 24B',
                 'Gemma 3 27B'],
    ),
    ModelSetting(
        label='Image generation',
        type='Image',
        selected='FLUX.1 [dev]',
        options=['FLUX.1 [dev]',
                 'FLUX.1 Kontext',
                 'SDXL 1.0',
                 'Stable Diffusion 3.5 Large'],
    ),
    ModelSetting(
        label='Video generation',
        type='Video',
        selected='Wan 2.2 T2V 14B',
        options=['Wan 2.2 T2V 14B', 'HunyuanVideo', 'LTX-Video 13B', 'CogVideoX 5B'],
    ),
    ModelSetting(
        label='Text to speech',
        type='TTS',
        selected='F5-TTS',
        options=['F5-TTS', 'XTTS v2', 'Kokoro 82M', 'Parler-TTS Large'],
    ),
    ModelSetting(
        label='Speech to text',
        type='STT',
        selected='Whisper Large v3 Turbo',
        options=['Whisper Large v3',
                 'Whisper Large v3 Turbo',
                 'Whisper Medium',
                 'Distil-Whisper Large v3'],
    ),
]


class SettingsResponse(BaseModel):
    models: list[ModelSetting]
    whisper_formatting: bool


class SettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: dict[ModelType, str]
    whisper_formatting: bool = True

    @model_validator(mode="after")
    def available_models(self) -> "SettingsRequest":
        for setting in MOCK_MODEL_SETTINGS:
            if setting.type in self.models and self.models[setting.type] not in setting.options:
                raise ValueError(f"Unsupported model for {setting.type}")
        return self


@router.get("", operation_id="getSettings")
def get_settings() -> SettingsResponse:
    return SettingsResponse(models=MOCK_MODEL_SETTINGS, whisper_formatting=True)


@router.put("", description="Validates settings and returns the unchanged mock settings. Does not persist changes.", operation_id="updateSettings")
def update_settings(body: SettingsRequest) -> SettingsResponse:
    return get_settings()
