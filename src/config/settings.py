"""Settings from .env via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config in one place."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Model Settings ---
    medasr_model_id: str = "google/medasr"
    whisper_model_size: str = "medium"
    audio_sample_rate: int = 16_000

    # LoRA fine-tuning defaults
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-5

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # experiment tracking
    wandb_api_key: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"

    # --- Hugging Face ---
    hf_token: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
