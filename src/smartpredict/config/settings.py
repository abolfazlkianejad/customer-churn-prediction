"""Environment-driven application settings."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from smartpredict.core.paths import ARTIFACTS_DIR, CONFIGS_DIR, PROJECT_ROOT


class Environment(StrEnum):
    """Deployment environments supported by the application."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Validated settings for every layer of the system.

    Each field can be overridden by an environment variable prefixed with
    ``SMARTPREDICT_`` (for example ``SMARTPREDICT_LOG_LEVEL=DEBUG``) or by a
    matching entry in a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SMARTPREDICT_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    log_to_file: bool = False
    log_file: Path = ARTIFACTS_DIR / "smartpredict.log"

    random_seed: int = Field(default=42, ge=0)

    project_root: Path = PROJECT_ROOT
    configs_dir: Path = CONFIGS_DIR
    artifacts_dir: Path = ARTIFACTS_DIR
    models_dir_override: Path | None = Field(
        default=None, validation_alias="SMARTPREDICT_MODELS_DIR"
    )
    target_column: str = "Churn"
    id_column: str = "customerID"

    model_name: str = "churn_xgboost"
    model_version: str = "v1"

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_prefix: str = "/api/v1"
    healthcheck_url: str | None = None
    healthcheck_timeout_seconds: float = Field(default=3.0, gt=0, le=60)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize and validate the configured logging level.

        Args:
            value: Raw logging level.

        Returns:
            Uppercase logging level.

        Raises:
            ValueError: If the level is not recognized by :mod:`logging`.

        """
        normalized = value.strip().upper()
        valid = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalized not in valid:
            msg = f"Invalid log level {value!r}; expected one of {sorted(valid)}."
            raise ValueError(msg)
        return normalized

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        """Normalize the API prefix to a leading slash without a trailing one.

        Args:
            value: Raw URL prefix.

        Returns:
            Normalized URL prefix.

        """
        normalized = value.strip()
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized.rstrip("/") or "/"

    @property
    def models_dir(self) -> Path:
        """Directory containing versioned model bundles."""
        return self.models_dir_override or self.artifacts_dir / "models"

    def healthcheck_target(self) -> str:
        """Return the URL probed by container health checks.

        Wildcard bind addresses are translated to loopback because the probe
        always runs inside the serving container. An explicit
        ``healthcheck_url`` overrides the derived value.
        """
        if self.healthcheck_url:
            return self.healthcheck_url
        host = "127.0.0.1" if self.api_host in {"0.0.0.0", "::"} else self.api_host
        prefix = "" if self.api_prefix == "/" else self.api_prefix
        return f"http://{host}:{self.api_port}{prefix}/health"

    @property
    def plots_dir(self) -> Path:
        """Directory containing evaluation figures."""
        return self.artifacts_dir / "plots"

    @property
    def reports_dir(self) -> Path:
        """Directory containing evaluation reports."""
        return self.artifacts_dir / "reports"

    @property
    def explainability_dir(self) -> Path:
        """Directory containing SHAP explanation artifacts."""
        return self.artifacts_dir / "explainability"

    @property
    def optimization_dir(self) -> Path:
        """Directory containing Optuna studies and threshold artifacts."""
        return self.artifacts_dir / "optimization"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance.

    Returns:
        Validated application settings.

    """
    return Settings()
