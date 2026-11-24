"""Synthetic fixtures for deployment-focused tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier

from smartpredict.api.dependencies import load_model_artifacts
from smartpredict.config.settings import Settings
from smartpredict.core.paths import CONFIGS_DIR, PROJECT_ROOT

# A deliberately internal-looking path verifies that the API response sanitizer
# removes filesystem details even when metadata contains them.
_SYNTHETIC_RAW_PATH = r"C:\Users\developer\Customer Churn\data\raw\synthetic.csv"


def write_bundle(
    models_dir: Path,
    preprocessor: ColumnTransformer,
    model: DummyClassifier,
    *,
    feature_count: int | None = None,
    threshold: float = 0.46,
) -> Path:
    """Persist a synthetic, schema-compatible inference bundle."""
    models_dir.mkdir(parents=True, exist_ok=True)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    joblib.dump(model, models_dir / "best_model.joblib")
    joblib.dump(preprocessor, models_dir / "preprocessing_pipeline.joblib")
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": {"name": "synthetic-deployment-fixture", "seed": 42},
        "model": {"selected": "xgboost", "parameters": {}, "random_seed": 42},
        "dataset": {"name": "synthetic", "raw_path": _SYNTHETIC_RAW_PATH},
        "features": {
            "count": feature_count if feature_count is not None else len(feature_names),
            "names": feature_names,
        },
        "decision_threshold": {
            "selected_threshold": threshold,
            "criterion": "synthetic_fixture",
        },
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return models_dir


@pytest.fixture()
def deployment_models_dir(
    tmp_path: Path, prepared_training_data: dict[str, Any]
) -> Path:
    """Create a model bundle from synthetic records only."""
    model = DummyClassifier(strategy="stratified", random_state=42)
    model.fit(
        prepared_training_data["features"].train,
        prepared_training_data["y_train"],
    )
    return write_bundle(
        tmp_path / "artifacts" / "models",
        prepared_training_data["preprocessor"],
        model,
    )


@pytest.fixture()
def deployment_settings(deployment_models_dir: Path) -> Settings:
    """Return settings that point at the synthetic model bundle."""
    return Settings(
        _env_file=None,
        models_dir_override=deployment_models_dir,
        configs_dir=CONFIGS_DIR,
        artifacts_dir=PROJECT_ROOT / "artifacts",
    )


@pytest.fixture()
def deployment_bundle(deployment_models_dir: Path) -> dict[str, Any]:
    """Load the synthetic bundle through the production loader."""
    artifacts = load_model_artifacts(models_dir=deployment_models_dir)
    return {
        "artifacts": artifacts,
        "models_dir": deployment_models_dir,
    }
