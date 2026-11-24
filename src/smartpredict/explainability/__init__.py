"""Post-hoc model explainability for the selected churn model."""

from smartpredict.explainability.service import (
    ExplanationArtifacts,
    generate_explanations,
)

__all__ = ["ExplanationArtifacts", "generate_explanations"]
