"""Optuna optimization and decision-threshold analysis services."""

from smartpredict.optimization.artifacts import (
    build_comparison_report,
    save_optimization_artifacts,
)
from smartpredict.optimization.cross_validation import (
    CrossValidationResult,
    FoldResult,
    cross_validate_params,
)
from smartpredict.optimization.pipeline import (
    OptimizationInputs,
    OptimizationResult,
    run_optimization_pipeline,
    run_optimization_workflow,
)
from smartpredict.optimization.search_space import (
    describe_search_space,
    merge_parameters,
    suggest_parameters,
    validate_search_space,
)
from smartpredict.optimization.study import (
    StudyResult,
    create_or_load_study,
    run_optimization,
    save_study_plots,
)
from smartpredict.optimization.threshold import (
    BASELINE_THRESHOLD,
    ThresholdAnalysisResult,
    ThresholdMetrics,
    analyze_thresholds,
    save_threshold_plots,
)

__all__ = [
    "BASELINE_THRESHOLD",
    "CrossValidationResult",
    "FoldResult",
    "OptimizationInputs",
    "OptimizationResult",
    "StudyResult",
    "ThresholdAnalysisResult",
    "ThresholdMetrics",
    "analyze_thresholds",
    "build_comparison_report",
    "create_or_load_study",
    "cross_validate_params",
    "describe_search_space",
    "merge_parameters",
    "run_optimization",
    "run_optimization_pipeline",
    "run_optimization_workflow",
    "save_optimization_artifacts",
    "save_study_plots",
    "save_threshold_plots",
    "suggest_parameters",
    "validate_search_space",
]
