"""
AI readiness assessment tools for TrustLens AI.

This package provides components for evaluating whether datasets are
structurally suitable for artificial intelligence and machine learning
workflows, including feature suitability, class imbalance, and data
leakage risk assessment.
"""

from .assessor import AIReadinessAssessor
from .class_imbalance import ClassImbalanceAnalyzer
from .feature_suitability import FeatureSuitabilityAnalyzer

__all__ = [
    "AIReadinessAssessor",
    "ClassImbalanceAnalyzer",
    "FeatureSuitabilityAnalyzer",
]
