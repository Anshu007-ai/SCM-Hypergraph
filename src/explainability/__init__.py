"""
Explainability module for HT-HGNN v2.0

Provides SHAP-based and gradient-based explainability tools for
hypergraph neural network predictions on supply chain data.
"""

from .hypershap import HyperSHAP, NodeExplanation
from .hyperedge_importance import HyperedgeImportanceAnalyzer
from .feature_attribution import FeatureAttributionAnalyzer

__all__ = [
    'HyperSHAP',
    'NodeExplanation',
    'HyperedgeImportanceAnalyzer',
    'FeatureAttributionAnalyzer',
]
