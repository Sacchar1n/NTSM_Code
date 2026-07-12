"""
NTSM Model Package

Neurophysiology-Guided Tri-Stream Selective State Space Model
for Real-Time Driver Fatigue Detection with EEG-EOG Signals.
"""

from .ntsm import NTSM
from .sru import SpectralRecalibrationUnit
from .ad_gate import ArtifactDisentanglementGate
from .fc_bridge import FunctionalConnectivityBridge
from .oga import OcularGuidedAttention
from .ssm_encoder import SSMUnit, BidirectionalSSMEncoder
from .bilinear_encoder import BilinearOcularEncoder
from .coherence_extractor import CoherenceExtractor

__all__ = [
    'NTSM',
    'SpectralRecalibrationUnit',
    'ArtifactDisentanglementGate',
    'FunctionalConnectivityBridge',
    'OcularGuidedAttention',
    'SSMUnit',
    'BidirectionalSSMEncoder',
    'BilinearOcularEncoder',
    'CoherenceExtractor',
]
