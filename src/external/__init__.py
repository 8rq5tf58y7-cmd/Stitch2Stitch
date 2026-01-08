"""
External Pipeline Integrations

Provides wrappers for battle-tested external tools:
- COLMAP: Industry-standard SfM
- HLOC: SuperPoint + SuperGlue + NetVLAD
- AliceVision/Meshroom: Open-source photogrammetry
"""

from .pipelines import (
    COLMAPPipeline,
    HLOCPipeline,
    AliceVisionPipeline,
    check_available_pipelines,
    get_recommended_pipeline
)

__all__ = [
    'COLMAPPipeline',
    'HLOCPipeline', 
    'AliceVisionPipeline',
    'check_available_pipelines',
    'get_recommended_pipeline'
]










