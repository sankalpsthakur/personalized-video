"""
Veo3 Production Pipeline
Professional video generation using FLUX Kontext → Veo 3 → ElevenLabs
"""

from .veo3_pipeline import (
    Veo3Pipeline,
    ProjectMetadata,
    KontextConfig,
    Veo3Config,
    ElevenLabsConfig,
    QualityMetrics
)

from .api_clients import (
    FluxKontextClient,
    Veo3Client,
    ElevenLabsClient,
    APIClientFactory
)

from .character_consistency import (
    CharacterProfile,
    CharacterConsistencyManager
)

from .post_production import (
    PostProductionPipeline,
    ColorGradeProfile,
    ExportPreset,
    QualityControl
)

__version__ = "1.0.0"
__all__ = [
    # Main pipeline
    "Veo3Pipeline",
    "ProjectMetadata",
    "KontextConfig",
    "Veo3Config", 
    "ElevenLabsConfig",
    "QualityMetrics",
    
    # API clients
    "FluxKontextClient",
    "Veo3Client",
    "ElevenLabsClient",
    "APIClientFactory",
    
    # Character consistency
    "CharacterProfile",
    "CharacterConsistencyManager",
    
    # Post-production
    "PostProductionPipeline",
    "ColorGradeProfile",
    "ExportPreset",
    "QualityControl"
]