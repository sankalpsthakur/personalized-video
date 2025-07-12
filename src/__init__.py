"""
Video Personalization Pipeline
Complete TTS generation with template-based variable replacement
"""

from .pipeline import VideoPersonalizationPipeline
from .templates import TRANSCRIPT_TEMPLATE, DEFAULT_VARIABLES

__all__ = ['VideoPersonalizationPipeline', 'TRANSCRIPT_TEMPLATE', 'DEFAULT_VARIABLES']