"""
Intelligent lip sync processing with multiple methods
"""

from .smart_selector import smart_selector, ProcessingOptions
from .easy_wav2lip import wav2lip_manager
from .replicate_client import replicate_manager

__all__ = [
    'smart_selector', 
    'ProcessingOptions',
    'wav2lip_manager', 
    'replicate_manager'
]