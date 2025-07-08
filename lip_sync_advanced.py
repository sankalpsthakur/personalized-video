#!/usr/bin/env python3
"""
Advanced lip sync integration module
Bridges existing pipeline with state-of-the-art models
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import subprocess
import json

# Import base lip sync module
try:
    from lip_sync import LipSyncProcessor, BaseLipSyncModel
except ImportError:
    # Define minimal interface for standalone use
    class BaseLipSyncModel:
        def apply_lip_sync(self, video_path: str, audio_segments: List[Dict], 
                          output_path: str) -> bool:
            raise NotImplementedError
    
    class LipSyncProcessor:
        pass

# Import advanced models
from advanced_models import (
    AdvancedModelManager, VASA1Model, EMOModel, 
    GaussianSplattingModel, AdvancedLipSyncModel
)

logger = logging.getLogger(__name__)


class AdvancedLipSyncWrapper(BaseLipSyncModel):
    """Wrapper to make advanced models compatible with existing pipeline"""
    
    def __init__(self, model: AdvancedLipSyncModel):
        self.model = model
        self.name = model.model_name
    
    def load_model(self):
        """Load the wrapped model"""
        return self.model.load_model()
    
    def check_requirements(self) -> Dict[str, any]:
        """Check model requirements"""
        return self.model.get_requirements()
    
    def process_frame(self, frame, audio_chunk):
        """Process a single frame (not used in advanced models)"""
        # Advanced models process entire videos, not frames
        return frame
        
    def apply_lip_sync(self, video_path: str, audio_segments: List[Dict], 
                      output_path: str) -> bool:
        """Apply lip sync using advanced model"""
        try:
            # For advanced models, we need to create a full audio track
            # from the segments
            if audio_segments:
                # Merge audio segments
                audio_path = self._merge_audio_segments(audio_segments)
            else:
                # Extract audio from original video
                audio_path = self._extract_audio(video_path)
            
            # Process with advanced model
            success = self.model.process_video(video_path, audio_path, output_path)
            
            # Cleanup temp audio
            if os.path.exists(audio_path) and audio_path.startswith("/tmp"):
                os.unlink(audio_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Advanced lip sync failed: {e}")
            return False
    
    def _merge_audio_segments(self, segments: List[Dict]) -> str:
        """Merge audio segments into a single file"""
        import tempfile
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        merged_path = os.path.join(temp_dir, "merged_audio.wav")
        
        # If only one segment, just use it
        if len(segments) == 1:
            return segments[0]["audio"]
        
        # Create concat file
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for seg in segments:
                f.write(f"file '{seg['audio']}'\n")
        
        # Merge with ffmpeg
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c", "copy",
            "-y", merged_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return merged_path
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video"""
        import tempfile
        
        audio_path = tempfile.mktemp(suffix=".wav")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ac", "1", "-ar", "16000",
            "-y", audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return audio_path


class ExtendedLipSyncProcessor(LipSyncProcessor):
    """Extended processor supporting both classic and advanced models"""
    
    def __init__(self, model_type: str = "musetalk"):
        # Check if it's an advanced model
        advanced_models = ["vasa1", "emo", "gaussian_splatting", "gstalker"]
        
        if model_type in advanced_models:
            # Initialize advanced model
            manager = AdvancedModelManager()
            advanced_model = manager.get_model(model_type)
            self.model = AdvancedLipSyncWrapper(advanced_model)
            self.model_type = model_type
            self.is_advanced = True
            logger.info(f"Initialized advanced model: {model_type}")
        else:
            # Fall back to classic models
            try:
                super().__init__(model_type)
                self.is_advanced = False
            except:
                # If parent class not available, create minimal version
                self.model_type = model_type
                self.model = None
                self.is_advanced = False
                logger.warning(f"Classic model {model_type} not available")
    
    def apply_lip_sync_simple(self, video_path: str, segments: List[Dict], 
                             output_path: str) -> bool:
        """Apply lip sync with automatic model selection"""
        if self.is_advanced:
            return self.model.apply_lip_sync(video_path, segments, output_path)
        elif self.model:
            # Use parent class method
            return super().apply_lip_sync_simple(video_path, segments, output_path)
        else:
            logger.error("No model available")
            return False
    
    @staticmethod
    def list_all_models() -> Dict[str, List[Dict]]:
        """List all available models (classic + advanced)"""
        models = {
            "classic": [],
            "advanced": []
        }
        
        # Classic models
        classic_models = [
            {
                "name": "musetalk",
                "type": "VAE-based",
                "resolution": "256x256",
                "fps": 30,
                "vram": 6,
                "status": "available"
            },
            {
                "name": "wav2lip",
                "type": "GAN-based",
                "resolution": "96x96",
                "fps": 25,
                "vram": 4,
                "status": "available"
            },
            {
                "name": "latentsync",
                "type": "Diffusion-based",
                "resolution": "512x512",
                "fps": 24,
                "vram": 20,
                "status": "available"
            }
        ]
        models["classic"] = classic_models
        
        # Advanced models
        manager = AdvancedModelManager()
        for model_info in manager.list_models():
            req = model_info["requirements"]
            models["advanced"].append({
                "name": model_info["name"],
                "type": model_info["class"],
                "resolution": f"{req['resolution'][0]}x{req['resolution'][1]}",
                "fps": req["fps"],
                "vram": req["vram"],
                "features": req["features"],
                "status": "simulation"  # Change to "available" when models are downloaded
            })
        
        return models
    
    @staticmethod
    def print_extended_comparison():
        """Print comparison of all models"""
        models = ExtendedLipSyncProcessor.list_all_models()
        
        print("\n" + "="*80)
        print("ALL AVAILABLE LIP SYNC MODELS")
        print("="*80)
        
        # Classic models
        print("\n## Classic Models (Production-Ready)")
        print("| Model | Type | Resolution | FPS | VRAM | Status |")
        print("|-------|------|------------|-----|------|--------|")
        
        for model in models["classic"]:
            print(f"| {model['name']} | {model['type']} | {model['resolution']} | "
                  f"{model['fps']} | {model['vram']}GB | {model['status']} |")
        
        # Advanced models
        print("\n## Advanced Models (State-of-the-Art)")
        print("| Model | Type | Resolution | FPS | VRAM | Features | Status |")
        print("|-------|------|------------|-----|------|----------|--------|")
        
        for model in models["advanced"]:
            features = ", ".join(model['features'][:2])
            print(f"| {model['name']} | {model['type']} | {model['resolution']} | "
                  f"{model['fps']} | {model['vram']}GB | {features} | {model['status']} |")
        
        print("\n" + "="*80)
        
        # Recommendations
        print("\n## Recommendations:")
        print("- **Real-time**: gaussian_splatting (100+ FPS)")
        print("- **Quality**: vasa1 or emo (512x512 with expressions)")
        print("- **Balanced**: musetalk (production-ready)")
        print("- **Low VRAM**: wav2lip (4GB)")
        print()


def update_pipeline_for_advanced_models():
    """Update the main pipeline to support advanced models"""
    pipeline_path = Path("personalization_pipeline.py")
    
    if not pipeline_path.exists():
        logger.error("Pipeline file not found")
        return False
    
    # Read current pipeline
    with open(pipeline_path, "r") as f:
        content = f.read()
    
    # Check if already updated
    if "lip_sync_advanced" in content:
        logger.info("Pipeline already updated for advanced models")
        return True
    
    # Create backup
    backup_path = pipeline_path.with_suffix(".py.bak")
    with open(backup_path, "w") as f:
        f.write(content)
    
    # Add import for advanced models
    import_section = """try:
    import lip_sync
    LIPSYNC_AVAILABLE = True
except ImportError:
    LIPSYNC_AVAILABLE = False
    print("Warning: LipSync module not available. Lip sync will be disabled.")

try:
    from lip_sync_advanced import ExtendedLipSyncProcessor
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("Info: Advanced models not available. Using classic models only.")"""
    
    # Replace import section
    content = content.replace(
        """try:
    import lip_sync
    LIPSYNC_AVAILABLE = True
except ImportError:
    LIPSYNC_AVAILABLE = False
    print("Warning: LipSync module not available. Lip sync will be disabled.")""",
        import_section
    )
    
    # Update model initialization
    init_update = """if self.enable_lip_sync:
            try:
                if ADVANCED_MODELS_AVAILABLE:
                    from lip_sync_advanced import ExtendedLipSyncProcessor
                    logger.info(f"Initializing lip sync with {lip_sync_model} model (extended)...")
                    self.lip_sync_processor = ExtendedLipSyncProcessor(model_type=lip_sync_model)
                else:
                    from lip_sync import LipSyncProcessor
                    logger.info(f"Initializing lip sync with {lip_sync_model} model...")
                    self.lip_sync_processor = LipSyncProcessor(model_type=lip_sync_model)
                logger.info(f"✓ Lip sync enabled with {lip_sync_model} model")
            except Exception as e:
                logger.error(f"Failed to initialize lip sync: {e}")
                self.enable_lip_sync = False"""
    
    # Find and replace the initialization code
    import re
    pattern = r"if self\.enable_lip_sync:.*?self\.enable_lip_sync = False"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        content = content.replace(match.group(0), init_update)
    
    # Update model choices in argparse
    content = content.replace(
        'choices=["musetalk", "wav2lip", "latentsync"]',
        'choices=["musetalk", "wav2lip", "latentsync", "vasa1", "emo", "gaussian_splatting"]'
    )
    
    # Update list models functionality
    list_models_update = """if args.list_models:
        if ADVANCED_MODELS_AVAILABLE:
            from lip_sync_advanced import ExtendedLipSyncProcessor
            ExtendedLipSyncProcessor.print_extended_comparison()
        elif LIPSYNC_AVAILABLE:
            from lip_sync import print_model_comparison
            print_model_comparison()
        else:
            print("Lip sync module not available. Install dependencies with:")
            print("pip install -r requirements.txt")
        return"""
    
    # Replace list models section
    pattern = r"if args\.list_models:.*?return"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        content = content.replace(match.group(0), list_models_update)
    
    # Write updated pipeline
    with open(pipeline_path, "w") as f:
        f.write(content)
    
    logger.info("Pipeline updated successfully for advanced models")
    return True


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test extended processor
    print("Testing Extended Lip Sync Processor")
    print("="*80)
    
    # List all models
    ExtendedLipSyncProcessor.print_extended_comparison()
    
    # Test each advanced model
    test_models = ["vasa1", "emo", "gaussian_splatting"]
    
    for model_name in test_models:
        print(f"\nTesting {model_name}...")
        try:
            processor = ExtendedLipSyncProcessor(model_name)
            print(f"✓ {model_name} initialized successfully")
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
    
    # Update pipeline
    print("\nUpdating main pipeline...")
    if update_pipeline_for_advanced_models():
        print("✓ Pipeline updated for advanced models")
    else:
        print("✗ Failed to update pipeline")