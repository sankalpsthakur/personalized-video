"""
Smart Lip Sync Model Selector
Automatically selects the best available lip sync method based on:
- Available hardware (GPU/CPU)
- Video characteristics (duration, resolution)
- Quality requirements
- Cost constraints
"""

import os
import torch
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SystemCapabilities:
    """System hardware capabilities"""
    has_cuda: bool = False
    has_mps: bool = False  # Apple Silicon
    total_vram_gb: float = 0.0
    available_vram_gb: float = 0.0
    cpu_cores: int = 0


@dataclass
class VideoCharacteristics:
    """Video file characteristics"""
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    file_size_mb: float = 0.0
    total_frames: int = 0


@dataclass
class ProcessingOptions:
    """Processing preferences and constraints"""
    quality_priority: bool = True  # Quality vs Speed
    max_cost_usd: float = 1.0  # Maximum acceptable cost
    max_processing_time_seconds: float = 300  # 5 minutes default
    prefer_local: bool = True  # Prefer local processing
    fallback_to_audio_only: bool = True


class SmartLipSyncSelector:
    """Intelligent lip sync method selector"""
    
    def __init__(self):
        self.system_caps = self._detect_system_capabilities()
        self.available_methods = self._check_available_methods()
        
        logger.info(f"System capabilities: {self.system_caps}")
        logger.info(f"Available methods: {list(self.available_methods.keys())}")
    
    def _detect_system_capabilities(self) -> SystemCapabilities:
        """Detect system hardware capabilities"""
        caps = SystemCapabilities()
        
        try:
            # Check CUDA
            caps.has_cuda = torch.cuda.is_available()
            if caps.has_cuda:
                caps.total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                caps.available_vram_gb = caps.total_vram_gb  # Approximate
            
            # Check Apple MPS
            if hasattr(torch.backends, 'mps'):
                caps.has_mps = torch.backends.mps.is_available()
            
            # CPU cores
            caps.cpu_cores = os.cpu_count() or 1
            
        except Exception as e:
            logger.warning(f"Could not detect all system capabilities: {e}")
        
        return caps
    
    def _check_available_methods(self) -> Dict[str, Dict[str, Any]]:
        """Check which lip sync methods are available"""
        methods = {}
        
        # Working HuggingFace Wav2Lip (cloud, free)
        try:
            from .working_lipsync import RealLipSyncSelector
            real_selector = RealLipSyncSelector()
            if real_selector.working_lipsync.available_methods.get('huggingface_wav2lip'):
                methods["huggingface_wav2lip"] = {
                    "type": "cloud",
                    "min_vram_gb": 0.0,
                    "quality_score": 8.5,  # Real lip sync, high quality
                    "speed_multiplier": 3.0,  # May be slower due to queue
                    "cost_per_second": 0.0,  # Free
                    "manager": real_selector
                }
        except ImportError:
            logger.warning("Working Wav2Lip not available")
        
        # Replicate (cloud)
        try:
            from .replicate_client import replicate_manager
            if replicate_manager.is_available():
                methods["replicate_wav2lip"] = {
                    "type": "cloud",
                    "min_vram_gb": 0.0,
                    "quality_score": 8.0,
                    "speed_multiplier": 2.0,
                    "cost_per_second": 0.005,  # ~$0.12 for 30s video
                    "manager": replicate_manager
                }
                methods["replicate_sadtalker"] = {
                    "type": "cloud",
                    "min_vram_gb": 0.0,
                    "quality_score": 9.0,
                    "speed_multiplier": 3.0,
                    "cost_per_second": 0.008,  # Higher quality = higher cost
                    "manager": replicate_manager
                }
        except ImportError:
            logger.warning("Replicate client not available")
        
        # Audio-only fallback (always available)
        methods["audio_only"] = {
            "type": "fallback",
            "min_vram_gb": 0.0,
            "quality_score": 1.0,  # No visual sync - should be lowest score
            "speed_multiplier": 0.1,  # Very fast
            "cost_per_second": 0.0,
            "manager": None  # Handled by main pipeline
        }
        
        return methods
    
    def analyze_video(self, video_path: str) -> VideoCharacteristics:
        """Analyze video file characteristics"""
        try:
            # Use ffprobe to get video info
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", 
                "-show_format", "-show_streams", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # Find video stream
                video_stream = None
                for stream in data['streams']:
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    duration = float(data['format'].get('duration', 0))
                    width = int(video_stream.get('width', 0))
                    height = int(video_stream.get('height', 0))
                    
                    # Parse frame rate
                    fps_str = video_stream.get('r_frame_rate', '30/1')
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = float(num) / float(den)
                    else:
                        fps = float(fps_str)
                    
                    file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
                    total_frames = int(duration * fps)
                    
                    return VideoCharacteristics(
                        duration_seconds=duration,
                        width=width,
                        height=height,
                        fps=fps,
                        file_size_mb=file_size_mb,
                        total_frames=total_frames
                    )
            
            # Fallback analysis
            file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            return VideoCharacteristics(
                duration_seconds=30.0,  # Estimate
                width=1920,
                height=1080,
                fps=30.0,
                file_size_mb=file_size_mb,
                total_frames=900
            )
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return VideoCharacteristics()
    
    def select_best_method(self, video_path: str, options: ProcessingOptions) -> str:
        """Select the best lip sync method"""
        
        # Analyze video
        video_chars = self.analyze_video(video_path)
        
        logger.info(f"Video analysis: {video_chars}")
        logger.info(f"Processing options: {options}")
        
        # Score each available method
        method_scores = {}
        
        for method_name, method_info in self.available_methods.items():
            score = self._score_method(method_info, video_chars, options)
            method_scores[method_name] = score
            
            logger.info(f"Method {method_name}: score = {score:.2f}")
        
        # Select best method
        if not method_scores:
            logger.warning("No methods available, falling back to audio-only")
            return "audio_only"
        
        best_method = max(method_scores.items(), key=lambda x: x[1])
        selected_method = best_method[0]
        
        logger.info(f"✅ Selected method: {selected_method} (score: {best_method[1]:.2f})")
        return selected_method
    
    def _score_method(self, method_info: Dict[str, Any], 
                     video_chars: VideoCharacteristics, 
                     options: ProcessingOptions) -> float:
        """Score a method based on various factors"""
        
        score = 0.0
        
        # Quality score (0-10)
        quality_weight = 10.0 if options.quality_priority else 1.0
        score += method_info["quality_score"] * quality_weight
        
        # Heavy penalty for audio-only when lip sync is requested
        if options.quality_priority and method_info["type"] == "fallback":
            score -= 50.0  # Heavy penalty for not doing actual lip sync
        
        # Bonus for real lip sync methods when quality is priority
        if options.quality_priority and method_info["type"] != "fallback":
            score += 30.0  # Bonus for providing actual lip sync
        
        # Speed score (faster = better, but less important when quality_priority=True)
        processing_time = video_chars.duration_seconds * method_info["speed_multiplier"]
        if processing_time <= options.max_processing_time_seconds:
            speed_score = 10.0 / method_info["speed_multiplier"]  # Inverse relationship
            speed_weight = 0.5 if options.quality_priority else 3.0  # Reduce speed importance for quality
            score += speed_score * speed_weight
        else:
            score -= 20.0  # Heavy penalty for exceeding time limit
        
        # Cost score
        estimated_cost = video_chars.duration_seconds * method_info["cost_per_second"]
        if estimated_cost <= options.max_cost_usd:
            cost_score = 10.0 - (estimated_cost / options.max_cost_usd) * 5.0
            score += cost_score * 2.0
        else:
            score -= 30.0  # Heavy penalty for exceeding budget
        
        # Local preference
        if options.prefer_local and method_info["type"] == "local":
            score += 5.0
        elif method_info["type"] == "cloud":
            score += 2.0  # Slight bonus for cloud reliability
        
        # Hardware compatibility
        if method_info["type"] == "local":
            required_vram = method_info["min_vram_gb"]
            if self.system_caps.has_cuda and self.system_caps.available_vram_gb >= required_vram:
                score += 8.0  # Strong bonus for local GPU
            elif self.system_caps.has_mps:
                score += 6.0  # Good for Apple Silicon (can run CPU/MPS models)
            elif required_vram <= 0:
                score += 3.0  # CPU fallback
            else:
                # For Apple Silicon, don't heavily penalize VRAM requirements since MPS can work
                if self.system_caps.has_mps:
                    score -= 5.0  # Light penalty for MPS
                else:
                    score -= 15.0  # Heavy penalty for insufficient hardware
        
        # Video complexity penalty for local methods
        if method_info["type"] == "local":
            complexity_factor = (video_chars.width * video_chars.height) / (1920 * 1080)
            if complexity_factor > 1.5:  # High resolution
                score -= 5.0 * complexity_factor
        
        return max(score, 0.0)  # Ensure non-negative score
    
    def process_video(self, video_path: str, audio_path: str, output_path: str,
                     options: Optional[ProcessingOptions] = None) -> Tuple[bool, str]:
        """Process video with automatically selected method"""
        
        if options is None:
            options = ProcessingOptions()
        
        # Select best method
        selected_method = self.select_best_method(video_path, options)
        
        # Process with selected method
        success = False
        
        if selected_method == "audio_only":
            # Fallback to audio-only processing
            logger.info("Using audio-only processing")
            success = self._process_audio_only(video_path, audio_path, output_path)
            
        elif selected_method in self.available_methods:
            method_info = self.available_methods[selected_method]
            manager = method_info["manager"]
            
            if manager:
                if selected_method.startswith("replicate"):
                    model = "sadtalker" if "sadtalker" in selected_method else "wav2lip"
                    success = manager.process_video(video_path, audio_path, output_path, model)
                elif selected_method == "huggingface_wav2lip":
                    success = manager.process_video(video_path, audio_path, output_path)
                else:
                    success = manager.process_video(video_path, audio_path, output_path)
            
            # Fallback if method fails
            if not success and options.fallback_to_audio_only:
                logger.warning(f"{selected_method} failed, falling back to audio-only")
                success = self._process_audio_only(video_path, audio_path, output_path)
                selected_method = "audio_only"
        
        return success, selected_method
    
    def _process_audio_only(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Fallback audio-only processing"""
        try:
            cmd = [
                "ffmpeg", "-i", video_path, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Audio-only processing failed: {e}")
            return False


# Global selector instance
smart_selector = SmartLipSyncSelector()