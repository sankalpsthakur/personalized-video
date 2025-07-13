"""
Advanced Smart Lip Sync Model Selector
Automatically selects the best available advanced lip sync method including:
- MuseTalk (Real-time high quality)
- LatentSync (Stable Diffusion based)
- VASA-1 (Microsoft expressive)
- EMO (Emotional expressions)  
- Gaussian Splatting (Ultra-fast 3D)
"""

import os
import torch
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
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
    cuda_compute_capability: float = 0.0


@dataclass
class VideoCharacteristics:
    """Video file characteristics"""
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    file_size_mb: float = 0.0
    total_frames: int = 0
    has_face: bool = False
    face_size: float = 0.0


@dataclass
class ProcessingOptions:
    """Processing preferences and constraints"""
    quality_priority: bool = True  # Quality vs Speed
    max_cost_usd: float = 5.0  # Maximum acceptable cost
    max_processing_time_seconds: float = 600  # 10 minutes default
    prefer_local: bool = True  # Prefer local processing
    allow_cloud: bool = True  # Allow cloud processing
    require_real_time: bool = False  # Require real-time capability
    enable_emotions: bool = False  # Enable emotional expressions
    enable_3d: bool = False  # Enable 3D processing
    target_fps: float = 30.0  # Target output FPS


class AdvancedSmartLipSyncSelector:
    """Advanced intelligent lip sync method selector with all state-of-the-art models"""
    
    def __init__(self):
        self.system_caps = self._detect_system_capabilities()
        self.available_methods = self._check_available_methods()
        
        logger.info(f"System capabilities: {self.system_caps}")
        logger.info(f"Available advanced methods: {list(self.available_methods.keys())}")
    
    def _detect_system_capabilities(self) -> SystemCapabilities:
        """Detect system hardware capabilities"""
        caps = SystemCapabilities()
        
        try:
            # Check CUDA
            caps.has_cuda = torch.cuda.is_available()
            if caps.has_cuda:
                device_props = torch.cuda.get_device_properties(0)
                caps.total_vram_gb = device_props.total_memory / (1024**3)
                caps.available_vram_gb = caps.total_vram_gb * 0.8  # Conservative estimate
                caps.cuda_compute_capability = device_props.major + device_props.minor / 10
            
            # Check Apple MPS
            if hasattr(torch.backends, 'mps'):
                caps.has_mps = torch.backends.mps.is_available()
            
            # CPU cores
            caps.cpu_cores = os.cpu_count() or 1
            
        except Exception as e:
            logger.warning(f"Could not detect all system capabilities: {e}")
        
        return caps
    
    def _check_available_methods(self) -> Dict[str, Dict[str, Any]]:
        """Check which advanced lip sync methods are available"""
        methods = {}
        
        # MuseTalk - Real-time high quality
        try:
            from .musetalk_model import musetalk_model
            if musetalk_model.is_available():
                methods["musetalk"] = {
                    "type": "local",
                    "min_vram_gb": 6.0,
                    "recommended_vram_gb": 8.0,
                    "quality_score": 9.2,
                    "speed_multiplier": 0.8,  # 30+ FPS = faster than real-time
                    "cost_per_second": 0.0,
                    "supports_real_time": True,
                    "supports_emotions": False,
                    "supports_3d": False,
                    "resolution": "256x256",
                    "manager": musetalk_model
                }
        except ImportError:
            logger.warning("MuseTalk not available")
        
        # LatentSync - Highest quality using Stable Diffusion
        try:
            from .latentsync_model import latentsync_model
            if latentsync_model.is_available():
                methods["latentsync"] = {
                    "type": "local",
                    "min_vram_gb": 12.0,
                    "recommended_vram_gb": 20.0,
                    "quality_score": 9.8,
                    "speed_multiplier": 2.5,  # 20-24 FPS
                    "cost_per_second": 0.0,
                    "supports_real_time": False,
                    "supports_emotions": False,
                    "supports_3d": False,
                    "resolution": "512x512",
                    "manager": latentsync_model
                }
        except ImportError:
            logger.warning("LatentSync not available")
        
        # VASA-1 - Microsoft's expressive talking face
        try:
            from .vasa1_model import vasa1_model
            if vasa1_model.is_available():
                methods["vasa1"] = {
                    "type": "local", 
                    "min_vram_gb": 12.0,
                    "recommended_vram_gb": 16.0,
                    "quality_score": 9.5,
                    "speed_multiplier": 0.75,  # 40 FPS = 1.33x real-time
                    "cost_per_second": 0.0,
                    "supports_real_time": True,
                    "supports_emotions": True,
                    "supports_3d": False,
                    "resolution": "512x512",
                    "manager": vasa1_model
                }
        except ImportError:
            logger.warning("VASA-1 not available")
        
        # EMO - Expressive portrait animation
        try:
            from .emo_model import emo_model
            if emo_model.is_available():
                methods["emo"] = {
                    "type": "local",
                    "min_vram_gb": 16.0,
                    "recommended_vram_gb": 24.0,
                    "quality_score": 9.7,
                    "speed_multiplier": 1.2,  # 25 FPS
                    "cost_per_second": 0.0,
                    "supports_real_time": False,
                    "supports_emotions": True,
                    "supports_3d": False,
                    "resolution": "512x512",
                    "manager": emo_model
                }
        except ImportError:
            logger.warning("EMO not available")
        
        # Gaussian Splatting - Ultra-fast 3D
        try:
            from .gaussian_splatting_model import gaussian_splatting_model
            if gaussian_splatting_model.is_available():
                methods["gaussian_splatting"] = {
                    "type": "local",
                    "min_vram_gb": 8.0,
                    "recommended_vram_gb": 12.0,
                    "quality_score": 9.0,
                    "speed_multiplier": 0.3,  # 100+ FPS = 3.33x real-time
                    "cost_per_second": 0.0,
                    "supports_real_time": True,
                    "supports_emotions": False,
                    "supports_3d": True,
                    "resolution": "512x512",
                    "manager": gaussian_splatting_model
                }
        except ImportError:
            logger.warning("Gaussian Splatting not available")
        
        # Cloud fallbacks (existing implementations)
        try:
            from .working_lipsync import RealLipSyncSelector
            real_selector = RealLipSyncSelector()
            if real_selector.working_lipsync.available_methods.get('huggingface_wav2lip'):
                methods["huggingface_wav2lip"] = {
                    "type": "cloud",
                    "min_vram_gb": 0.0,
                    "recommended_vram_gb": 0.0,
                    "quality_score": 7.5,
                    "speed_multiplier": 3.0,
                    "cost_per_second": 0.0,
                    "supports_real_time": False,
                    "supports_emotions": False,
                    "supports_3d": False,
                    "resolution": "96x96",
                    "manager": real_selector
                }
        except ImportError:
            logger.warning("Working Wav2Lip not available")
        
        # Replicate cloud API
        try:
            from .replicate_client import replicate_manager
            if replicate_manager.is_available():
                methods["replicate_wav2lip"] = {
                    "type": "cloud",
                    "min_vram_gb": 0.0,
                    "recommended_vram_gb": 0.0,
                    "quality_score": 8.0,
                    "speed_multiplier": 2.0,
                    "cost_per_second": 0.005,
                    "supports_real_time": False,
                    "supports_emotions": False,
                    "supports_3d": False,
                    "resolution": "96x96",
                    "manager": replicate_manager
                }
        except ImportError:
            logger.warning("Replicate client not available")
        
        # Audio-only fallback
        methods["audio_only"] = {
            "type": "fallback",
            "min_vram_gb": 0.0,
            "recommended_vram_gb": 0.0,
            "quality_score": 1.0,
            "speed_multiplier": 0.1,
            "cost_per_second": 0.0,
            "supports_real_time": True,
            "supports_emotions": False,
            "supports_3d": False,
            "resolution": "any",
            "manager": None
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
                    
                    # Simple face detection
                    has_face, face_size = self._detect_face_in_video(video_path)
                    
                    return VideoCharacteristics(
                        duration_seconds=duration,
                        width=width,
                        height=height,
                        fps=fps,
                        file_size_mb=file_size_mb,
                        total_frames=total_frames,
                        has_face=has_face,
                        face_size=face_size
                    )
            
            # Fallback analysis
            file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            return VideoCharacteristics(
                duration_seconds=30.0,
                width=1920,
                height=1080,
                fps=30.0,
                file_size_mb=file_size_mb,
                total_frames=900,
                has_face=True,
                face_size=0.3
            )
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return VideoCharacteristics()
    
    def _detect_face_in_video(self, video_path: str) -> Tuple[bool, float]:
        """Detect if video contains a face and estimate size"""
        try:
            import cv2
            import mediapipe as mp
            
            cap = cv2.VideoCapture(video_path)
            
            # Check a few frames
            face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            
            frames_checked = 0
            faces_detected = 0
            total_face_size = 0.0
            
            while frames_checked < 10:  # Check first 10 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames_checked += 1
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    faces_detected += 1
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    face_area = bbox.width * bbox.height
                    total_face_size += face_area
            
            cap.release()
            
            if faces_detected > 0:
                avg_face_size = total_face_size / faces_detected
                return True, avg_face_size
            
            return False, 0.0
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return True, 0.3  # Assume face present
    
    def select_best_method(self, video_path: str, options: ProcessingOptions) -> str:
        """Select the best lip sync method based on all criteria"""
        
        # Analyze video
        video_chars = self.analyze_video(video_path)
        
        logger.info(f"Video analysis: {video_chars}")
        logger.info(f"Processing options: {options}")
        
        # Score each available method
        method_scores = {}
        
        for method_name, method_info in self.available_methods.items():
            score = self._score_advanced_method(method_info, video_chars, options)
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
    
    def _score_advanced_method(self, method_info: Dict[str, Any], 
                              video_chars: VideoCharacteristics, 
                              options: ProcessingOptions) -> float:
        """Score a method based on advanced criteria"""
        
        score = 0.0
        
        # Base quality score (0-10)
        quality_weight = 15.0 if options.quality_priority else 5.0
        score += method_info["quality_score"] * quality_weight
        
        # Heavy penalty for audio-only when real lip sync is requested
        if options.quality_priority and method_info["type"] == "fallback":
            score -= 80.0
        
        # Bonus for real lip sync methods
        if method_info["type"] != "fallback":
            score += 40.0 if options.quality_priority else 20.0
        
        # Real-time requirement
        if options.require_real_time:
            if method_info.get("supports_real_time", False):
                score += 30.0
            else:
                score -= 50.0
        
        # Emotion support requirement
        if options.enable_emotions:
            if method_info.get("supports_emotions", False):
                score += 25.0
            else:
                score -= 10.0
        
        # 3D processing requirement
        if options.enable_3d:
            if method_info.get("supports_3d", False):
                score += 25.0
            else:
                score -= 10.0
        
        # Speed/FPS considerations
        processing_time = video_chars.duration_seconds * method_info["speed_multiplier"]
        if processing_time <= options.max_processing_time_seconds:
            # Reward faster methods
            speed_score = 20.0 / max(method_info["speed_multiplier"], 0.1)
            speed_weight = 2.0 if options.require_real_time else 1.0
            score += speed_score * speed_weight
        else:
            score -= 40.0  # Heavy penalty for exceeding time limit
        
        # Cost considerations
        estimated_cost = video_chars.duration_seconds * method_info["cost_per_second"]
        if estimated_cost <= options.max_cost_usd:
            cost_score = 15.0 - (estimated_cost / max(options.max_cost_usd, 0.1)) * 10.0
            score += cost_score
        else:
            score -= 50.0  # Heavy penalty for exceeding budget
        
        # Hardware compatibility
        if method_info["type"] == "local":
            required_vram = method_info["min_vram_gb"]
            recommended_vram = method_info["recommended_vram_gb"]
            
            if self.system_caps.has_cuda:
                available_vram = self.system_caps.available_vram_gb
                
                if available_vram >= recommended_vram:
                    score += 25.0  # Perfect hardware match
                elif available_vram >= required_vram:
                    score += 15.0  # Meets minimum requirements
                else:
                    score -= 60.0  # Insufficient VRAM
            elif self.system_caps.has_mps:
                # Apple Silicon can often run smaller models
                if required_vram <= 8.0:
                    score += 10.0
                else:
                    score -= 20.0
            else:
                score -= 70.0  # No GPU acceleration
        
        elif method_info["type"] == "cloud":
            if options.allow_cloud:
                score += 15.0  # Cloud availability bonus
            else:
                score -= 40.0  # Cloud not allowed
        
        # Local preference
        if options.prefer_local and method_info["type"] == "local":
            score += 10.0
        
        # Video characteristics penalties
        if video_chars.has_face:
            score += 10.0  # Bonus for having a face
        else:
            score -= 30.0  # Penalty for no face
        
        # Face size considerations
        if video_chars.face_size < 0.1:  # Very small face
            if method_info["resolution"] == "512x512":
                score += 5.0  # High resolution helps with small faces
        elif video_chars.face_size > 0.5:  # Large face
            score += 5.0  # Any method should work well
        
        # Video resolution and complexity
        pixel_count = video_chars.width * video_chars.height
        if pixel_count > 1920 * 1080:  # High resolution video
            if method_info["resolution"] in ["512x512", "256x256"]:
                score += 5.0  # Good for high-res input
        
        # Duration considerations
        if video_chars.duration_seconds > 300:  # Long video (5+ minutes)
            if method_info["speed_multiplier"] < 1.0:  # Fast processing
                score += 15.0
            elif method_info["speed_multiplier"] > 2.0:  # Slow processing
                score -= 15.0
        
        return max(score, 0.0)  # Ensure non-negative score
    
    def process_video(self, video_path: str, audio_path: str, output_path: str,
                     options: Optional[ProcessingOptions] = None) -> Tuple[bool, str]:
        """Process video with automatically selected advanced method"""
        
        if options is None:
            options = ProcessingOptions()
        
        # Select best method
        selected_method = self.select_best_method(video_path, options)
        
        # Process with selected method
        success = False
        
        if selected_method == "audio_only":
            logger.info("Using audio-only processing")
            success = self._process_audio_only(video_path, audio_path, output_path)
            
        elif selected_method in self.available_methods:
            method_info = self.available_methods[selected_method]
            manager = method_info["manager"]
            
            if manager:
                logger.info(f"Processing with {selected_method}...")
                try:
                    success = manager.process_video(video_path, audio_path, output_path)
                except Exception as e:
                    logger.error(f"{selected_method} processing failed: {e}")
                    success = False
            
            # Fallback chain
            if not success:
                logger.warning(f"{selected_method} failed, trying fallbacks...")
                success = self._try_fallback_methods(video_path, audio_path, output_path, selected_method)
        
        return success, selected_method
    
    def _try_fallback_methods(self, video_path: str, audio_path: str, output_path: str, 
                             failed_method: str) -> bool:
        """Try fallback methods when primary method fails"""
        
        # Define fallback order
        fallback_order = [
            "huggingface_wav2lip",  # Cloud fallback
            "replicate_wav2lip",    # Paid cloud fallback
            "audio_only"            # Final fallback
        ]
        
        # Remove the failed method from fallbacks
        if failed_method in fallback_order:
            fallback_order.remove(failed_method)
        
        for fallback_method in fallback_order:
            if fallback_method in self.available_methods:
                logger.info(f"Trying fallback method: {fallback_method}")
                
                try:
                    method_info = self.available_methods[fallback_method]
                    manager = method_info["manager"]
                    
                    if fallback_method == "audio_only":
                        return self._process_audio_only(video_path, audio_path, output_path)
                    elif manager:
                        success = manager.process_video(video_path, audio_path, output_path)
                        if success:
                            logger.info(f"✅ Fallback {fallback_method} succeeded")
                            return True
                        
                except Exception as e:
                    logger.warning(f"Fallback {fallback_method} failed: {e}")
                    continue
        
        logger.error("All fallback methods failed")
        return False
    
    def _process_audio_only(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Fallback audio-only processing"""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Audio-only processing failed: {e}")
            return False
    
    def get_method_info(self, method_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific method"""
        return self.available_methods.get(method_name)
    
    def list_available_methods(self) -> List[str]:
        """List all available methods"""
        return list(self.available_methods.keys())
    
    def get_recommended_method(self, video_path: str, 
                              requirements: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Get recommended method with detailed reasoning"""
        
        if requirements is None:
            requirements = {}
        
        options = ProcessingOptions(
            quality_priority=requirements.get('quality_priority', True),
            max_cost_usd=requirements.get('max_cost_usd', 5.0),
            max_processing_time_seconds=requirements.get('max_time', 600),
            require_real_time=requirements.get('real_time', False),
            enable_emotions=requirements.get('emotions', False),
            enable_3d=requirements.get('3d', False)
        )
        
        selected_method = self.select_best_method(video_path, options)
        method_info = self.available_methods.get(selected_method, {})
        
        return selected_method, method_info


# Global advanced selector instance
advanced_smart_selector = AdvancedSmartLipSyncSelector()