#!/usr/bin/env python3
"""
Multi-model lip sync module supporting MuseTalk, Wav2Lip, and LatentSync
"""

import os
import sys
import torch
import numpy as np
import cv2
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
import json
import time
from enum import Enum
from abc import ABC, abstractmethod

# Import model implementations
try:
    from lip_sync_models import (
        Wav2LipInference, MuseTalkInference, LatentSyncInference,
        download_wav2lip_model, create_musetalk_placeholder, setup_latentsync_env
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lip sync model types
class LipSyncModel(Enum):
    MUSETALK = "musetalk"
    WAV2LIP = "wav2lip"
    LATENTSYNC = "latentsync"


class BaseLipSyncModel(ABC):
    """Base class for lip sync models"""
    
    def __init__(self, model_path: Path, device: str):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.inference_engine = None
        logger.info(f"Initializing {self.__class__.__name__} on {device}")
        
    @abstractmethod
    def load_model(self):
        """Load the model weights"""
        pass
        
    @abstractmethod
    def process_frame(self, frame: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        """Process a single frame with lip sync"""
        pass
        
    @abstractmethod
    def check_requirements(self) -> bool:
        """Check if model requirements are met"""
        pass


class MuseTalkModel(BaseLipSyncModel):
    """MuseTalk implementation"""
    
    def __init__(self, model_path: Path, device: str):
        super().__init__(model_path, device)
        self.face_size = 256
        
    def load_model(self):
        """Load MuseTalk model"""
        logger.info("Loading MuseTalk model...")
        
        if not MODELS_AVAILABLE:
            logger.error("Model implementations not available")
            return False
            
        try:
            # Check for required files
            musetalk_path = self.model_path / "musetalk"
            if not (musetalk_path / "musetalk.json").exists():
                logger.warning("MuseTalk model files not found, creating placeholders...")
                create_musetalk_placeholder(self.model_path)
            
            # Initialize inference engine
            self.inference_engine = MuseTalkInference(str(self.model_path), self.device)
            success = self.inference_engine.load_model()
            
            if success:
                logger.info("✓ MuseTalk model loaded successfully")
            else:
                logger.error("✗ Failed to load MuseTalk model")
                
            return success
            
        except Exception as e:
            logger.error(f"Exception loading MuseTalk: {e}")
            return False
            
    def process_frame(self, frame: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        """Process frame with MuseTalk"""
        if self.inference_engine is None:
            logger.warning("MuseTalk inference engine not initialized")
            return frame
            
        start_time = time.time()
        result = self.inference_engine.inference(frame, audio_features)
        inference_time = time.time() - start_time
        logger.debug(f"MuseTalk inference took {inference_time:.3f}s")
        
        return result
        
    def check_requirements(self) -> bool:
        """Check MuseTalk requirements"""
        required_files = [
            self.model_path / "musetalk" / "musetalk.json",
            self.model_path / "whisper" / "tiny.pt"
        ]
        
        missing = []
        for file in required_files:
            if not file.exists():
                missing.append(str(file))
                
        if missing:
            logger.warning(f"Missing MuseTalk files: {missing}")
            
        return len(missing) == 0


class Wav2LipModel(BaseLipSyncModel):
    """Wav2Lip implementation"""
    
    def __init__(self, model_path: Path, device: str):
        super().__init__(model_path, device)
        self.face_size = 96  # Wav2Lip uses 96x96
        
    def load_model(self):
        """Load Wav2Lip model"""
        logger.info("Loading Wav2Lip model...")
        
        if not MODELS_AVAILABLE:
            logger.error("Model implementations not available")
            return False
            
        try:
            # Check if we need to download Wav2Lip
            wav2lip_path = self.model_path / "wav2lip" / "wav2lip_gan.pth"
            if not wav2lip_path.exists():
                logger.warning("Wav2Lip model not found, attempting download...")
                try:
                    download_wav2lip_model(self.model_path)
                except Exception as e:
                    logger.error(f"Failed to download Wav2Lip: {e}")
                    return False
            
            # Initialize inference engine
            self.inference_engine = Wav2LipInference(str(wav2lip_path), self.device)
            success = self.inference_engine.load_model()
            
            if success:
                logger.info("✓ Wav2Lip model loaded successfully")
            else:
                logger.error("✗ Failed to load Wav2Lip model")
                
            return success
            
        except Exception as e:
            logger.error(f"Exception loading Wav2Lip: {e}")
            return False
            
    def process_frame(self, frame: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        """Process frame with Wav2Lip"""
        if self.inference_engine is None:
            logger.warning("Wav2Lip inference engine not initialized")
            return frame
            
        start_time = time.time()
        # Wav2Lip expects list of frames
        results = self.inference_engine.inference([frame], audio_features)
        inference_time = time.time() - start_time
        logger.debug(f"Wav2Lip inference took {inference_time:.3f}s")
        
        return results[0] if results else frame
        
    def check_requirements(self) -> bool:
        """Check Wav2Lip requirements"""
        model_file = self.model_path / "wav2lip" / "wav2lip_gan.pth"
        exists = model_file.exists()
        
        if not exists:
            logger.warning(f"Wav2Lip model not found at {model_file}")
            
        return exists


class LatentSyncModel(BaseLipSyncModel):
    """LatentSync implementation"""
    
    def __init__(self, model_path: Path, device: str):
        super().__init__(model_path, device)
        self.face_size = 512  # LatentSync v1.6 uses 512x512
        
    def load_model(self):
        """Load LatentSync model"""
        logger.info("Loading LatentSync model...")
        
        if not MODELS_AVAILABLE:
            logger.error("Model implementations not available")
            return False
            
        try:
            # Check for model files
            latentsync_path = self.model_path / "latentsync" / "stable_syncnet.pt"
            if not latentsync_path.exists():
                logger.warning("LatentSync model not found, creating placeholders...")
                setup_latentsync_env(self.model_path)
            
            # Check VRAM requirements
            if self.device == "cuda":
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if vram < 20:
                    logger.warning(f"LatentSync requires 20GB+ VRAM, found {vram:.1f}GB")
            
            # Initialize inference engine
            self.inference_engine = LatentSyncInference(str(self.model_path), self.device)
            success = self.inference_engine.load_model()
            
            if success:
                logger.info("✓ LatentSync model loaded successfully")
            else:
                logger.error("✗ Failed to load LatentSync model")
                
            return success
            
        except Exception as e:
            logger.error(f"Exception loading LatentSync: {e}")
            return False
            
    def process_frame(self, frame: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        """Process frame with LatentSync"""
        if self.inference_engine is None:
            logger.warning("LatentSync inference engine not initialized")
            return frame
            
        start_time = time.time()
        result = self.inference_engine.inference(frame, audio_features)
        inference_time = time.time() - start_time
        logger.debug(f"LatentSync inference took {inference_time:.3f}s")
        
        return result
        
    def check_requirements(self) -> bool:
        """Check LatentSync requirements"""
        syncnet_file = self.model_path / "latentsync" / "stable_syncnet.pt"
        exists = syncnet_file.exists()
        
        # Check VRAM
        vram_ok = True
        if self.device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_ok = vram >= 20
            
        if not exists:
            logger.warning(f"LatentSync model not found at {syncnet_file}")
        if not vram_ok:
            logger.warning("Insufficient VRAM for LatentSync")
                
        return exists and vram_ok


class LipSyncProcessor:
    """Multi-model lip synchronization processor"""
    
    # Model configurations
    MODEL_CONFIGS = {
        LipSyncModel.MUSETALK: {
            "class": MuseTalkModel,
            "face_size": 256,
            "fps": 30,
            "quality": "high",
            "vram_required": 6
        },
        LipSyncModel.WAV2LIP: {
            "class": Wav2LipModel,
            "face_size": 96,
            "fps": 25,
            "quality": "medium",
            "vram_required": 4
        },
        LipSyncModel.LATENTSYNC: {
            "class": LatentSyncModel,
            "face_size": 512,
            "fps": 24,
            "quality": "highest",
            "vram_required": 20
        }
    }
    
    def __init__(self, 
                 model_type: Union[str, LipSyncModel] = LipSyncModel.MUSETALK,
                 model_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize lip sync processor with specified model
        
        Args:
            model_type: Type of lip sync model to use
            model_path: Path to model directory
            device: Device to run model on (cuda/cpu)
        """
        # Log initialization
        logger.info(f"Initializing LipSyncProcessor with model: {model_type}, device: {device}")
        
        # Convert string to enum if needed
        if isinstance(model_type, str):
            try:
                model_type = LipSyncModel(model_type.lower())
            except ValueError:
                logger.warning(f"Unknown model type: {model_type}, using MuseTalk")
                model_type = LipSyncModel.MUSETALK
                
        self.model_type = model_type
        self.device = device
        self.model_path = Path(model_path) if model_path else Path("models")
        
        # Create models directory if needed
        self.model_path.mkdir(exist_ok=True)
        
        # Get model configuration
        self.config = self.MODEL_CONFIGS[model_type]
        self.face_size = self.config["face_size"]
        
        # Log system info
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name}, VRAM: {vram:.1f}GB")
        else:
            logger.info("Running on CPU (slower performance expected)")
        
        # Initialize face detection
        self.face_detector = self._init_face_detector()
        
        # Initialize selected model
        self.model = self._init_model()
        
        logger.info(f"LipSyncProcessor ready with {model_type.value}")
        
    def _init_model(self) -> Optional[BaseLipSyncModel]:
        """Initialize the selected lip sync model"""
        logger.info(f"Initializing {self.model_type.value} model...")
        
        model_class = self.config["class"]
        model = model_class(self.model_path, self.device)
        
        # Check requirements
        if not model.check_requirements():
            logger.error(f"{self.model_type.value} requirements not met")
            return None
            
        # Load model
        if model.load_model():
            return model
        else:
            return None
            
    def _init_face_detector(self):
        """Initialize face detection model"""
        logger.info("Initializing face detector...")
        
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            detector = mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            logger.info("✓ Using MediaPipe face detection")
            return detector
        except ImportError:
            logger.warning("MediaPipe not available, falling back to OpenCV")
            # Fallback to OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            detector = cv2.CascadeClassifier(cascade_path)
            logger.info("✓ Using OpenCV Haar Cascade face detection")
            return detector
    
    def get_available_models(self) -> List[Dict[str, any]]:
        """Get list of available models with their status"""
        available = []
        
        logger.info("Checking available models...")
        
        for model_type in LipSyncModel:
            config = self.MODEL_CONFIGS[model_type]
            model_class = config["class"]
            model = model_class(self.model_path, self.device)
            
            info = {
                "name": model_type.value,
                "available": model.check_requirements(),
                "quality": config["quality"],
                "fps": config["fps"],
                "vram_required": config["vram_required"],
                "face_size": config["face_size"]
            }
            
            # Check VRAM if CUDA
            if self.device == "cuda":
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                info["vram_available"] = f"{vram:.1f}GB"
                info["vram_sufficient"] = vram >= config["vram_required"]
            
            available.append(info)
            logger.info(f"  {model_type.value}: {'Available' if info['available'] else 'Not available'}")
            
        return available
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in frame and return bounding box"""
        if hasattr(self.face_detector, 'process'):
            # MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                return (x, y, width, height)
        else:
            # OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            if len(faces) > 0:
                return tuple(faces[0])
        
        return None
    
    def extract_face_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          padding: float = 0.3) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract and resize face region from frame"""
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        # Extract region
        face_region = frame[y1:y2, x1:x2]
        
        # Resize to model input size
        face_resized = cv2.resize(face_region, (self.face_size, self.face_size))
        
        return face_resized, (x1, y1, x2, y2)
    
    def process_video_segment(self, video_path: str, audio_path: str,
                            start_time: float, end_time: float,
                            output_path: str) -> bool:
        """Process a video segment with lip sync"""
        logger.info(f"Processing segment {start_time:.2f}s - {end_time:.2f}s")
        
        if not self.model:
            logger.error(f"{self.model_type.value} model not loaded")
            return False
            
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract video segment
                logger.info("Extracting video segment...")
                segment_video = temp_path / "segment.mp4"
                cmd = [
                    "ffmpeg", "-i", video_path,
                    "-ss", str(start_time), "-to", str(end_time),
                    "-c", "copy", "-y", str(segment_video)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Extract frames
                logger.info("Extracting frames...")
                frames_dir = temp_path / "frames"
                frames_dir.mkdir(exist_ok=True)
                
                cmd = [
                    "ffmpeg", "-i", str(segment_video),
                    "-q:v", "1",
                    str(frames_dir / "frame_%06d.png")
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Get video properties
                cap = cv2.VideoCapture(str(segment_video))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                logger.info(f"Video properties: {width}x{height} @ {fps}fps")
                
                # Process frames with lip sync
                processed_frames_dir = temp_path / "processed"
                processed_frames_dir.mkdir(exist_ok=True)
                
                # Get list of frames
                frame_files = sorted(frames_dir.glob("frame_*.png"))
                logger.info(f"Processing {len(frame_files)} frames...")
                
                # Extract audio features based on model type
                audio_features = None
                if self.model_type == LipSyncModel.WAV2LIP and hasattr(self.model, 'inference_engine'):
                    audio_features = self.model.inference_engine.preprocess_audio(audio_path, fps)
                elif self.model_type == LipSyncModel.MUSETALK and hasattr(self.model, 'inference_engine'):
                    audio_features = self.model.inference_engine.extract_audio_features(audio_path)
                else:
                    # Dummy features
                    audio_features = np.random.randn(len(frame_files), 384)
                
                # Process each frame
                face_bbox = None
                processed_count = 0
                
                for i, frame_file in enumerate(frame_files):
                    frame = cv2.imread(str(frame_file))
                    
                    # Detect face if not already detected
                    if face_bbox is None:
                        face_bbox = self.detect_face(frame)
                        if face_bbox:
                            logger.info(f"Face detected at frame {i}: {face_bbox}")
                    
                    if face_bbox:
                        # Extract face region
                        face_region, coords = self.extract_face_region(frame, face_bbox)
                        
                        # Get appropriate audio features for this frame
                        frame_audio = audio_features[min(i, len(audio_features)-1)] if i < len(audio_features) else audio_features[-1]
                        
                        # Apply lip sync
                        processed_face = self.model.process_frame(face_region, frame_audio)
                        
                        # Paste back to frame
                        x1, y1, x2, y2 = coords
                        processed_face_resized = cv2.resize(processed_face, (x2-x1, y2-y1))
                        frame[y1:y2, x1:x2] = processed_face_resized
                        
                        processed_count += 1
                    
                    # Save processed frame
                    output_file = processed_frames_dir / f"frame_{i+1:06d}.png"
                    cv2.imwrite(str(output_file), frame)
                
                logger.info(f"Processed {processed_count}/{len(frame_files)} frames with lip sync")
                
                # Reassemble video with new audio
                logger.info("Reassembling video...")
                cmd = [
                    "ffmpeg",
                    "-framerate", str(fps),
                    "-i", str(processed_frames_dir / "frame_%06d.png"),
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    "-y", output_path
                ]
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr.decode()}")
                    return False
                
                logger.info(f"✓ Segment processed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error processing video segment: {e}", exc_info=True)
            return False
    
    def apply_lip_sync(self, video_path: str, segments: List[Dict],
                      output_path: str) -> bool:
        """Apply lip sync to video segments"""
        logger.info(f"Applying {self.model_type.value} lip sync to {len(segments)} segments")
        
        start_time = time.time()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Get video info
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                logger.info(f"Video info: {duration:.2f}s @ {fps}fps")
                
                # Process segments
                segment_videos = []
                last_end = 0
                
                for i, segment in enumerate(segments):
                    logger.info(f"Processing segment {i+1}/{len(segments)}")
                    
                    # Keep original before segment
                    if segment["start"] > last_end:
                        before_video = temp_path / f"before_{i}.mp4"
                        logger.info(f"Extracting original video {last_end:.2f}s - {segment['start']:.2f}s")
                        cmd = [
                            "ffmpeg", "-i", video_path,
                            "-ss", str(last_end), "-to", str(segment["start"]),
                            "-c", "copy", "-y", str(before_video)
                        ]
                        subprocess.run(cmd, check=True, capture_output=True)
                        segment_videos.append(str(before_video))
                    
                    # Process segment
                    segment_output = temp_path / f"segment_{i}.mp4"
                    if self.process_video_segment(
                        video_path, segment["audio"],
                        segment["start"], segment["end"],
                        str(segment_output)
                    ):
                        segment_videos.append(str(segment_output))
                    else:
                        logger.warning(f"Failed to process segment {i+1}, using original")
                        # Fallback: copy original segment
                        cmd = [
                            "ffmpeg", "-i", video_path,
                            "-ss", str(segment["start"]), "-to", str(segment["end"]),
                            "-c", "copy", "-y", str(segment_output)
                        ]
                        subprocess.run(cmd, check=True, capture_output=True)
                        segment_videos.append(str(segment_output))
                    
                    last_end = segment["end"]
                
                # Add final segment
                if last_end < duration:
                    final_video = temp_path / "final.mp4"
                    logger.info(f"Extracting final segment {last_end:.2f}s - {duration:.2f}s")
                    cmd = [
                        "ffmpeg", "-i", video_path,
                        "-ss", str(last_end),
                        "-c", "copy", "-y", str(final_video)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    segment_videos.append(str(final_video))
                
                # Concatenate all segments
                logger.info(f"Concatenating {len(segment_videos)} video segments...")
                if len(segment_videos) > 1:
                    concat_file = temp_path / "concat.txt"
                    with open(concat_file, "w") as f:
                        for video in segment_videos:
                            f.write(f"file '{video}'\n")
                    
                    cmd = [
                        "ffmpeg", "-f", "concat", "-safe", "0",
                        "-i", str(concat_file),
                        "-c", "copy", "-y", output_path
                    ]
                    result = subprocess.run(cmd, capture_output=True)
                    
                    if result.returncode != 0:
                        logger.error(f"Concatenation error: {result.stderr.decode()}")
                        return False
                else:
                    shutil.copy(segment_videos[0], output_path)
                
                total_time = time.time() - start_time
                logger.info(f"✓ Lip sync completed in {total_time:.1f}s")
                
                # Log output file info
                if os.path.exists(output_path):
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    logger.info(f"Output file: {output_path} ({size_mb:.1f} MB)")
                
                return True
                
        except Exception as e:
            logger.error(f"Error applying lip sync: {e}", exc_info=True)
            return False


def print_model_comparison():
    """Print comparison of available lip sync models"""
    print("\nLip Sync Model Comparison")
    print("=" * 80)
    print(f"{'Model':<12} {'Quality':<10} {'FPS':<6} {'Face Size':<10} {'VRAM':<8} {'Status':<10}")
    print("-" * 80)
    
    processor = LipSyncProcessor()
    models = processor.get_available_models()
    
    for model in models:
        status = "Ready" if model["available"] else "Missing"
        if model.get("vram_sufficient") is False:
            status = "Low VRAM"
            
        print(f"{model['name']:<12} {model['quality']:<10} {model['fps']:<6} "
              f"{model['face_size']:<10} {model['vram_required']}GB{'':<6} {status:<10}")
    
    if processor.device == "cuda":
        print(f"\nGPU VRAM Available: {models[0].get('vram_available', 'N/A')}")
    else:
        print("\nRunning on CPU (GPU recommended for better performance)")
    
    print("\nRecommendations:")
    print("- MuseTalk: Best balance of quality and performance")
    print("- Wav2Lip: Fastest, lower quality, good for real-time")
    print("- LatentSync: Highest quality, requires high-end GPU")
    print()


if __name__ == "__main__":
    # Show model comparison
    print_model_comparison()
    
    # Test with different models
    print("\nTesting model initialization...")
    for model_type in ["musetalk", "wav2lip", "latentsync"]:
        print(f"\nTesting {model_type}...")
        processor = LipSyncProcessor(model_type=model_type)
        
        # Test face detection
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = processor.detect_face(test_frame)
        
        if bbox:
            print(f"✓ Face detection working")
        else:
            print(f"✗ No face detected in test frame")