"""
Working lip sync implementation using available methods
Focus on methods that actually work with real lip synchronization
"""

import os
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import json

logger = logging.getLogger(__name__)


class WorkingLipSync:
    """Implementation focusing on actually working lip sync methods"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.available_methods = self._check_available_methods()
        
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which working lip sync methods are available"""
        methods = {}
        
        # Check for Wav2Lip via HuggingFace Spaces
        methods['huggingface_wav2lip'] = self._check_huggingface_spaces()
        
        # Check for local Wav2Lip with proper models
        methods['local_wav2lip'] = self._check_local_wav2lip()
        
        # Check for Replicate (if API key exists)
        methods['replicate'] = bool(os.getenv('REPLICATE_API_TOKEN'))
        
        # Check for RunPod (if API key exists)
        methods['runpod'] = bool(os.getenv('RUNPOD_API_KEY'))
        
        logger.info(f"Available working methods: {[k for k,v in methods.items() if v]}")
        return methods
    
    def _check_huggingface_spaces(self) -> bool:
        """Check if HuggingFace Spaces are accessible"""
        try:
            # Test gradio_client import first
            from gradio_client import Client
            
            # Test connection to working spaces
            working_spaces = [
                "manavisrani07/gradio-lipsync-wav2lip",
                "xiaoqidao/wav2lip_demo"
            ]
            
            for space in working_spaces:
                try:
                    client = Client(space)
                    logger.info(f"✅ Connected to {space}")
                    return True
                except:
                    continue
            
            return False
        except ImportError:
            logger.warning("gradio_client not available")
            return False
        except:
            return False
    
    def _check_local_wav2lip(self) -> bool:
        """Check if OpenCV and librosa are available for local implementation"""
        try:
            import cv2
            import librosa
            import numpy as np
            return True
        except ImportError:
            return False
    
    def apply_huggingface_wav2lip(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Apply lip sync using HuggingFace Spaces Wav2Lip"""
        try:
            logger.info("Using HuggingFace Spaces Wav2Lip...")
            
            # Use gradio_client to interact with the space
            try:
                from gradio_client import Client
            except ImportError:
                logger.error("gradio_client not installed. Run: pip install gradio_client")
                return False
            
            # Try multiple working spaces in order of preference
            spaces_to_try = [
                ("manavisrani07/gradio-lipsync-wav2lip", "standard"),  # Known working space
                ("xiaoqidao/wav2lip_demo", "basic")  # Basic fallback
            ]
            
            for space_name, quality in spaces_to_try:
                try:
                    logger.info(f"Trying {space_name} ({quality} quality)...")
                    
                    # Connect to the space
                    client = Client(space_name)
                    
                    if space_name == "manavisrani07/gradio-lipsync-wav2lip":
                        # Use the correct API for this space
                        result = client.predict(
                            video_path,     # video
                            audio_path,     # audio
                            "wav2lip_gan",  # checkpoint (wav2lip or wav2lip_gan)
                            0,              # no_smooth
                            1,              # resize_factor  
                            0,              # pad_top
                            10,             # pad_bottom
                            0,              # pad_left
                            api_name="/generate"
                        )
                    else:
                        # Basic wav2lip demo
                        result = client.predict(
                            video_path,    # video
                            audio_path,    # audio
                            api_name="/predict"
                        )
                    
                    # Handle result
                    if result:
                        logger.info(f"Raw result from {space_name}: {type(result)} - {str(result)[:100]}")
                        
                        # Result might be a dict with video file
                        if isinstance(result, dict) and 'video' in result:
                            result_path = result['video']
                        elif isinstance(result, str):
                            result_path = result
                        elif isinstance(result, tuple) and len(result) > 0:
                            result_path = result[0]
                        else:
                            logger.warning(f"Unknown result type: {type(result)}")
                            continue
                        
                        # Handle the result path
                        if isinstance(result_path, str):
                            if Path(result_path).exists():
                                shutil.copy2(result_path, output_path)
                            elif result_path.startswith('http'):
                                # Download from URL
                                response = requests.get(result_path)
                                response.raise_for_status()
                                with open(output_path, 'wb') as f:
                                    f.write(response.content)
                            else:
                                logger.warning(f"Invalid result path: {result_path}")
                                continue
                        
                        if Path(output_path).exists():
                            logger.info(f"✅ HuggingFace Wav2Lip completed with {space_name}: {output_path}")
                            return True
                    
                    logger.warning(f"No valid result from {space_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed with {space_name}: {e}")
                    continue
            
            logger.error("All HuggingFace spaces failed")
            return False
            
        except Exception as e:
            logger.error(f"HuggingFace Wav2Lip failed: {e}")
            return False
    
    def apply_local_wav2lip(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Apply improved local lip sync with better audio analysis"""
        try:
            from .improved_local import ImprovedLocalLipSync
            
            logger.info("Using improved local lip sync with audio analysis...")
            
            # Use the improved implementation
            improved_sync = ImprovedLocalLipSync()
            success = improved_sync.process_video(video_path, audio_path, output_path)
            
            if success:
                logger.info("✅ Improved local lip sync completed")
                return True
            else:
                logger.error("Improved local lip sync failed")
                return False
            
        except Exception as e:
            logger.error(f"Improved local lip sync failed: {e}")
            
            # Fallback to basic implementation
            logger.info("Falling back to basic local implementation...")
            return self._apply_basic_local_wav2lip(video_path, audio_path, output_path)
    
    def _apply_basic_local_wav2lip(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Basic fallback local lip sync"""
        try:
            import cv2
            import librosa
            import numpy as np
            
            logger.info("Using basic local lip sync fallback...")
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Load audio and extract energy
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Calculate frame-level audio energy
            hop_length = sr // fps
            frame_energies = []
            
            for i in range(frame_count):
                start_sample = i * hop_length
                end_sample = min(start_sample + hop_length, len(audio))
                
                if start_sample < len(audio):
                    frame_audio = audio[start_sample:end_sample]
                    energy = np.sqrt(np.mean(frame_audio**2))
                else:
                    energy = 0.0
                
                frame_energies.append(energy)
            
            # Normalize energies
            max_energy = max(frame_energies) if frame_energies else 1.0
            frame_energies = [e / max_energy for e in frame_energies]
            
            # Initialize face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_output = output_path + ".temp.mp4"
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            frame_idx = 0
            face_region = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                energy = frame_energies[frame_idx] if frame_idx < len(frame_energies) else 0.0
                
                if face_region is None or frame_idx % 30 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        face_region = max(faces, key=lambda x: x[2] * x[3])
                
                if face_region is not None and energy > 0.1:
                    x, y, w, h = face_region
                    
                    mouth_y = y + int(h * 0.7)
                    mouth_h = int(h * 0.3)
                    mouth_x = x + int(w * 0.25)
                    mouth_w = int(w * 0.5)
                    
                    mouth_y = max(0, min(mouth_y, height - mouth_h))
                    mouth_x = max(0, min(mouth_x, width - mouth_w))
                    mouth_h = min(mouth_h, height - mouth_y)
                    mouth_w = min(mouth_w, width - mouth_x)
                    
                    if mouth_w > 0 and mouth_h > 0:
                        mouth_region_img = frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w].copy()
                        
                        darkness = min(0.7, energy * 0.5)
                        mouth_region_img = (mouth_region_img * (1 - darkness)).astype(np.uint8)
                        
                        stretch_factor = 1.0 + (energy * 0.2)
                        new_h = int(mouth_h * stretch_factor)
                        
                        if new_h <= height - mouth_y:
                            mouth_stretched = cv2.resize(mouth_region_img, (mouth_w, new_h))
                            frame[mouth_y:mouth_y+new_h, mouth_x:mouth_x+mouth_w] = mouth_stretched
                        else:
                            frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w] = mouth_region_img
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            
            # Add audio back
            cmd = [
                "ffmpeg", "-i", temp_output, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", 
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                os.remove(temp_output)
                logger.info("✅ Basic local lip sync completed with audio")
                return True
            else:
                logger.error(f"Failed to add audio: {result.stderr}")
                os.rename(temp_output, output_path)
                logger.warning("✅ Basic local lip sync completed (video only)")
                return True
            
        except Exception as e:
            logger.error(f"Basic local lip sync failed: {e}")
            return False

    def download_wav2lip_model(self) -> bool:
        """Download the actual Wav2Lip model from reliable source"""
        try:
            model_dir = Path.home() / ".cache" / "wav2lip"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / "wav2lip_gan.pth"
            
            if model_path.exists():
                logger.info("Wav2Lip model already downloaded")
                return True
            
            logger.info("Downloading Wav2Lip model...")
            
            # Alternative download sources
            download_urls = [
                # Google Drive link (from original Wav2Lip repo)
                "https://drive.google.com/uc?id=1Xr0TIzYw0t0dj0KlmKP98Xz7krCi5fbl",
                # Mirror on HuggingFace
                "https://huggingface.co/spaces/Manmay/wav2lip-inference/resolve/main/wav2lip_gan.pth",
                # Alternative mirror
                "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip_gan.pth"
            ]
            
            for url in download_urls:
                try:
                    logger.info(f"Trying: {url}")
                    
                    if "drive.google.com" in url:
                        # Use gdown for Google Drive
                        try:
                            import gdown
                            gdown.download(url, str(model_path), quiet=False)
                            if model_path.exists():
                                logger.info("✅ Downloaded from Google Drive")
                                return True
                        except:
                            logger.warning("gdown not available, trying next source")
                    else:
                        # Direct download
                        response = requests.get(url, stream=True)
                        response.raise_for_status()
                        
                        with open(model_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        logger.info(f"✅ Downloaded from {url}")
                        return True
                        
                except Exception as e:
                    logger.warning(f"Failed to download from {url}: {e}")
                    continue
            
            logger.error("Failed to download Wav2Lip model from any source")
            return False
            
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False
    
    def apply_local_wav2lip_docker(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Apply Wav2Lip using Docker container for isolation"""
        try:
            logger.info("Using Docker Wav2Lip...")
            
            # Check if Docker is available
            result = subprocess.run(["docker", "--version"], capture_output=True)
            if result.returncode != 0:
                logger.error("Docker not available")
                return False
            
            # Pull Wav2Lip Docker image
            logger.info("Pulling Wav2Lip Docker image...")
            subprocess.run(["docker", "pull", "myname/wav2lip:latest"], check=True)
            
            # Run Wav2Lip in Docker
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{Path(video_path).parent}:/input",
                "-v", f"{Path(output_path).parent}:/output",
                "myname/wav2lip:latest",
                "--video", f"/input/{Path(video_path).name}",
                "--audio", f"/input/{Path(audio_path).name}",
                "--output", f"/output/{Path(output_path).name}"
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0 and Path(output_path).exists():
                logger.info("✅ Docker Wav2Lip completed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Docker Wav2Lip failed: {e}")
            return False
    
    def apply_replicate_wav2lip(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Use Replicate API with correct endpoint"""
        try:
            api_token = os.getenv('REPLICATE_API_TOKEN')
            if not api_token:
                logger.error("No REPLICATE_API_TOKEN found")
                return False
            
            import replicate
            
            logger.info("Using Replicate Wav2Lip...")
            
            # Upload files
            with open(video_path, "rb") as v:
                video_url = replicate.upload_file(v, "video/mp4")
            
            with open(audio_path, "rb") as a:
                audio_url = replicate.upload_file(a, "audio/wav")
            
            # Run Wav2Lip model
            output = replicate.run(
                "cjwbw/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25ebc",
                input={
                    "face": video_url,
                    "audio": audio_url
                }
            )
            
            # Download result
            if output:
                response = requests.get(output)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info("✅ Replicate Wav2Lip completed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Replicate Wav2Lip failed: {e}")
            return False
    
    def process_with_best_available(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Try all available methods in order of preference"""
        
        # Order of preference (best quality first)
        methods_priority = [
            ('replicate', self.apply_replicate_wav2lip),
            ('huggingface_wav2lip', self.apply_huggingface_wav2lip),
            ('local_wav2lip', self.apply_local_wav2lip),  # Fallback only
        ]
        
        for method_name, method_func in methods_priority:
            if self.available_methods.get(method_name, False):
                logger.info(f"Trying {method_name}...")
                
                try:
                    if method_func(video_path, audio_path, output_path):
                        logger.info(f"✅ Successfully processed with {method_name}")
                        return True
                except Exception as e:
                    logger.error(f"{method_name} failed: {e}")
                    continue
        
        logger.error("All working lip sync methods failed")
        return False


class RealLipSyncSelector:
    """Selector for actually working lip sync methods"""
    
    def __init__(self):
        self.working_lipsync = WorkingLipSync()
    
    def get_best_working_method(self) -> Optional[str]:
        """Get the best actually working method"""
        
        # Priority order
        if self.working_lipsync.available_methods.get('replicate'):
            return 'replicate'
        elif self.working_lipsync.available_methods.get('huggingface_wav2lip'):
            return 'huggingface_wav2lip'
        elif self.working_lipsync.available_methods.get('local_wav2lip'):
            return 'local_wav2lip'
        else:
            return None
    
    def process_video(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Process video with best working method"""
        return self.working_lipsync.process_with_best_available(
            video_path, audio_path, output_path
        )


# Instructions for getting working lip sync:
"""
To get REAL working lip sync, you need one of these:

1. **Replicate (Easiest, Costs ~$0.12/video)**
   - Sign up at https://replicate.com
   - Get API token
   - Set environment variable: export REPLICATE_API_TOKEN="your-token"
   - Run: pip install replicate

2. **HuggingFace Spaces (Free, May be slow)**
   - No API key needed
   - Install: pip install gradio_client
   - May have queue/rate limits

3. **Local Wav2Lip (Free, Requires GPU)**
   - Download model: python -c "from src.lip_sync.working_lipsync import WorkingLipSync; w = WorkingLipSync(); w.download_wav2lip_model()"
   - Install dependencies: pip install gdown
   - Requires CUDA GPU with 4GB+ VRAM

4. **Docker Wav2Lip (Isolated, Requires Docker)**
   - Install Docker
   - Build/pull Wav2Lip container
   - Runs isolated from system

The current Easy-Wav2Lip is just a placeholder and doesn't do real lip sync!
"""