"""
Replicate API client for professional lip sync processing
High-quality cloud-based lip sync with automatic scaling
"""

import os
import time
import logging
import requests
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ReplicateClient:
    """Client for Replicate lip sync APIs"""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            logger.warning("No Replicate API token provided. Cloud lip sync will be disabled.")
        
        self.base_url = "https://api.replicate.com/v1"
        self.session = requests.Session()
        if self.api_token:
            self.session.headers.update({
                "Authorization": f"Token {self.api_token}",
                "Content-Type": "application/json"
            })
    
    def test_connection(self) -> bool:
        """Test API connection"""
        if not self.api_token:
            return False
        
        try:
            response = self.session.get(f"{self.base_url}/account")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Replicate connection test failed: {e}")
            return False
    
    def upload_file(self, file_path: str) -> Optional[str]:
        """Upload file to Replicate and return URL"""
        try:
            # Get upload URL
            response = self.session.post(
                f"{self.base_url}/files",
                json={"type": "image" if file_path.endswith(('.jpg', '.png')) else "video"}
            )
            response.raise_for_status()
            
            upload_data = response.json()
            upload_url = upload_data["upload_url"]
            
            # Upload file
            with open(file_path, 'rb') as f:
                upload_response = requests.put(upload_url, data=f)
                upload_response.raise_for_status()
            
            logger.info(f"Uploaded {Path(file_path).name} to Replicate")
            return upload_data["url"]
            
        except Exception as e:
            logger.error(f"Failed to upload file to Replicate: {e}")
            return None
    
    def run_wav2lip(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Run Wav2Lip on Replicate"""
        try:
            if not self.api_token:
                logger.error("No Replicate API token available")
                return False
            
            logger.info("Starting Replicate Wav2Lip processing...")
            start_time = time.time()
            
            # Upload files
            video_url = self.upload_file(video_path)
            audio_url = self.upload_file(audio_path)
            
            if not video_url or not audio_url:
                logger.error("Failed to upload files")
                return False
            
            # Run prediction
            prediction_data = {
                "version": "cjwbw/wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25ebc",
                "input": {
                    "face": video_url,
                    "audio": audio_url,
                    "pads": [0, 10, 0, 0],  # Padding for face detection
                    "smooth": True,
                    "resize_factor": 1
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/predictions",
                json=prediction_data
            )
            response.raise_for_status()
            
            prediction = response.json()
            prediction_id = prediction["id"]
            
            logger.info(f"Prediction started: {prediction_id}")
            
            # Poll for completion
            while True:
                response = self.session.get(f"{self.base_url}/predictions/{prediction_id}")
                response.raise_for_status()
                
                prediction = response.json()
                status = prediction["status"]
                
                if status == "succeeded":
                    output_url = prediction["output"]
                    break
                elif status == "failed":
                    error = prediction.get("error", "Unknown error")
                    logger.error(f"Replicate prediction failed: {error}")
                    return False
                elif status in ["starting", "processing"]:
                    logger.info(f"Status: {status}...")
                    time.sleep(2)
                else:
                    logger.warning(f"Unknown status: {status}")
                    time.sleep(2)
            
            # Download result
            logger.info("Downloading result...")
            download_response = requests.get(output_url)
            download_response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(download_response.content)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Replicate Wav2Lip completed in {processing_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Replicate Wav2Lip failed: {e}")
            return False
    
    def run_sadtalker(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Run SadTalker on Replicate for higher quality"""
        try:
            if not self.api_token:
                logger.error("No Replicate API token available")
                return False
            
            logger.info("Starting Replicate SadTalker processing...")
            start_time = time.time()
            
            # Upload files
            video_url = self.upload_file(video_path)
            audio_url = self.upload_file(audio_path)
            
            if not video_url or not audio_url:
                logger.error("Failed to upload files")
                return False
            
            # Run SadTalker prediction
            prediction_data = {
                "version": "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
                "input": {
                    "source_image": video_url,  # First frame will be extracted
                    "driven_audio": audio_url,
                    "preprocess": "crop",
                    "still": True,
                    "use_enhancer": True,
                    "use_eye_blinking": True
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/predictions",
                json=prediction_data
            )
            response.raise_for_status()
            
            prediction = response.json()
            prediction_id = prediction["id"]
            
            logger.info(f"SadTalker prediction started: {prediction_id}")
            
            # Poll for completion
            while True:
                response = self.session.get(f"{self.base_url}/predictions/{prediction_id}")
                response.raise_for_status()
                
                prediction = response.json()
                status = prediction["status"]
                
                if status == "succeeded":
                    output_url = prediction["output"]
                    break
                elif status == "failed":
                    error = prediction.get("error", "Unknown error")
                    logger.error(f"SadTalker prediction failed: {error}")
                    return False
                elif status in ["starting", "processing"]:
                    logger.info(f"Status: {status}...")
                    time.sleep(3)
                else:
                    logger.warning(f"Unknown status: {status}")
                    time.sleep(3)
            
            # Download result
            logger.info("Downloading SadTalker result...")
            download_response = requests.get(output_url)
            download_response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(download_response.content)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Replicate SadTalker completed in {processing_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Replicate SadTalker failed: {e}")
            return False


class ReplicateManager:
    """Manager for Replicate lip sync services"""
    
    def __init__(self, api_token: Optional[str] = None):
        self.client = ReplicateClient(api_token)
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        """Check if Replicate is available"""
        self.available = self.client.test_connection()
        if self.available:
            logger.info("✅ Replicate API available")
        else:
            logger.warning("❌ Replicate API not available")
    
    def is_available(self) -> bool:
        """Check if Replicate is available"""
        return self.available
    
    def process_video(self, video_path: str, audio_path: str, output_path: str, 
                     model: str = "wav2lip") -> bool:
        """Process video with specified model"""
        if not self.is_available():
            logger.error("Replicate not available")
            return False
        
        if model == "sadtalker":
            return self.client.run_sadtalker(video_path, audio_path, output_path)
        else:
            return self.client.run_wav2lip(video_path, audio_path, output_path)
    
    def estimate_cost(self, duration_seconds: float, model: str = "wav2lip") -> float:
        """Estimate processing cost"""
        # Replicate pricing (approximate)
        if model == "sadtalker":
            cost_per_second = 0.0032  # A100 pricing
            processing_time = duration_seconds * 3  # Estimate 3x real-time
        else:
            cost_per_second = 0.0025  # T4 pricing
            processing_time = duration_seconds * 2  # Estimate 2x real-time
        
        return cost_per_second * processing_time


# Global manager instance
replicate_manager = ReplicateManager()