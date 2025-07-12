"""
Cloud-based lip sync clients for fal.ai and other providers
State-of-the-art lip sync quality with zero local dependencies
"""

import os
import logging
import time
import requests
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile

logger = logging.getLogger(__name__)


class CloudLipSyncClient:
    """Client for cloud-based lip sync APIs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FAL_KEY")
        if not self.api_key:
            logger.warning("No fal.ai API key provided. Cloud lip sync will be disabled.")
        
        self.base_url = "https://fal.run"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def _upload_file_to_fal(self, file_path: str) -> str:
        """Upload file to fal.ai storage and return URL"""
        try:
            # Get upload URL
            response = self.session.post(
                f"{self.base_url}/storage/upload/initiate",
                json={"file_name": Path(file_path).name}
            )
            response.raise_for_status()
            
            upload_data = response.json()
            upload_url = upload_data["upload_url"]
            file_url = upload_data["file_url"]
            
            # Upload file
            with open(file_path, 'rb') as f:
                upload_response = requests.put(upload_url, data=f)
                upload_response.raise_for_status()
            
            logger.info(f"Uploaded {Path(file_path).name} to fal.ai storage")
            return file_url
            
        except Exception as e:
            logger.error(f"Failed to upload file to fal.ai: {e}")
            raise
    
    def apply_pixverse_lipsync(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Apply Pixverse lip sync using fal.ai API"""
        try:
            if not self.api_key:
                logger.error("No fal.ai API key available for Pixverse lip sync")
                return False
            
            logger.info("Starting Pixverse lip sync...")
            start_time = time.time()
            
            # Upload video and audio files
            logger.info("Uploading video and audio files...")
            video_url = self._upload_file_to_fal(video_path)
            audio_url = self._upload_file_to_fal(audio_path)
            
            # Submit lip sync job
            job_data = {
                "video_url": video_url,
                "audio_url": audio_url
            }
            
            response = self.session.post(
                f"{self.base_url}/fal-ai/pixverse/lipsync",
                json=job_data
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check if job completed immediately or needs polling
            if "video_url" in result:
                # Job completed immediately
                output_video_url = result["video_url"]
            else:
                # Job submitted, need to poll for completion
                job_id = result.get("request_id")
                if not job_id:
                    logger.error("No job ID returned from Pixverse API")
                    return False
                
                logger.info(f"Job submitted with ID: {job_id}. Polling for completion...")
                output_video_url = self._poll_for_completion(job_id)
                
                if not output_video_url:
                    logger.error("Pixverse lip sync job failed or timed out")
                    return False
            
            # Download result
            logger.info("Downloading lip-synced video...")
            download_response = requests.get(output_video_url)
            download_response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(download_response.content)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Pixverse lip sync completed in {processing_time:.2f}s")
            logger.info(f"Output saved to: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pixverse lip sync failed: {e}")
            return False
    
    def apply_synclabs_lipsync(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Apply SyncLabs lip sync using fal.ai API (alternative option)"""
        try:
            if not self.api_key:
                logger.error("No fal.ai API key available for SyncLabs lip sync")
                return False
            
            logger.info("Starting SyncLabs lip sync...")
            start_time = time.time()
            
            # Upload files
            video_url = self._upload_file_to_fal(video_path)
            audio_url = self._upload_file_to_fal(audio_path)
            
            # Submit job
            job_data = {
                "video_url": video_url,
                "audio_url": audio_url
            }
            
            response = self.session.post(
                f"{self.base_url}/fal-ai/sync-lipsync",
                json=job_data
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "video_url" in result:
                output_video_url = result["video_url"]
            else:
                job_id = result.get("request_id")
                if not job_id:
                    logger.error("No job ID returned from SyncLabs API")
                    return False
                
                output_video_url = self._poll_for_completion(job_id)
                if not output_video_url:
                    return False
            
            # Download result
            download_response = requests.get(output_video_url)
            download_response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(download_response.content)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ SyncLabs lip sync completed in {processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"SyncLabs lip sync failed: {e}")
            return False
    
    def _poll_for_completion(self, job_id: str, max_wait: int = 300) -> Optional[str]:
        """Poll for job completion and return result URL"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = self.session.get(f"{self.base_url}/fal-ai/queue/requests/{job_id}/status")
                response.raise_for_status()
                
                status_data = response.json()
                status = status_data.get("status")
                
                if status == "COMPLETED":
                    result = status_data.get("result", {})
                    return result.get("video_url")
                elif status == "FAILED":
                    error = status_data.get("error", "Unknown error")
                    logger.error(f"Job failed: {error}")
                    return None
                elif status in ["IN_PROGRESS", "IN_QUEUE"]:
                    logger.info(f"Job status: {status}. Waiting...")
                    time.sleep(5)
                else:
                    logger.warning(f"Unknown job status: {status}")
                    time.sleep(5)
                    
            except Exception as e:
                logger.error(f"Error polling job status: {e}")
                time.sleep(5)
        
        logger.error(f"Job {job_id} timed out after {max_wait}s")
        return None
    
    def test_connection(self) -> bool:
        """Test API connection and authentication"""
        if not self.api_key:
            logger.error("No API key provided")
            return False
        
        try:
            # Test with a simple API call
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                logger.info("✅ fal.ai API connection successful")
                return True
            else:
                logger.error(f"API connection failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False


class LipSyncEngineSelector:
    """Selects the best available lip sync engine based on quality and availability"""
    
    def __init__(self, fal_api_key: Optional[str] = None):
        self.cloud_client = CloudLipSyncClient(fal_api_key)
        self.available_engines = self._check_available_engines()
    
    def _check_available_engines(self) -> Dict[str, bool]:
        """Check which lip sync engines are available"""
        engines = {
            "pixverse": False,
            "synclabs": False,
            "wav2lip": False,
            "musetalk": False,
            "audio_only": True  # Always available fallback
        }
        
        # Check cloud APIs
        if self.cloud_client.test_connection():
            engines["pixverse"] = True
            engines["synclabs"] = True
        
        # Check local models (simplified check)
        try:
            from .models import Wav2LipInference
            engines["wav2lip"] = True
        except:
            pass
        
        try:
            from .models import MuseTalkInference
            engines["musetalk"] = True
        except:
            pass
        
        logger.info(f"Available lip sync engines: {[k for k, v in engines.items() if v]}")
        return engines
    
    def get_best_engine(self, quality_priority: bool = True) -> str:
        """Get the best available engine"""
        # Priority order: highest quality first
        if quality_priority:
            priority_order = ["pixverse", "synclabs", "musetalk", "wav2lip", "audio_only"]
        else:
            # Speed priority
            priority_order = ["wav2lip", "musetalk", "pixverse", "synclabs", "audio_only"]
        
        for engine in priority_order:
            if self.available_engines.get(engine, False):
                logger.info(f"Selected lip sync engine: {engine}")
                return engine
        
        return "audio_only"
    
    def apply_lip_sync(self, video_path: str, audio_path: str, output_path: str, 
                      engine: Optional[str] = None) -> bool:
        """Apply lip sync using the best available engine"""
        if not engine:
            engine = self.get_best_engine(quality_priority=True)
        
        logger.info(f"Applying lip sync with engine: {engine}")
        
        try:
            if engine == "pixverse":
                return self.cloud_client.apply_pixverse_lipsync(video_path, audio_path, output_path)
            elif engine == "synclabs":
                return self.cloud_client.apply_synclabs_lipsync(video_path, audio_path, output_path)
            elif engine in ["wav2lip", "musetalk"]:
                # Use existing local implementation (with fixes)
                from .lip_sync import LipSyncProcessor
                processor = LipSyncProcessor(model=engine)
                # Note: This will need to be updated to work with complete audio
                return processor.apply_complete_lip_sync(video_path, audio_path, output_path)
            else:
                # Fallback to audio replacement
                logger.info("No lip sync engine available, using audio replacement")
                return self._apply_audio_replacement(video_path, audio_path, output_path)
                
        except Exception as e:
            logger.error(f"Lip sync with {engine} failed: {e}")
            # Fallback to audio replacement
            return self._apply_audio_replacement(video_path, audio_path, output_path)
    
    def _apply_audio_replacement(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Fallback: simple audio replacement without lip sync"""
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-i", video_path, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("✅ Audio replacement completed (no lip sync)")
            return True
        except Exception as e:
            logger.error(f"Audio replacement failed: {e}")
            return False