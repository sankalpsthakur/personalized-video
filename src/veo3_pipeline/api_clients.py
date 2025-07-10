#!/usr/bin/env python3
"""
API client implementations for FLUX Kontext, Veo 3, and ElevenLabs
Production-ready with error handling, retries, and rate limiting
"""

import os
import json
import time
import logging
from typing import Dict, Optional, List, Any
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
import aiohttp
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)


class BaseAPIClient:
    """Base class for API clients with common functionality"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
        
    def _handle_response(self, response: requests.Response) -> Dict:
        """Handle API response with error checking"""
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            raise Exception(f"Rate limited. Retry after {retry_after} seconds")
            
        response.raise_for_status()
        return response.json()


class FluxKontextClient(BaseAPIClient):
    """FLUX Kontext API client for image generation and editing"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            api_key=api_key or os.getenv("FLUX_KONTEXT_API_KEY"),
            base_url="https://api.flux-kontext.ai/v1"
        )
        
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_version: str = "dev",
        seed: Optional[int] = None,
        width: int = 1920,
        height: int = 1080,
        reference_image: Optional[Path] = None
    ) -> Dict:
        """Generate or edit image using FLUX Kontext"""
        
        endpoint = f"{self.base_url}/generate"
        
        # Build layered prompt
        full_prompt = f"{prompt} ++ {negative_prompt}"
        
        payload = {
            "prompt": full_prompt,
            "model_version": model_version,
            "width": width,
            "height": height,
            "num_outputs": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }
        
        if seed:
            payload["seed"] = seed
            
        # Add reference image for editing
        if reference_image and reference_image.exists():
            with open(reference_image, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            payload["init_image"] = image_data
            payload["strength"] = 0.75  # Control how much to modify
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = self.session.post(endpoint, json=payload, headers=headers)
        result = self._handle_response(response)
        
        # Add edit log
        result["edit_log"] = {
            "prompt": full_prompt,
            "model_version": model_version,
            "seed": result.get("seed", seed),
            "parameters": payload
        }
        
        return result
        
    def upscale_image(self, image_path: Path, scale_factor: int = 2) -> Dict:
        """Upscale image using FLUX enhancement"""
        endpoint = f"{self.base_url}/upscale"
        
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"scale_factor": scale_factor}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = self.session.post(
                endpoint, files=files, data=data, headers=headers
            )
            
        return self._handle_response(response)


class Veo3Client(BaseAPIClient):
    """Veo 3 API client for video generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            api_key=api_key or os.getenv("VEO3_API_KEY"),
            base_url="https://api.veo3.ai/v1"
        )
        
    async def generate_video_async(
        self,
        conditioning_frame: Path,
        prompt: str,
        duration: float = 6.0,
        fps: int = 24,
        quality: str = "standard",
        camera_motion: Optional[str] = None
    ) -> Dict:
        """Generate video from conditioning frame (async)"""
        
        endpoint = f"{self.base_url}/generate"
        
        # Structure prompt with 6-part schema
        structured_prompt = self._structure_prompt(prompt, camera_motion)
        
        # Prepare multipart data
        with open(conditioning_frame, "rb") as f:
            frame_data = base64.b64encode(f.read()).decode()
            
        payload = {
            "conditioning_frame": frame_data,
            "prompt": structured_prompt,
            "duration": duration,
            "fps": fps,
            "quality": quality,
            "output_format": "prores"
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                if response.status == 202:  # Accepted for processing
                    result = await response.json()
                    # Poll for completion
                    return await self._poll_generation(session, result["job_id"])
                else:
                    response.raise_for_status()
                    
    async def _poll_generation(self, session: aiohttp.ClientSession, job_id: str) -> Dict:
        """Poll for video generation completion"""
        endpoint = f"{self.base_url}/status/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        while True:
            async with session.get(endpoint, headers=headers) as response:
                result = await response.json()
                
                if result["status"] == "completed":
                    return result
                elif result["status"] == "failed":
                    raise Exception(f"Generation failed: {result.get('error')}")
                    
                # Wait before polling again
                await asyncio.sleep(5)
                
    def _structure_prompt(self, base_prompt: str, camera_motion: Optional[str]) -> str:
        """Structure prompt according to 6-part schema"""
        parts = base_prompt.split("::")
        
        # Ensure we have all parts
        schema = {
            "subject": parts[0] if len(parts) > 0 else "character",
            "context": parts[1] if len(parts) > 1 else "in scene",
            "action": parts[2] if len(parts) > 2 else "natural movement",
            "style": parts[3] if len(parts) > 3 else "cinematic",
            "camera_motion": camera_motion or "static",
            "composition": parts[4] if len(parts) > 4 else "medium shot"
        }
        
        return " :: ".join([
            f"{schema['subject']} {schema['context']}",
            schema['action'],
            schema['style'],
            schema['camera_motion'],
            schema['composition']
        ])
        
    def generate_multi_angle(
        self,
        conditioning_frame: Path,
        base_prompt: str,
        angles: List[str] = None
    ) -> List[Dict]:
        """Generate multiple angle variations"""
        
        if angles is None:
            angles = ["front", "three_quarter_left", "profile_left", "three_quarter_right"]
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        tasks = []
        for angle in angles:
            angle_prompt = f"{base_prompt} :: {angle} view"
            task = self.generate_video_async(
                conditioning_frame=conditioning_frame,
                prompt=angle_prompt
            )
            tasks.append(task)
            
        results = loop.run_until_complete(asyncio.gather(*tasks))
        loop.close()
        
        return results


class ElevenLabsClient(BaseAPIClient):
    """ElevenLabs API client for voice cloning and TTS"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            api_key=api_key or os.getenv("ELEVENLABS_API_KEY"),
            base_url="https://api.elevenlabs.io/v1"
        )
        
    def clone_voice(
        self,
        name: str,
        files: List[Path],
        description: str = "",
        labels: Optional[Dict] = None
    ) -> Dict:
        """Create professional voice clone"""
        
        endpoint = f"{self.base_url}/voices/add"
        
        # Validate audio files
        for file_path in files:
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
                
        # Prepare multipart upload
        files_data = []
        for i, file_path in enumerate(files):
            with open(file_path, "rb") as f:
                files_data.append(("files", (file_path.name, f.read(), "audio/wav")))
                
        data = {
            "name": name,
            "description": description,
            "labels": json.dumps(labels or {})
        }
        
        headers = {"xi-api-key": self.api_key}
        
        response = self.session.post(
            endpoint, files=files_data, data=data, headers=headers
        )
        
        return self._handle_response(response)
        
    def generate_audio(
        self,
        text: str,
        voice_id: str,
        model_id: str = "eleven_turbo_v2",
        voice_settings: Optional[Dict] = None
    ) -> bytes:
        """Generate audio using TTS"""
        
        endpoint = f"{self.base_url}/text-to-speech/{voice_id}"
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings or {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        
        response = self.session.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        
        return response.content
        
    def audio_to_audio(
        self,
        audio_file: Path,
        voice_id: str,
        model_id: str = "eleven_english_sts_v2"
    ) -> bytes:
        """Convert audio to match target voice (audio-to-audio)"""
        
        endpoint = f"{self.base_url}/speech-to-speech/{voice_id}"
        
        with open(audio_file, "rb") as f:
            files = {"audio": (audio_file.name, f.read(), "audio/wav")}
            
        data = {"model_id": model_id}
        headers = {"xi-api-key": self.api_key}
        
        response = self.session.post(
            endpoint, files=files, data=data, headers=headers
        )
        response.raise_for_status()
        
        return response.content
        
    async def stream_audio(
        self,
        text: str,
        voice_id: str,
        websocket_url: Optional[str] = None
    ):
        """Stream audio generation via WebSocket"""
        
        if websocket_url is None:
            websocket_url = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
            
        # WebSocket streaming implementation
        # This would connect to ElevenLabs WebSocket API for real-time streaming
        pass


# Rate limiting decorator
def rate_limit(calls: int, period: int):
    """Rate limit API calls"""
    def decorator(func):
        last_called = [0.0]
        
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < period / calls:
                time.sleep(period / calls - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


# API client factory
class APIClientFactory:
    """Factory for creating API clients with proper configuration"""
    
    @staticmethod
    def create_flux_client() -> FluxKontextClient:
        """Create FLUX Kontext client"""
        return FluxKontextClient()
        
    @staticmethod
    def create_veo3_client() -> Veo3Client:
        """Create Veo 3 client"""
        return Veo3Client()
        
    @staticmethod
    def create_elevenlabs_client() -> ElevenLabsClient:
        """Create ElevenLabs client"""
        return ElevenLabsClient()