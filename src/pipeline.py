"""
Template-based video personalization pipeline
Complete TTS generation with variable replacement
"""

import asyncio
import logging
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import soundfile as sf

from .templates import TRANSCRIPT_TEMPLATE, DEFAULT_VARIABLES, REQUIRED_VARIABLES

logger = logging.getLogger(__name__)

class VideoPersonalizationPipeline:
    """
    Template-based pipeline for video personalization
    Generates complete TTS audio from template with variable replacement
    """
    
    def __init__(self, output_dir: str = "output", log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Setup comprehensive logging
        self._setup_logging(log_level)
        
        # Pipeline state
        self.original_duration = None
        self.personalized_transcript = None
        self.processing_stats = {
            "start_time": None,
            "end_time": None,
            "stages": {},
            "errors": [],
            "warnings": []
        }
        
        logger.info(f"VideoPersonalizationPipeline initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Temp directory: {self.temp_dir}")
        
    def _setup_logging(self, log_level: str):
        """Setup comprehensive logging configuration"""
        # Create logs directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup file logging with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")
    
    def _log_stage_start(self, stage_name: str):
        """Log the start of a processing stage"""
        start_time = time.time()
        self.processing_stats["stages"][stage_name] = {"start": start_time, "end": None, "duration": None}
        logger.info(f"🚀 Starting stage: {stage_name}")
        return start_time
    
    def _log_stage_end(self, stage_name: str, start_time: float):
        """Log the end of a processing stage"""
        end_time = time.time()
        duration = end_time - start_time
        self.processing_stats["stages"][stage_name].update({
            "end": end_time,
            "duration": duration
        })
        logger.info(f"✅ Completed stage: {stage_name} (Duration: {duration:.2f}s)")
        return duration
    
    def _log_warning(self, message: str):
        """Log a warning and track it"""
        logger.warning(message)
        self.processing_stats["warnings"].append(message)
    
    def _log_error(self, message: str, exception: Exception = None):
        """Log an error and track it"""
        error_info = {"message": message, "exception": str(exception) if exception else None}
        logger.error(message)
        if exception:
            logger.error(f"Exception details: {exception}")
        self.processing_stats["errors"].append(error_info)
        
    def validate_variables(self, variables: Dict[str, str]) -> bool:
        """Validate that all required variables are provided"""
        missing = [var for var in REQUIRED_VARIABLES if var not in variables]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return True
    
    def create_personalized_transcript(self, 
                                     template: str = TRANSCRIPT_TEMPLATE,
                                     variables: Dict[str, str] = None) -> str:
        """Create personalized transcript from template"""
        if variables is None:
            variables = DEFAULT_VARIABLES.copy()
        
        # Validate variables
        self.validate_variables(variables)
        
        # Generate personalized transcript
        try:
            personalized = template.format(**variables)
            self.personalized_transcript = personalized
            
            logger.info(f"Personalized transcript created:")
            logger.info(f"  Customer: {variables.get('customer_name')}")
            logger.info(f"  Destination: {variables.get('destination')}")
            logger.info(f"  Length: {len(personalized)} characters")
            
            return personalized
            
        except KeyError as e:
            raise ValueError(f"Template contains undefined variable: {e}")
    
    def get_video_duration(self, video_path: str) -> float:
        """Get original video duration for timing reference"""
        start_time = self._log_stage_start("video_duration_extraction")
        
        try:
            # Validate video file exists
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Get video info
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", 
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            
            logger.info(f"Extracting duration from: {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            duration = float(result.stdout.strip())
            self.original_duration = duration
            
            # Get additional video info for logging
            info_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0", 
                "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                "-of", "csv=p=0", video_path
            ]
            info_result = subprocess.run(info_cmd, capture_output=True, text=True)
            
            if info_result.returncode == 0:
                info_parts = info_result.stdout.strip().split(',')
                if len(info_parts) >= 4:
                    width, height, fps, codec = info_parts
                    logger.info(f"Video properties: {width}x{height}, {fps} fps, codec: {codec}")
            
            logger.info(f"Original video duration: {duration:.2f}s")
            self._log_stage_end("video_duration_extraction", start_time)
            return duration
            
        except Exception as e:
            self._log_error(f"Failed to extract video duration: {str(e)}", e)
            raise
    
    def generate_complete_tts(self, text: str, target_duration: float = None) -> Path:
        """Generate complete TTS audio for the entire transcript"""
        start_time = self._log_stage_start("tts_generation")
        
        try:
            logger.info(f"Generating TTS for {len(text.split())} words, {len(text)} characters")
            if target_duration:
                logger.info(f"Target duration: {target_duration:.2f}s")
            
            # Try different TTS engines in order of preference
            tts_path = self.temp_dir / "complete_tts.wav"
            tts_engine_used = None
            
            # Try Edge-TTS first
            logger.info("Attempting TTS generation with Edge-TTS...")
            if self._try_edge_tts(text, tts_path):
                logger.info("✓ TTS generated with Edge-TTS")
                tts_engine_used = "Edge-TTS"
            else:
                # Try ElevenLabs
                logger.info("Edge-TTS failed, attempting ElevenLabs...")
                if self._try_elevenlabs(text, tts_path):
                    logger.info("✓ TTS generated with ElevenLabs")
                    tts_engine_used = "ElevenLabs"
                else:
                    # Fallback to gTTS
                    logger.info("ElevenLabs failed, using gTTS fallback...")
                    self._generate_gtts(text, tts_path)
                    logger.info("✓ TTS generated with gTTS (fallback)")
                    tts_engine_used = "gTTS"
            
            # Validate generated audio
            if not tts_path.exists():
                raise FileNotFoundError("TTS audio file was not created")
            
            # Log audio properties
            try:
                audio, sr = sf.read(tts_path)
                generated_duration = len(audio) / sr
                logger.info(f"Generated audio: {generated_duration:.2f}s, {sr}Hz, {len(audio)} samples")
                logger.info(f"TTS engine used: {tts_engine_used}")
            except Exception as e:
                self._log_warning(f"Could not analyze generated audio: {e}")
            
            # Check duration and adjust if needed
            final_path = tts_path
            if target_duration:
                logger.info("Applying duration matching...")
                adjusted_path = self._adjust_tts_duration(tts_path, target_duration)
                final_path = adjusted_path
            
            self._log_stage_end("tts_generation", start_time)
            return final_path
            
        except Exception as e:
            self._log_error(f"TTS generation failed: {str(e)}", e)
            raise
    
    def _try_edge_tts(self, text: str, output_path: Path) -> bool:
        """Generate TTS using Edge-TTS (Microsoft)"""
        try:
            import edge_tts
            
            async def generate():
                # Use a professional, clear voice
                voice = "en-US-AriaNeural"  # Professional female voice
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(output_path.with_suffix('.mp3')))
            
            asyncio.run(generate())
            
            # Convert to WAV at 48kHz for video compatibility
            cmd = [
                "ffmpeg", "-i", str(output_path.with_suffix('.mp3')),
                "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
                "-y", str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Clean up MP3
            output_path.with_suffix('.mp3').unlink()
            
            return True
            
        except Exception as e:
            logger.warning(f"Edge-TTS failed: {e}")
            return False
    
    def _try_elevenlabs(self, text: str, output_path: Path) -> bool:
        """Generate TTS using ElevenLabs (if API key available)"""
        import os
        
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            return False
        
        try:
            import requests
            
            # Use a professional voice
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (professional)
            
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.7,
                    "similarity_boost": 0.8,
                    "style": 0.2,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                # Save as MP3 first
                mp3_path = output_path.with_suffix('.mp3')
                mp3_path.write_bytes(response.content)
                
                # Convert to WAV
                cmd = [
                    "ffmpeg", "-i", str(mp3_path),
                    "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
                    "-y", str(output_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                mp3_path.unlink()
                
                return True
                
        except Exception as e:
            logger.warning(f"ElevenLabs failed: {e}")
        
        return False
    
    def _generate_gtts(self, text: str, output_path: Path):
        """Fallback TTS using Google Text-to-Speech"""
        from gtts import gTTS
        
        tts = gTTS(text=text, lang='en', slow=False)
        mp3_path = output_path.with_suffix('.mp3')
        tts.save(str(mp3_path))
        
        # Convert to WAV
        cmd = [
            "ffmpeg", "-i", str(mp3_path),
            "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
            "-y", str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        mp3_path.unlink()
    
    def _adjust_tts_duration(self, tts_path: Path, target_duration: float) -> Path:
        """Adjust TTS duration to match target (gentle adjustment only)"""
        # Load audio
        audio, sr = sf.read(tts_path)
        current_duration = len(audio) / sr
        
        duration_ratio = target_duration / current_duration
        
        logger.info(f"TTS duration: {current_duration:.2f}s, target: {target_duration:.2f}s")
        logger.info(f"Duration ratio: {duration_ratio:.3f}")
        
        # Only adjust if significantly different and within reasonable bounds
        if 0.8 <= duration_ratio <= 1.2:
            if abs(duration_ratio - 1.0) > 0.05:  # More than 5% difference
                logger.info(f"Applying gentle speed adjustment: {duration_ratio:.3f}x")
                adjusted_audio = self._apply_speed_adjustment(audio, sr, duration_ratio)
                
                adjusted_path = self.temp_dir / "duration_adjusted_tts.wav"
                sf.write(adjusted_path, adjusted_audio, sr)
                return adjusted_path
            else:
                logger.info("Duration close enough, no adjustment needed")
        else:
            logger.warning(f"Duration ratio {duration_ratio:.3f} outside safe range, keeping original")
        
        return tts_path
    
    def _apply_speed_adjustment(self, audio: np.ndarray, sr: int, speed_factor: float) -> np.ndarray:
        """Apply gentle speed adjustment using high-quality time stretching"""
        try:
            import librosa
            # Use phase vocoder for high-quality time stretching
            adjusted = librosa.effects.time_stretch(audio, rate=1/speed_factor)
            return adjusted
        except:
            logger.warning("Librosa not available, using simple resampling")
            # Fallback to simple resampling
            from scipy import signal
            num_samples = int(len(audio) * speed_factor)
            adjusted = signal.resample(audio, num_samples)
            return adjusted
    
    def combine_with_video(self, video_path: str, audio_path: Path, apply_lip_sync: bool = True) -> Path:
        """Combine generated audio with video"""
        output_path = self.output_dir / "personalized_video.mp4"
        
        if apply_lip_sync:
            logger.info("Applying lip sync to complete video...")
            # Apply lip sync to entire video with new audio
            lip_synced_path = self._apply_lip_sync(video_path, audio_path)
            
            # Copy to final output
            import shutil
            shutil.copy2(lip_synced_path, output_path)
        else:
            logger.info("Combining audio with video (no lip sync)...")
            # Simple audio replacement
            cmd = [
                "ffmpeg", "-i", video_path, "-i", str(audio_path),
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest",  # Match shortest stream
                "-y", str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def _apply_lip_sync(self, video_path: str, audio_path: Path) -> Path:
        """Apply lip sync using smart selector for optimal method"""
        logger.info("Applying intelligent lip sync selection...")
        
        try:
            # Import the smart selector system
            from .lip_sync.smart_selector import smart_selector, ProcessingOptions
            
            # Create processing options that FORCE real lip sync
            options = ProcessingOptions(
                quality_priority=True,  # Prioritize quality for video personalization
                max_cost_usd=1.0,  # Reasonable cost limit
                max_processing_time_seconds=300,  # 5 minutes max
                prefer_local=True,  # Try local first
                fallback_to_audio_only=False  # NO fallback to audio-only - force lip sync
            )
            
            # Apply lip sync with intelligent selection
            lip_synced_path = self.temp_dir / "lip_synced_video.mp4"
            
            success, method_used = smart_selector.process_video(
                video_path=video_path,
                audio_path=str(audio_path),
                output_path=str(lip_synced_path),
                options=options
            )
            
            if success and lip_synced_path.exists() and method_used != "audio_only":
                logger.info(f"✅ Lip sync completed successfully using: {method_used}")
                return lip_synced_path
            else:
                # If smart selector fell back to audio_only, force use of working lip sync
                logger.warning("Smart selector failed, trying direct working lip sync...")
                
                from .lip_sync.working_lipsync import RealLipSyncSelector
                real_selector = RealLipSyncSelector()
                
                direct_output = self.temp_dir / "direct_lip_synced.mp4"
                direct_success = real_selector.process_video(
                    video_path=video_path,
                    audio_path=str(audio_path),
                    output_path=str(direct_output)
                )
                
                if direct_success and direct_output.exists():
                    logger.info("✅ Direct lip sync completed successfully")
                    return direct_output
                else:
                    raise Exception("All lip sync methods failed")
                
        except Exception as e:
            logger.error(f"Lip sync completely failed: {e}")
            # Only as absolute last resort, use audio replacement
            logger.warning("FINAL FALLBACK: Using audio replacement only (no lip sync)")
            fallback_path = self.temp_dir / "audio_replaced_video.mp4"
            cmd = [
                "ffmpeg", "-i", video_path, "-i", str(audio_path),
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", str(fallback_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return fallback_path
    
    def create_personalized_video(self, 
                                video_path: str,
                                variables: Dict[str, str],
                                template: str = TRANSCRIPT_TEMPLATE,
                                apply_lip_sync: bool = True) -> Path:
        """
        Create personalized video using template approach
        
        Args:
            video_path: Path to original video
            variables: Variables to replace (e.g., {"customer_name": "Sarah", "destination": "Tokyo"})
            template: Transcript template with {variable} placeholders
            apply_lip_sync: Whether to apply lip sync (True) or just replace audio (False)
        
        Returns:
            Path to personalized video
        """
        # Start overall timing
        self.processing_stats["start_time"] = time.time()
        
        logger.info("="*80)
        logger.info("VIDEO PERSONALIZATION PIPELINE")
        logger.info("="*80)
        logger.info(f"📹 Input video: {Path(video_path).name}")
        logger.info(f"📂 Input size: {Path(video_path).stat().st_size / (1024*1024):.1f} MB")
        logger.info(f"🔧 Variables: {variables}")
        logger.info(f"🎭 Lip sync: {'Enabled' if apply_lip_sync else 'Disabled'}")
        logger.info(f"📝 Template length: {len(template)} characters")
        logger.info("="*80)
        
        try:
            # Step 1: Get original video duration and properties
            logger.info("📊 Step 1: Analyzing input video...")
            self.get_video_duration(video_path)
            logger.info(f"Video duration: {self.original_duration:.2f}s")
            
            # Step 2: Create personalized transcript
            logger.info("📝 Step 2: Creating personalized transcript...")
            transcript_start = self._log_stage_start("transcript_personalization")
            personalized_text = self.create_personalized_transcript(template, variables)
            logger.info(f"Original template: {template[:100]}...")
            logger.info(f"Personalized text: {personalized_text[:100]}...")
            logger.info(f"Word count: {len(personalized_text.split())} words")
            self._log_stage_end("transcript_personalization", transcript_start)
            
            # Step 3: Generate complete TTS audio
            logger.info("🎵 Step 3: Generating TTS audio...")
            tts_audio_path = self.generate_complete_tts(personalized_text, self.original_duration)
            logger.info(f"TTS audio generated: {tts_audio_path}")
            
            # Step 4: Combine with video (with or without lip sync)
            logger.info("🎬 Step 4: Combining audio with video...")
            video_combine_start = self._log_stage_start("video_combination")
            output_path = self.combine_with_video(video_path, tts_audio_path, apply_lip_sync)
            self._log_stage_end("video_combination", video_combine_start)
            
            # End overall timing
            self.processing_stats["end_time"] = time.time()
            total_duration = self.processing_stats["end_time"] - self.processing_stats["start_time"]
            
            # Log final results and statistics
            logger.info("="*80)
            logger.info("✅ PERSONALIZATION COMPLETE")
            logger.info("="*80)
            logger.info(f"📁 Output: {output_path}")
            logger.info(f"📊 Output size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            logger.info(f"⏱️  Total processing time: {total_duration:.2f}s")
            
            # Log stage breakdown
            logger.info("📈 Processing breakdown:")
            for stage, timing in self.processing_stats["stages"].items():
                if timing["duration"]:
                    percentage = (timing["duration"] / total_duration) * 100
                    logger.info(f"  {stage}: {timing['duration']:.2f}s ({percentage:.1f}%)")
            
            # Log performance metrics
            if self.original_duration:
                processing_speed = self.original_duration / total_duration
                logger.info(f"🚀 Processing speed: {processing_speed:.2f}x real-time")
            
            # Log warnings and errors summary
            if self.processing_stats["warnings"]:
                logger.info(f"⚠️  Warnings: {len(self.processing_stats['warnings'])}")
                for warning in self.processing_stats["warnings"]:
                    logger.info(f"  - {warning}")
            
            if self.processing_stats["errors"]:
                logger.info(f"❌ Errors: {len(self.processing_stats['errors'])}")
            
            logger.info("="*80)
            
            # Save processing stats to file
            self._save_processing_stats(output_path)
            
            return output_path
            
        except Exception as e:
            self.processing_stats["end_time"] = time.time()
            self._log_error(f"Pipeline failed: {str(e)}", e)
            self._save_processing_stats(None, failed=True)
            logger.error(f"❌ Pipeline failed after {time.time() - self.processing_stats['start_time']:.2f}s")
            raise
    
    def _save_processing_stats(self, output_path: Path = None, failed: bool = False):
        """Save processing statistics to JSON file"""
        try:
            import json
            from datetime import datetime
            
            stats = self.processing_stats.copy()
            stats.update({
                "video_duration": self.original_duration,
                "output_path": str(output_path) if output_path else None,
                "failed": failed,
                "timestamp": datetime.now().isoformat(),
                "total_duration": stats["end_time"] - stats["start_time"] if stats["end_time"] else None
            })
            
            stats_file = self.output_dir / "logs" / "processing_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Processing stats saved to: {stats_file}")
            
        except Exception as e:
            self._log_warning(f"Could not save processing stats: {e}")