#!/usr/bin/env python3
"""
Production-ready video personalization pipeline
Automatically finds and replaces variables with API-provided values
"""

import subprocess
import json
import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not available. Install with: pip install gtts")

try:
    from ..lip_sync import lip_sync
    LIPSYNC_AVAILABLE = True
except ImportError:
    LIPSYNC_AVAILABLE = False
    print("Warning: LipSync module not available. Lip sync will be disabled.")


@dataclass
class Variable:
    """Represents a variable to be replaced"""
    original_text: str
    start_time: float
    end_time: float
    replacement_text: str = ""
    occurrence: int = 1


class VideoPersonalizationPipeline:
    def __init__(self, video_path: str, output_dir: str = "output", 
                 enable_lip_sync: bool = True, lip_sync_model: str = "musetalk"):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.enable_lip_sync = enable_lip_sync and LIPSYNC_AVAILABLE
        self.lip_sync_model = lip_sync_model
        
        # Initialize lip sync processor if available
        self.lip_sync_processor = None
        if self.enable_lip_sync:
            try:
                from ..lip_sync.lip_sync import LipSyncProcessor
                logger.info(f"Initializing lip sync with {lip_sync_model} model...")
                self.lip_sync_processor = LipSyncProcessor(model_type=lip_sync_model)
                logger.info(f"✓ Lip sync enabled with {lip_sync_model} model")
            except Exception as e:
                logger.error(f"Failed to initialize lip sync: {e}")
                self.enable_lip_sync = False
        
        # Default variables to search for
        self.search_patterns = {
            "customer_name": ["Anuji", "Anuj ji", "Anuj"],
            "destination": ["Bali"]
        }
        
    def __del__(self):
        """Cleanup temp directory"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def extract_audio(self) -> Path:
        """Extract audio from video for transcription"""
        logger.info("Extracting audio from video...")
        audio_path = self.temp_dir / "audio.wav"
        cmd = [
            "ffmpeg", "-i", str(self.video_path),
            "-ac", "1", "-ar", "16000",
            "-y", str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr.decode()}")
            raise RuntimeError("Failed to extract audio")
        
        # Log audio info
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Audio extracted: {size_mb:.1f} MB")
        return audio_path
    
    def transcribe_with_whisper(self, audio_path: Path) -> Dict:
        """Transcribe audio using Whisper (or mock for testing)"""
        try:
            import whisper
            print("Using Whisper for transcription...")
            model = whisper.load_model("base")
            result = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language="en"
            )
            return result
        except ImportError:
            print("Whisper not installed. Using mock transcription...")
            return self.mock_transcription()
    
    def mock_transcription(self) -> Dict:
        """Mock transcription for testing without Whisper"""
        return {
            "segments": [{
                "start": 0.0,
                "end": 30.0,
                "text": "Hello Anuj ji, welcome to our presentation about Anuj ji's favorite destination Bali.",
                "words": [
                    {"word": "Hello", "start": 0.5, "end": 0.8},
                    {"word": "Anuj", "start": 1.0, "end": 1.3},
                    {"word": "ji", "start": 1.3, "end": 1.5},
                    {"word": "welcome", "start": 1.6, "end": 2.0},
                    {"word": "to", "start": 2.1, "end": 2.2},
                    {"word": "our", "start": 2.3, "end": 2.4},
                    {"word": "presentation", "start": 2.5, "end": 3.0},
                    {"word": "about", "start": 3.1, "end": 3.3},
                    {"word": "Anuj", "start": 21.8, "end": 22.0},
                    {"word": "ji's", "start": 22.0, "end": 22.3},
                    {"word": "favorite", "start": 22.4, "end": 22.8},
                    {"word": "destination", "start": 22.9, "end": 23.4},
                    {"word": "Bali", "start": 23.5, "end": 23.8}
                ]
            }]
        }
    
    def find_variables(self, transcription: Dict) -> List[Variable]:
        """Find all occurrences of variables in transcription"""
        variables = []
        occurrence_count = {}
        
        for segment in transcription.get("segments", []):
            words = segment.get("words", [])
            
            # Search for multi-word patterns
            for var_type, patterns in self.search_patterns.items():
                for pattern in patterns:
                    pattern_words = pattern.split()
                    
                    for i in range(len(words) - len(pattern_words) + 1):
                        # Check if words match pattern
                        match = True
                        for j, pattern_word in enumerate(pattern_words):
                            if i + j >= len(words):
                                match = False
                                break
                            # Clean the word (remove leading/trailing spaces)
                            word_text = words[i + j]["word"].strip()
                            if word_text.lower() != pattern_word.lower():
                                match = False
                                break
                        
                        if match:
                            # Found a match
                            key = f"{var_type}_{pattern}"
                            occurrence_count[key] = occurrence_count.get(key, 0) + 1
                            
                            variable = Variable(
                                original_text=pattern,
                                start_time=words[i]["start"],
                                end_time=words[i + len(pattern_words) - 1]["end"],
                                occurrence=occurrence_count[key]
                            )
                            variables.append(variable)
                            print(f"Found variable: {pattern} at {words[i]['start']:.2f}s")
        
        return variables
    
    def generate_replacement_audio(self, text: str, duration: float) -> Path:
        """Generate replacement audio using TTS"""
        output_path = self.temp_dir / f"replacement_{abs(hash(text))}.wav"
        
        if GTTS_AVAILABLE:
            try:
                # Generate TTS audio
                tts = gTTS(text=text, lang='en', slow=False)
                
                # Save as MP3 first
                mp3_path = self.temp_dir / f"tts_temp_{abs(hash(text))}.mp3"
                tts.save(str(mp3_path))
                
                # Convert to WAV and adjust duration
                # First, get the actual duration of the TTS audio
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(mp3_path)
                ]
                
                tts_duration = float(subprocess.check_output(probe_cmd).decode().strip())
                
                # Calculate tempo adjustment to match target duration
                tempo = tts_duration / duration
                
                # Convert to WAV with tempo adjustment
                if abs(tempo - 1.0) > 0.1:  # Only adjust if significant difference
                    # Use atempo filter (limited to 0.5-2.0 range)
                    tempo = max(0.5, min(2.0, tempo))
                    cmd = [
                        "ffmpeg", "-i", str(mp3_path),
                        "-filter:a", f"atempo={tempo}",
                        "-ar", "48000", "-ac", "1",
                        "-y", str(output_path)
                    ]
                else:
                    # Simple conversion without tempo change
                    cmd = [
                        "ffmpeg", "-i", str(mp3_path),
                        "-ar", "48000", "-ac", "1",
                        "-y", str(output_path)
                    ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Clean up temp MP3
                mp3_path.unlink()
                
                # If duration still doesn't match, pad or trim
                actual_duration = float(subprocess.check_output([
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(output_path)
                ]).decode().strip())
                
                if abs(actual_duration - duration) > 0.05:  # 50ms tolerance
                    # Create final adjusted audio
                    final_path = self.temp_dir / f"final_{abs(hash(text))}.wav"
                    
                    if actual_duration < duration:
                        # Pad with silence
                        pad_duration = duration - actual_duration
                        cmd = [
                            "ffmpeg", "-i", str(output_path),
                            "-filter_complex",
                            f"[0:a]apad=pad_dur={pad_duration}[out]",
                            "-map", "[out]",
                            "-y", str(final_path)
                        ]
                    else:
                        # Trim to exact duration
                        cmd = [
                            "ffmpeg", "-i", str(output_path),
                            "-t", str(duration),
                            "-y", str(final_path)
                        ]
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    output_path.unlink()
                    final_path.rename(output_path)
                
                return output_path
                
            except Exception as e:
                logger.error(f"TTS generation failed: {e}. Falling back to silence.")
        
        # Fallback: create silence with target duration
        cmd = [
            "ffmpeg", "-f", "lavfi",
            "-i", f"anullsrc=r=48000:cl=mono:d={duration}",
            "-y", str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def create_audio_with_replacements(self, variables: List[Variable]) -> Path:
        """Create new audio track with replacements"""
        # Extract original audio at 48kHz
        original_audio = self.temp_dir / "original_audio.wav"
        cmd = [
            "ffmpeg", "-i", str(self.video_path),
            "-ac", "1", "-ar", "48000",
            "-y", str(original_audio)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Sort variables by time
        variables.sort(key=lambda x: x.start_time)
        
        # Create segments
        segments = []
        last_end = 0
        
        for var in variables:
            # Segment before replacement
            if var.start_time > last_end:
                seg_path = self.temp_dir / f"seg_before_{len(segments)}.wav"
                cmd = [
                    "ffmpeg", "-i", str(original_audio),
                    "-ss", str(last_end), "-to", str(var.start_time),
                    "-y", str(seg_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                segments.append(seg_path)
                logger.info(f"Created segment before replacement: {last_end:.2f}s to {var.start_time:.2f}s")
            
            # Replacement audio
            duration = var.end_time - var.start_time
            replacement_audio = self.generate_replacement_audio(
                var.replacement_text, duration
            )
            segments.append(replacement_audio)
            logger.info(f"Created replacement segment: '{var.original_text}' -> '{var.replacement_text}' ({duration:.2f}s)")
            
            last_end = var.end_time
        
        # Final segment
        video_duration = self.get_video_duration()
        if last_end < video_duration:
            seg_path = self.temp_dir / f"seg_final.wav"
            cmd = [
                "ffmpeg", "-i", str(original_audio),
                "-ss", str(last_end),
                "-y", str(seg_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            segments.append(seg_path)
        
        # Concatenate all segments with crossfade
        return self.concatenate_audio_segments(segments)
    
    def concatenate_audio_segments(self, segments: List[Path]) -> Path:
        """Concatenate audio segments with crossfade"""
        output_path = self.temp_dir / "replaced_audio.wav"
        
        if len(segments) == 1:
            shutil.copy(segments[0], output_path)
            return output_path
        
        # Create concat file
        concat_file = self.temp_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for i, seg in enumerate(segments):
                f.write(f"file '{seg}'\n")
                logger.info(f"Segment {i}: {seg.name}")
        
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-y", str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def get_video_duration(self) -> float:
        """Get video duration in seconds"""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    
    def create_final_video(self, audio_path: Path, variables: List[Variable], show_overlay: bool = False) -> Path:
        """Create final video with replaced audio and optional lip sync"""
        output_path = self.output_dir / f"personalized_{self.video_path.stem}.mp4"
        
        # If lip sync is enabled and we have variables to replace
        if self.enable_lip_sync and self.lip_sync_processor and variables:
            logger.info(f"Applying {self.lip_sync_model} lip sync to {len(variables)} segments...")
            
            # Prepare segments for lip sync
            lip_sync_segments = []
            for var in variables:
                # Get the audio segment for this variable
                segment_audio = self.temp_dir / f"segment_{var.start_time}_{var.end_time}.wav"
                
                # Extract the specific audio segment
                cmd = [
                    "ffmpeg", "-i", str(audio_path),
                    "-ss", str(var.start_time),
                    "-to", str(var.end_time),
                    "-y", str(segment_audio)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                lip_sync_segments.append({
                    "start": var.start_time,
                    "end": var.end_time,
                    "audio": str(segment_audio)
                })
            
            # Apply lip sync
            temp_output = self.temp_dir / "temp_lipsync.mp4"
            success = self.lip_sync_processor.apply_lip_sync_simple(
                str(self.video_path),
                lip_sync_segments,
                str(temp_output)
            )
            
            if success:
                # Replace audio in lip-synced video
                cmd = [
                    "ffmpeg", "-i", str(temp_output), "-i", str(audio_path),
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-y", str(output_path)
                ]
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0:
                    logger.info("✓ Lip sync applied successfully")
                    return output_path
                else:
                    logger.error(f"Final mux failed: {result.stderr.decode()}")
            else:
                logger.warning("Lip sync failed, falling back to audio-only replacement")
        
        # Fallback: Original implementation without lip sync
        if show_overlay and variables:
            # Build filter for visual indicators
            filter_parts = []
            for var in variables:
                filter_parts.append(
                    f"drawtext="
                    f"text='[{var.replacement_text}]':"
                    f"fontcolor=green:fontsize=40:"
                    f"box=1:boxcolor=black@0.8:"
                    f"x=(w-text_w)/2:y=h-100:"
                    f"enable='between(t,{var.start_time},{var.end_time})'"
                )
            
            filter_str = ",".join(filter_parts)
            cmd = [
                "ffmpeg", "-i", str(self.video_path), "-i", str(audio_path),
                "-vf", filter_str,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                "-y", str(output_path)
            ]
        else:
            # No visual overlay
            cmd = [
                "ffmpeg", "-i", str(self.video_path), "-i", str(audio_path),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-y", str(output_path)
            ]
        
        subprocess.run(cmd, check=True)
        return output_path
    
    def process(self, replacements: Dict[str, str], show_overlay: bool = False) -> Path:
        """Main processing pipeline"""
        logger.info(f"Starting video personalization pipeline")
        logger.info(f"Input video: {self.video_path}")
        logger.info(f"Replacements: {replacements}")
        logger.info(f"Lip sync: {'Enabled' if self.enable_lip_sync else 'Disabled'}")
        
        start_time = time.time()
        
        # Step 1: Extract audio
        print("1. Extracting audio...")
        audio_path = self.extract_audio()
        
        # Step 2: Transcribe
        print("2. Transcribing audio...")
        transcription = self.transcribe_with_whisper(audio_path)
        
        # Save transcription
        with open(self.output_dir / "transcription.json", "w") as f:
            json.dump(transcription, f, indent=2)
        
        # Step 3: Find variables
        print("3. Finding variables...")
        variables = self.find_variables(transcription)
        
        if not variables:
            print("No variables found!")
            return self.video_path
        
        print(f"Found {len(variables)} variables:")
        for var in variables:
            print(f"  - '{var.original_text}' at {var.start_time:.2f}-{var.end_time:.2f}s")
        
        # Step 4: Apply replacements
        print("4. Applying replacements...")
        for var in variables:
            # Match variable to replacement
            for var_type, patterns in self.search_patterns.items():
                if var.original_text in patterns:
                    var.replacement_text = replacements.get(var_type, var.original_text)
                    break
        
        # Save variable report
        report = {
            "video": str(self.video_path),
            "variables_found": [
                {
                    "original": var.original_text,
                    "replacement": var.replacement_text,
                    "start": var.start_time,
                    "end": var.end_time,
                    "occurrence": var.occurrence
                }
                for var in variables
            ]
        }
        with open(self.output_dir / "replacement_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Step 5: Create replacement audio
        print("5. Creating replacement audio...")
        new_audio = self.create_audio_with_replacements(variables)
        
        # Step 6: Create final video
        print("6. Creating final video...")
        final_video = self.create_final_video(new_audio, variables, show_overlay)
        
        # Log completion
        total_time = time.time() - start_time
        output_size = final_video.stat().st_size / (1024 * 1024)
        
        logger.info("=" * 60)
        logger.info(f"✓ Video personalization complete!")
        logger.info(f"  Output: {final_video}")
        logger.info(f"  Size: {output_size:.1f} MB")
        logger.info(f"  Processing time: {total_time:.1f}s")
        logger.info(f"  Model used: {self.lip_sync_model if self.enable_lip_sync else 'Audio only'}")
        logger.info("=" * 60)
        
        return final_video


def main():
    parser = argparse.ArgumentParser(description="Video Personalization Pipeline")
    parser.add_argument("video", nargs='?', help="Input video file")
    parser.add_argument("--customer-name", default="John", help="Customer name replacement")
    parser.add_argument("--destination", default="Paris", help="Destination replacement")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--show-overlay", action="store_true", help="Show visual overlay")
    parser.add_argument("--no-lip-sync", action="store_true", help="Disable lip sync")
    parser.add_argument("--lip-sync-model", default="musetalk", 
                       choices=["musetalk", "wav2lip", "latentsync"],
                       help="Lip sync model to use (default: musetalk)")
    parser.add_argument("--list-models", action="store_true", 
                       help="List available lip sync models and exit")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    try:
        from logging_config import setup_logging
        setup_logging(log_level=args.log_level, verbose=args.verbose)
    except ImportError:
        # Fallback to basic logging
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    # List models if requested
    if args.list_models:
        if LIPSYNC_AVAILABLE:
            from ..lip_sync.lip_sync import print_model_comparison
            print_model_comparison()
        else:
            print("Lip sync module not available. Install dependencies with:")
            print("pip install -r requirements.txt")
        return
    
    # Check if video argument is provided
    if not args.video:
        parser.error("Video file is required unless using --list-models")
        return
    
    # Create pipeline
    pipeline = VideoPersonalizationPipeline(
        args.video, 
        args.output_dir,
        enable_lip_sync=not args.no_lip_sync,
        lip_sync_model=args.lip_sync_model
    )
    
    # Set replacements from API/arguments
    replacements = {
        "customer_name": args.customer_name,
        "destination": args.destination
    }
    
    # Process video
    output_video = pipeline.process(replacements, args.show_overlay)
    
    print(f"\nPersonalized video created: {output_video}")


if __name__ == "__main__":
    main()