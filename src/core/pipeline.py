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
from typing import List, Dict, Tuple, Optional
import re
import logging
import time
import numpy as np
import librosa

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
    from TTS.api import TTS as CoquiTTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False
    # print("Warning: Coqui TTS not available. Voice cloning disabled.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available. Install with: pip install soundfile")

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
            "customer_name": ["Anurji", "Anuji", "Anuj ji", "Anuj"],  # Added "Anurji" as Whisper transcribes it
            "destination": ["Bali"]
        }
        
        # Initialize voice cloning TTS if available
        self.tts_model = None
        self.speaker_wav = None
        self.speaker_features = None  # Store speaker voice features for conversion
        if COQUI_TTS_AVAILABLE:
            try:
                # Use XTTS-v2 for high quality voice cloning
                self.tts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
                logger.info("Initialized Coqui TTS with XTTS-v2 for voice cloning")
            except Exception as e:
                logger.warning(f"Failed to initialize Coqui TTS: {e}")
                self.tts_model = None
        
    def __del__(self):
        """Cleanup temp directory"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def extract_voice_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract voice features for voice conversion"""
        try:
            # Extract pitch contour
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # Get average pitch (excluding unvoiced segments)
            valid_f0 = f0[~np.isnan(f0)]
            avg_pitch = np.mean(valid_f0) if len(valid_f0) > 0 else 150.0
            
            # Extract spectral features (timbre)
            n_fft = 2048
            hop_length = n_fft // 4
            
            # Compute spectral envelope using cepstral analysis
            spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            
            # Get mel-frequency cepstral coefficients for timbre
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Compute average spectral envelope
            avg_spectrum = np.mean(spec, axis=1)
            
            return {
                'pitch': avg_pitch,
                'pitch_contour': f0,
                'spectral_envelope': avg_spectrum,
                'mfcc': mfcc,
                'sample_rate': sr
            }
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {
                'pitch': 150.0,
                'pitch_contour': None,
                'spectral_envelope': None,
                'mfcc': None,
                'sample_rate': sr
            }
    
    def apply_voice_conversion(self, audio: np.ndarray, sr: int, target_features: Dict) -> np.ndarray:
        """Apply voice conversion to match target speaker characteristics"""
        try:
            if target_features is None or target_features.get('spectral_envelope') is None:
                return audio
            
            # Apply spectral envelope conversion
            n_fft = 2048
            hop_length = n_fft // 4
            
            # STFT of input audio
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            spec = np.abs(stft)
            phase = np.angle(stft)
            
            # Get current spectral envelope
            current_envelope = np.mean(spec, axis=1)
            
            # Compute spectral transfer function
            target_envelope = target_features['spectral_envelope']
            
            # Ensure envelopes have same length
            min_len = min(len(current_envelope), len(target_envelope))
            current_envelope = current_envelope[:min_len]
            target_envelope = target_envelope[:min_len]
            
            # Compute transfer function with smoothing
            transfer = target_envelope / (current_envelope + 1e-10)
            transfer = np.clip(transfer, 0.1, 10.0)  # Limit extreme values
            
            # Apply Gaussian smoothing to transfer function
            from scipy.ndimage import gaussian_filter1d
            transfer_smooth = gaussian_filter1d(transfer, sigma=5)
            
            # Apply transfer function to spectrum
            spec_converted = spec[:min_len, :] * transfer_smooth[:, np.newaxis]
            
            # Reconstruct with converted spectrum
            stft_converted = spec_converted * np.exp(1j * phase[:min_len, :])
            audio_converted = librosa.istft(stft_converted, hop_length=hop_length)
            
            # Apply pitch shift if needed
            if target_features.get('pitch') and target_features['pitch'] > 0:
                # Calculate pitch shift in semitones
                current_pitch = self.estimate_pitch(audio, sr)
                if current_pitch > 0:
                    semitones = 12 * np.log2(target_features['pitch'] / current_pitch)
                    semitones = np.clip(semitones, -12, 12)  # Limit to 1 octave
                    
                    if abs(semitones) > 0.5:
                        audio_converted = librosa.effects.pitch_shift(
                            audio_converted, sr=sr, n_steps=semitones
                        )
            
            return audio_converted
            
        except Exception as e:
            logger.warning(f"Voice conversion failed: {e}")
            return audio
    
    def estimate_pitch(self, audio: np.ndarray, sr: int) -> float:
        """Estimate average pitch of audio"""
        try:
            f0, _, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            valid_f0 = f0[~np.isnan(f0)]
            return np.mean(valid_f0) if len(valid_f0) > 0 else 0
        except:
            return 0
    
    def wsola_stretch(self, audio: np.ndarray, stretch_factor: float, sr: int) -> np.ndarray:
        """Apply WSOLA (Waveform Similarity Overlap-Add) time stretching"""
        try:
            import librosa
            # Use librosa's phase vocoder as a good approximation of WSOLA
            stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
            return stretched
        except Exception as e:
            logger.warning(f"WSOLA stretching failed: {e}")
            # Simple resampling fallback
            target_length = int(len(audio) / stretch_factor)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)
    
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
            # Use 'small' or 'medium' model for better accuracy
            # 'small' = 244M params, 'medium' = 769M params, 'large' = 1550M params
            model_size = "small"  # Can be upgraded to "medium" or "large" for even better accuracy
            print(f"Loading Whisper {model_size} model for better accuracy...")
            model = whisper.load_model(model_size)
            
            # Transcribe with more precise settings
            result = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language="en",
                fp16=False,  # Use full precision for better accuracy
                condition_on_previous_text=True,  # Better context understanding
                temperature=0.0,  # Deterministic output
                no_speech_threshold=0.6,  # Adjust sensitivity
                logprob_threshold=-1.0,  # Keep all predictions
                compression_ratio_threshold=2.4,
                verbose=True  # Show progress
            )
            
            # Log transcription quality metrics
            print(f"\nTranscription complete:")
            print(f"- Total segments: {len(result.get('segments', []))}")
            total_words = sum(len(seg.get('words', [])) for seg in result.get('segments', []))
            print(f"- Total words: {total_words}")
            
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
        """Find all occurrences of variables in transcription with improved matching"""
        variables = []
        occurrence_count = {}
        
        for segment in transcription.get("segments", []):
            words = segment.get("words", [])
            
            # First pass: find exact multi-word patterns
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
                            # Clean the word (remove leading/trailing spaces and punctuation)
                            word_text = words[i + j]["word"].strip()
                            # Remove common punctuation for matching
                            word_text_clean = word_text.rstrip('.,;:!?').lstrip('.,;:!?')
                            if word_text_clean.lower() != pattern_word.lower():
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
            
            # Second pass: handle special cases like possessives
            # Look for patterns followed by possessive markers ('s)
            for var_type, patterns in self.search_patterns.items():
                for pattern in patterns:
                    pattern_words = pattern.split()
                    
                    for i in range(len(words) - len(pattern_words)):
                        # Check if base pattern matches
                        match = True
                        for j, pattern_word in enumerate(pattern_words):
                            if i + j >= len(words):
                                match = False
                                break
                            word_text = words[i + j]["word"].strip()
                            # Remove common punctuation for matching
                            word_text_clean = word_text.rstrip('.,;:!?').lstrip('.,;:!?')
                            if word_text_clean.lower() != pattern_word.lower():
                                match = False
                                break
                        
                        if match and i + len(pattern_words) < len(words):
                            # Check if next word is possessive
                            next_word = words[i + len(pattern_words)]["word"].strip().lower()
                            if next_word in ["'s", "s", "'s", "'s"]:
                                # Found possessive form
                                key = f"{var_type}_{pattern}_possessive"
                                occurrence_count[key] = occurrence_count.get(key, 0) + 1
                                
                                variable = Variable(
                                    original_text=pattern,  # Keep original pattern for replacement
                                    start_time=words[i]["start"],
                                    end_time=words[i + len(pattern_words) - 1]["end"],  # Don't include 's
                                    occurrence=occurrence_count[key]
                                )
                                variables.append(variable)
                                print(f"Found possessive variable: {pattern} ('s) at {words[i]['start']:.2f}s")
        
        # Sort by time and log summary
        variables.sort(key=lambda x: x.start_time)
        print(f"\nTotal variables found: {len(variables)}")
        for var in variables:
            print(f"  - '{var.original_text}' at {var.start_time:.2f}-{var.end_time:.2f}s")
        
        return variables
    
    def extract_speaker_reference(self, audio_path: Path, duration: float = 10.0) -> Path:
        """Extract a clean speaker reference from the original audio and analyze voice features"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Find a clean segment without silence
            # Use first 10 seconds or available duration
            ref_duration = min(duration, len(audio) / sr)
            ref_samples = int(ref_duration * sr)
            
            # Find segment with highest energy (likely speech)
            hop_length = 512
            rms = librosa.feature.rms(y=audio[:ref_samples], hop_length=hop_length)[0]
            
            # Get the segment with highest average energy
            window_size = int(3.0 * sr / hop_length)  # 3 second window
            best_start = 0
            best_energy = 0
            
            for i in range(len(rms) - window_size):
                window_energy = np.mean(rms[i:i+window_size])
                if window_energy > best_energy:
                    best_energy = window_energy
                    best_start = i
            
            # Extract the best segment
            start_sample = best_start * hop_length
            end_sample = start_sample + int(3.0 * sr)
            speaker_ref = audio[start_sample:end_sample]
            
            # Extract speaker features for voice conversion
            self.speaker_features = self.extract_voice_features(speaker_ref, sr)
            
            # Save reference audio
            ref_path = self.temp_dir / "speaker_reference.wav"
            sf.write(ref_path, speaker_ref, sr)
            
            logger.info(f"Extracted speaker reference: {ref_path}")
            logger.info(f"Extracted voice features: pitch={self.speaker_features['pitch']:.1f}Hz, timbre shape")
            return ref_path
            
        except Exception as e:
            logger.error(f"Failed to extract speaker reference: {e}")
            return None
    
    def generate_replacement_audio(self, text: str, duration: float, context_audio: Optional[np.ndarray] = None, sr: int = 48000) -> Path:
        """Generate replacement audio using voice cloning or TTS"""
        output_path = self.temp_dir / f"replacement_{abs(hash(text))}.wav"
        
        # Try voice cloning first if available
        if self.tts_model is not None and self.speaker_wav is not None:
            try:
                logger.info(f"Generating '{text}' with voice cloning...")
                
                # Generate with voice cloning
                self.tts_model.tts_to_file(
                    text=text,
                    speaker_wav=str(self.speaker_wav),
                    language="en",
                    file_path=str(output_path)
                )
                
                # Load and apply WSOLA time stretching to match duration
                generated, gen_sr = librosa.load(output_path, sr=sr)
                current_duration = len(generated) / gen_sr
                
                if abs(current_duration - duration) > 0.05:  # 50ms tolerance
                    # Apply WSOLA time stretching
                    stretch_factor = current_duration / duration
                    stretched = self.wsola_stretch(generated, stretch_factor, gen_sr)
                    
                    # Ensure exact duration
                    target_samples = int(duration * gen_sr)
                    if len(stretched) > target_samples:
                        stretched = stretched[:target_samples]
                        # Apply fade out
                        fade_samples = int(0.01 * gen_sr)
                        stretched[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                    elif len(stretched) < target_samples:
                        # Pad with silence
                        stretched = np.pad(stretched, (0, target_samples - len(stretched)))
                    
                    # Save stretched audio
                    sf.write(output_path, stretched, gen_sr)
                
                logger.info(f"✓ Voice cloning successful for '{text}'")
                return output_path
                
            except Exception as e:
                logger.warning(f"Voice cloning failed: {e}, falling back to gTTS")
        
        # Fallback to gTTS
        if GTTS_AVAILABLE:
            try:
                # Analyze context for prosody if provided
                speaking_rate = 1.0
                if context_audio is not None and len(context_audio) > 0:
                    try:
                        import parselmouth
                        sound = parselmouth.Sound(context_audio, sampling_frequency=sr)
                        
                        # Extract pitch to estimate speaking rate
                        pitch = sound.to_pitch()
                        pitch_values = pitch.selected_array['frequency']
                        voiced_frames = np.sum(pitch_values > 0)
                        
                        # Rough estimate: more voiced frames = slower speech
                        if voiced_frames > 0:
                            expected_voiced = len(pitch_values) * 0.6  # typical voicing ratio
                            rate_factor = expected_voiced / voiced_frames
                            speaking_rate = np.clip(rate_factor, 0.8, 1.2)
                    except:
                        pass
                
                # Generate TTS audio with adjusted rate
                slow = speaking_rate < 0.95
                tts = gTTS(text=text, lang='en', slow=slow)
                
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
                
                # Apply voice conversion using speaker features
                if self.speaker_features is not None:
                    try:
                        import librosa
                        # Load the generated audio
                        generated, _ = librosa.load(output_path, sr=sr)
                        
                        # Apply full voice conversion to match original speaker
                        converted = self.apply_voice_conversion(generated, sr, self.speaker_features)
                        
                        # Additional spectral matching if context provided
                        if context_audio is not None and len(context_audio) > 0:
                            # Extract spectral characteristics from context
                            n_fft = 2048
                            # Context spectrum
                            spec_context = np.abs(librosa.stft(context_audio, n_fft=n_fft))
                            context_centroid = np.mean(spec_context, axis=1)
                            
                            # Converted spectrum
                            spec_conv = np.abs(librosa.stft(converted, n_fft=n_fft))
                            conv_centroid = np.mean(spec_conv, axis=1)
                            
                            # Compute residual transfer function
                            min_len = min(len(context_centroid), len(conv_centroid))
                            transfer = context_centroid[:min_len] / (conv_centroid[:min_len] + 1e-10)
                            transfer = np.clip(transfer, 0.5, 2.0)  # Less aggressive for residual
                            
                            # Smooth transfer function
                            from scipy.ndimage import gaussian_filter1d
                            transfer_smooth = gaussian_filter1d(transfer, sigma=5)
                            
                            # Apply residual spectral shaping
                            spec_shaped = spec_conv[:min_len, :] * transfer_smooth[:, np.newaxis]
                            phase = np.angle(librosa.stft(converted, n_fft=n_fft))
                            stft_shaped = spec_shaped * np.exp(1j * phase[:min_len, :])
                            converted = librosa.istft(stft_shaped)
                        
                        # Save converted audio
                        sf.write(str(output_path), converted, sr)
                        logger.info(f"✓ Voice conversion applied to '{text}'")
                    except Exception as e:
                        logger.warning(f"Voice conversion failed: {e}")
                
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
        
        # Load original audio for adaptive buffer calculation
        original_audio_data, _ = sf.read(str(original_audio))
        
        for var in variables:
            # Calculate adaptive buffer time based on speech energy
            try:
                from ..core.audio_processing_enhanced import EnhancedAudioProcessor
                processor = EnhancedAudioProcessor()
                pre_buffer, post_buffer = processor.adaptive_buffer_time(
                    original_audio_data, var.start_time, var.end_time, base_buffer=0.05
                )
            except:
                # Fallback to fixed buffer
                pre_buffer = post_buffer = 0.05
            
            # Segment before replacement
            cut_point = max(0, var.start_time - pre_buffer)
            if cut_point > last_end:
                seg_path = self.temp_dir / f"seg_before_{len(segments)}.wav"
                cmd = [
                    "ffmpeg", "-i", str(original_audio),
                    "-ss", str(last_end), "-to", str(cut_point),
                    "-y", str(seg_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                segments.append(seg_path)
                logger.info(f"Created segment before replacement: {last_end:.2f}s to {cut_point:.2f}s")
            
            # Extract context audio for prosody matching
            context_start = max(0, var.start_time - 1.0)  # 1 second before
            context_end = min(len(original_audio_data) / 48000, var.end_time + 1.0)  # 1 second after
            context_start_idx = int(context_start * 48000)
            context_end_idx = int(context_end * 48000)
            context_audio = original_audio_data[context_start_idx:context_end_idx]
            
            # Replacement audio (with adaptive buffer and context)
            duration = var.end_time - var.start_time + pre_buffer + post_buffer
            replacement_audio = self.generate_replacement_audio(
                var.replacement_text, duration, context_audio
            )
            segments.append(replacement_audio)
            logger.info(f"Created replacement segment: '{var.original_text}' -> '{var.replacement_text}' ({duration:.2f}s)")
            
            last_end = var.end_time + post_buffer
        
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
        """Concatenate audio segments with enhanced crossfade and spectral smoothing"""
        output_path = self.temp_dir / "replaced_audio.wav"
        
        if len(segments) == 1:
            shutil.copy(segments[0], output_path)
            return output_path
        
        try:
            import soundfile as sf
            import numpy as np
            from scipy.signal import windows, butter, filtfilt
            import librosa
            
            # Load all segments
            audio_segments = []
            sample_rate = None
            for seg_path in segments:
                audio, sr = sf.read(str(seg_path))
                if sample_rate is None:
                    sample_rate = sr
                audio_segments.append(audio)
            
            # Define fade parameters - INCREASED to 100ms for better continuity
            fade_duration_ms = 100  # 100ms crossfade (doubled from 50ms)
            fade_samples = int(fade_duration_ms * sample_rate / 1000)
            
            # Apply standard crossfade between segments
            # Process segments with crossfade
            processed_segments = []
            for i in range(len(audio_segments)):
                if i == 0:
                    # First segment - only fade out at end
                    seg = audio_segments[i].copy()
                    if len(seg) > fade_samples and i < len(audio_segments) - 1:
                        # Use raised cosine window for smoother fade
                        fade_curve = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))
                        seg[-fade_samples:] *= fade_curve
                    processed_segments.append(seg)
                elif i == len(audio_segments) - 1:
                    # Last segment - only fade in at start
                    seg = audio_segments[i].copy()
                    if len(seg) > fade_samples:
                        # Use raised cosine window for smoother fade
                        fade_curve = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
                        seg[:fade_samples] *= fade_curve
                    processed_segments.append(seg)
                else:
                    # Middle segments - fade both ends
                    seg = audio_segments[i].copy()
                    if len(seg) > 2 * fade_samples:
                        # Use raised cosine windows
                        fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
                        fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))
                        seg[:fade_samples] *= fade_in
                        seg[-fade_samples:] *= fade_out
                    processed_segments.append(seg)
            
            # Apply spectral envelope matching BEFORE concatenation
            for i in range(1, len(processed_segments)):
                # Get boundary regions
                prev_seg = processed_segments[i-1]
                curr_seg = processed_segments[i]
                
                # Extract spectral characteristics from boundary
                if len(prev_seg) > fade_samples * 2 and len(curr_seg) > fade_samples * 2:
                    # Analyze spectral envelope of the boundary region
                    boundary_before = prev_seg[-fade_samples*2:]
                    boundary_after = curr_seg[:fade_samples*2]
                    
                    # Compute spectral envelopes using cepstral analysis
                    n_fft = 2048
                    hop_length = n_fft // 4
                    
                    # Get spectral envelope of original voice
                    spec_before = np.abs(librosa.stft(boundary_before, n_fft=n_fft, hop_length=hop_length))
                    spec_after = np.abs(librosa.stft(boundary_after, n_fft=n_fft, hop_length=hop_length))
                    
                    # Smooth spectral envelopes
                    from scipy.ndimage import gaussian_filter1d
                    spec_before_smooth = gaussian_filter1d(spec_before, sigma=3, axis=0)
                    spec_after_smooth = gaussian_filter1d(spec_after, sigma=3, axis=0)
                    
                    # Compute average spectral shape for voice conversion
                    avg_spectrum = (spec_before_smooth.mean(axis=1) + spec_after_smooth.mean(axis=1)) / 2
                    
                    # Apply spectral shaping to the replacement segment
                    if i % 2 == 0:  # This is likely a replacement segment
                        # Get spectrum of replacement
                        spec_replacement = np.abs(librosa.stft(curr_seg, n_fft=n_fft, hop_length=hop_length))
                        
                        # Compute spectral transfer function
                        replacement_avg = spec_replacement.mean(axis=1)
                        transfer_function = avg_spectrum / (replacement_avg + 1e-10)
                        transfer_function = np.clip(transfer_function, 0.5, 2.0)
                        
                        # Apply transfer function with smoothing
                        transfer_smooth = gaussian_filter1d(transfer_function, sigma=5)
                        spec_shaped = spec_replacement * transfer_smooth[:, np.newaxis]
                        
                        # Reconstruct audio with shaped spectrum
                        phase = np.angle(librosa.stft(curr_seg, n_fft=n_fft, hop_length=hop_length))
                        stft_shaped = spec_shaped * np.exp(1j * phase)
                        curr_seg_shaped = librosa.istft(stft_shaped, hop_length=hop_length)
                        
                        # Ensure same length
                        if len(curr_seg_shaped) != len(curr_seg):
                            curr_seg_shaped = librosa.util.fix_length(curr_seg_shaped, size=len(curr_seg))
                        
                        # Blend shaped and original (70% shaped for voice matching)
                        processed_segments[i] = 0.7 * curr_seg_shaped + 0.3 * curr_seg
            
            # Concatenate with overlap and enhanced blending
            output_audio = []
            for i, seg in enumerate(processed_segments):
                if i == 0:
                    output_audio = seg
                else:
                    # Overlap the faded regions with spectral smoothing
                    if len(output_audio) >= fade_samples and len(seg) >= fade_samples:
                        # Extract overlap regions
                        overlap_end = output_audio[-fade_samples:]
                        overlap_start = seg[:fade_samples]
                        
                        # Apply logarithmic crossfade for more natural perception
                        t = np.linspace(0, 1, fade_samples)
                        # Logarithmic fade curves
                        fade_out = np.log10(10 - 9*t) / np.log10(10)
                        fade_in = np.log10(1 + 9*t) / np.log10(10)
                        
                        # Mix with logarithmic fades
                        overlap = overlap_end * fade_out + overlap_start * fade_in
                        
                        output_audio = np.concatenate([
                            output_audio[:-fade_samples],
                            overlap,
                            seg[fade_samples:] if len(seg) > fade_samples else []
                        ])
                    else:
                        output_audio = np.concatenate([output_audio, seg])
            
            # Apply final spectral smoothing at transition points
            # Find approximate transition points based on segment lengths
            transition_points = []
            current_pos = 0
            for i in range(len(audio_segments) - 1):
                current_pos += len(audio_segments[i])
                transition_points.append(current_pos)
            
            # Apply spectral smoothing around transitions
            window_size = int(0.2 * sample_rate)  # 200ms window
            for trans_point in transition_points:
                start = max(0, trans_point - window_size)
                end = min(len(output_audio), trans_point + window_size)
                
                if end - start > 1024:
                    # Extract transition region
                    transition_audio = output_audio[start:end]
                    
                    # Apply spectral smoothing
                    D = librosa.stft(transition_audio, n_fft=2048)
                    mag = np.abs(D)
                    phase = np.angle(D)
                    
                    # Smooth magnitude spectrum
                    from scipy.ndimage import uniform_filter1d
                    mag_smooth = uniform_filter1d(mag, size=5, axis=1)
                    
                    # Reconstruct
                    D_smooth = mag_smooth * np.exp(1j * phase)
                    transition_smooth = librosa.istft(D_smooth)
                    
                    # Ensure same length
                    if len(transition_smooth) != len(transition_audio):
                        transition_smooth = librosa.util.fix_length(transition_smooth, size=len(transition_audio))
                    
                    # Replace with smoothed version
                    output_audio[start:end] = transition_smooth
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(output_audio))
            if max_val > 0.95:
                output_audio = output_audio * 0.95 / max_val
            
            # Save the blended audio
            sf.write(str(output_path), output_audio, sample_rate)
            logger.info("Audio segments blended with enhanced spectral matching and 100ms crossfade")
            
        except ImportError:
            logger.warning("soundfile not available, falling back to simple concatenation")
            # Fallback to original method
            concat_file = self.temp_dir / "concat.txt"
            with open(concat_file, "w") as f:
                for i, seg in enumerate(segments):
                    f.write(f"file '{seg}'\n")
                    logger.info(f"Segment {i}: {seg.name}")
            
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-af", "aresample=async=1:first_pts=0",
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
            success = self.lip_sync_processor.apply_lip_sync(
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
        
        # Extract speaker reference for voice cloning if TTS model is available
        if self.tts_model is not None:
            # Extract original audio at full quality
            full_audio_path = self.temp_dir / "full_audio.wav"
            cmd = [
                "ffmpeg", "-i", str(self.video_path),
                "-ac", "1", "-ar", "48000",
                "-y", str(full_audio_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Extract speaker reference
            self.speaker_wav = self.extract_speaker_reference(full_audio_path)
            if self.speaker_wav:
                logger.info("✓ Speaker reference extracted for voice cloning")
        
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