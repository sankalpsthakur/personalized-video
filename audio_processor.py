#!/usr/bin/env python3
"""
Audio processing module for production-quality replacements
Implements spectral cross-fade and LUFS normalization
"""

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from pathlib import Path
from typing import Tuple, Optional
import subprocess
import json


class AudioProcessor:
    def __init__(self, target_lufs: float = -16.0):
        """
        Initialize audio processor
        
        Args:
            target_lufs: Target loudness in LUFS (broadcast standard)
        """
        self.target_lufs = target_lufs
        self.meter = pyln.Meter(48000)  # Standard broadcast sample rate
        
    def extract_audio_segment(self, 
                            audio_path: Path, 
                            start_time: float, 
                            end_time: float,
                            padding: float = 0.05) -> Tuple[np.ndarray, int]:
        """
        Extract audio segment with padding for cross-fade
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            padding: Padding in seconds for cross-fade
            
        Returns:
            Audio data and sample rate
        """
        # Load with padding
        offset = max(0, start_time - padding)
        duration = (end_time - start_time) + (2 * padding)
        
        audio, sr = librosa.load(
            str(audio_path),
            sr=48000,
            offset=offset,
            duration=duration,
            mono=True
        )
        
        return audio, sr
    
    def analyze_spectral_characteristics(self, audio: np.ndarray, sr: int) -> dict:
        """
        Analyze spectral characteristics for matching
        
        Returns:
            Dictionary with spectral features
        """
        # Compute spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Compute MFCC for timbre matching
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'mfcc_mean': np.mean(mfcc, axis=1),
            'loudness_lufs': self.meter.integrated_loudness(audio)
        }
    
    def spectral_crossfade(self,
                          audio1: np.ndarray,
                          audio2: np.ndarray,
                          audio3: np.ndarray,
                          fade_duration: float = 0.05,
                          sr: int = 48000) -> np.ndarray:
        """
        Apply spectral cross-fade between three segments
        
        Args:
            audio1: Before segment
            audio2: Replacement segment
            audio3: After segment
            fade_duration: Cross-fade duration in seconds
            sr: Sample rate
            
        Returns:
            Cross-faded audio
        """
        fade_samples = int(fade_duration * sr)
        
        # Apply fade out to audio1
        if len(audio1) > fade_samples:
            fade_out = np.linspace(1, 0, fade_samples)
            audio1[-fade_samples:] *= fade_out
        
        # Apply fade in/out to audio2
        if len(audio2) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio2[:fade_samples] *= fade_in
            audio2[-fade_samples:] *= fade_out
        
        # Apply fade in to audio3
        if len(audio3) > fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            audio3[:fade_samples] *= fade_in
        
        # Overlap and add
        result = np.zeros(len(audio1) + len(audio2) + len(audio3) - 2 * fade_samples)
        
        # Place audio1
        result[:len(audio1)] = audio1
        
        # Overlap audio2 with audio1
        overlap_start = len(audio1) - fade_samples
        result[overlap_start:overlap_start + len(audio2)] += audio2
        
        # Overlap audio3 with audio2
        overlap_start = len(audio1) + len(audio2) - 2 * fade_samples
        result[overlap_start:] = audio3
        
        return result
    
    def match_audio_characteristics(self,
                                  replacement_audio: np.ndarray,
                                  reference_features: dict,
                                  sr: int = 48000) -> np.ndarray:
        """
        Match replacement audio to reference characteristics
        
        Args:
            replacement_audio: Audio to process
            reference_features: Target spectral features
            sr: Sample rate
            
        Returns:
            Processed audio
        """
        # Apply spectral shaping using parametric EQ
        # This is a simplified version - production would use more sophisticated processing
        
        # Analyze replacement
        replacement_features = self.analyze_spectral_characteristics(replacement_audio, sr)
        
        # Calculate spectral tilt correction
        centroid_ratio = reference_features['spectral_centroid_mean'] / replacement_features['spectral_centroid_mean']
        
        # Apply gentle high-frequency boost/cut
        if centroid_ratio > 1.2:  # Needs brightening
            # Apply high shelf filter
            from scipy import signal
            sos = signal.butter(2, 8000, 'highpass', fs=sr, output='sos')
            high_freq = signal.sosfilt(sos, replacement_audio)
            replacement_audio = replacement_audio + 0.2 * high_freq
        elif centroid_ratio < 0.8:  # Needs darkening
            # Apply low-pass filter
            from scipy import signal
            sos = signal.butter(2, 8000, 'lowpass', fs=sr, output='sos')
            replacement_audio = signal.sosfilt(sos, replacement_audio)
        
        return replacement_audio
    
    def normalize_loudness(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """
        Normalize audio to target LUFS
        
        Args:
            audio: Input audio
            sr: Sample rate
            
        Returns:
            Normalized audio
        """
        # Measure current loudness
        loudness = self.meter.integrated_loudness(audio)
        
        # Calculate required gain
        loudness_delta = self.target_lufs - loudness
        gain_linear = 10 ** (loudness_delta / 20)
        
        # Apply gain with limiting to prevent clipping
        normalized = audio * gain_linear
        
        # Soft limiting if needed
        if np.max(np.abs(normalized)) > 0.99:
            from scipy import signal
            # Simple soft clipper
            threshold = 0.9
            normalized = np.where(
                normalized > threshold,
                threshold + (normalized - threshold) * 0.3,
                normalized
            )
            normalized = np.where(
                normalized < -threshold,
                -threshold + (normalized + threshold) * 0.3,
                normalized
            )
        
        return normalized
    
    def process_replacement(self,
                          original_audio_path: Path,
                          replacement_audio_path: Path,
                          start_time: float,
                          end_time: float,
                          output_path: Path) -> Path:
        """
        Process a single audio replacement with production quality
        
        Args:
            original_audio_path: Path to original audio
            replacement_audio_path: Path to replacement audio
            start_time: Start time of replacement
            end_time: End time of replacement
            output_path: Output path for processed audio
            
        Returns:
            Path to processed audio
        """
        # Load original audio segments
        before_audio, sr = librosa.load(
            str(original_audio_path),
            sr=48000,
            duration=start_time,
            mono=True
        )
        
        # Get reference from around the replacement point
        ref_start = max(0, start_time - 2.0)
        ref_duration = min(4.0, end_time - ref_start + 2.0)
        reference_audio, _ = librosa.load(
            str(original_audio_path),
            sr=48000,
            offset=ref_start,
            duration=ref_duration,
            mono=True
        )
        
        after_audio, _ = librosa.load(
            str(original_audio_path),
            sr=48000,
            offset=end_time,
            mono=True
        )
        
        # Load replacement
        replacement_audio, _ = librosa.load(
            str(replacement_audio_path),
            sr=48000,
            mono=True
        )
        
        # Analyze reference characteristics
        ref_features = self.analyze_spectral_characteristics(reference_audio, sr)
        
        # Match replacement to reference
        matched_replacement = self.match_audio_characteristics(
            replacement_audio, ref_features, sr
        )
        
        # Apply spectral crossfade
        processed = self.spectral_crossfade(
            before_audio,
            matched_replacement,
            after_audio
        )
        
        # Normalize loudness
        final_audio = self.normalize_loudness(processed, sr)
        
        # Save
        sf.write(str(output_path), final_audio, sr)
        
        return output_path
    
    def create_silence_preserving_noise_floor(self,
                                            reference_audio: np.ndarray,
                                            duration: float,
                                            sr: int = 48000) -> np.ndarray:
        """
        Create silence that preserves the noise floor characteristics
        
        Args:
            reference_audio: Reference audio for noise floor
            duration: Duration of silence in seconds
            sr: Sample rate
            
        Returns:
            Silence with matching noise floor
        """
        # Estimate noise floor from quiet parts
        # Find quietest 10% of the signal
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = frame_length // 2
        
        # Compute RMS energy for each frame
        rms = librosa.feature.rms(y=reference_audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Get quietest frames
        threshold = np.percentile(rms, 10)
        quiet_mask = rms < threshold
        
        # Extract noise floor samples
        noise_samples = []
        for i, is_quiet in enumerate(quiet_mask):
            if is_quiet:
                start = i * hop_length
                end = start + frame_length
                if end < len(reference_audio):
                    noise_samples.append(reference_audio[start:end])
        
        if noise_samples:
            # Create noise floor by concatenating random quiet segments
            samples_needed = int(duration * sr)
            noise_floor = np.zeros(samples_needed)
            
            idx = 0
            while idx < samples_needed:
                segment = noise_samples[np.random.randint(len(noise_samples))]
                segment_len = min(len(segment), samples_needed - idx)
                noise_floor[idx:idx + segment_len] = segment[:segment_len]
                idx += segment_len
            
            # Scale to very low level
            noise_floor *= 0.01
            
            return noise_floor
        else:
            # Fallback to very quiet white noise
            return np.random.randn(int(duration * sr)) * 0.0001


def test_audio_processor():
    """Test the audio processor with sample files"""
    processor = AudioProcessor()
    
    # Test spectral analysis
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000))
    features = processor.analyze_spectral_characteristics(test_audio, 48000)
    print("Test tone features:", features)
    
    # Test crossfade
    audio1 = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 24000))
    audio2 = np.sin(2 * np.pi * 880 * np.linspace(0, 0.5, 24000))
    audio3 = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 24000))
    
    result = processor.spectral_crossfade(audio1, audio2, audio3)
    print(f"Crossfade result length: {len(result)} samples")


if __name__ == "__main__":
    test_audio_processor()