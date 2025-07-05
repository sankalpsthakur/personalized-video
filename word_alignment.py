#!/usr/bin/env python3
"""
Word alignment module for precise word and phoneme-level timing
Uses Whisper for transcription with word timestamps
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False


class WordAligner:
    def __init__(self, 
                 whisper_model: str = "base",
                 vad_aggressiveness: int = 2):
        """
        Initialize word aligner
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            vad_aggressiveness: WebRTC VAD aggressiveness (0-3)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load Whisper model
        if WHISPER_AVAILABLE:
            self.logger.info(f"Loading Whisper model: {whisper_model}")
            self.whisper_model = whisper.load_model(whisper_model)
        else:
            self.logger.warning("Whisper not available, using mock alignment")
            self.whisper_model = None
            
        # Initialize VAD
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(vad_aggressiveness)
        else:
            self.vad = None
    
    def align_words(self, audio_path: Path) -> Dict:
        """
        Perform word-level alignment on audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with word alignments
        """
        if self.whisper_model:
            return self._whisper_alignment(audio_path)
        else:
            return self._mock_alignment()
    
    def _whisper_alignment(self, audio_path: Path) -> Dict:
        """Use Whisper for word-level alignment"""
        # Transcribe with word timestamps
        result = self.whisper_model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language="en",
            fp16=False  # Disable FP16 for CPU
        )
        
        # Extract word-level information
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info["word"].strip(),
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "confidence": word_info.get("probability", 1.0)
                })
        
        return {
            "text": result["text"],
            "language": result.get("language", "en"),
            "words": words,
            "segments": result.get("segments", [])
        }
    
    def _mock_alignment(self) -> Dict:
        """Mock alignment for testing without Whisper"""
        return {
            "text": "Hello Anuj ji, welcome to our presentation about Anuj ji's favorite destination Bali.",
            "language": "en",
            "words": [
                {"word": "Hello", "start": 0.5, "end": 0.8, "confidence": 0.95},
                {"word": "Anuj", "start": 1.0, "end": 1.3, "confidence": 0.98},
                {"word": "ji", "start": 1.3, "end": 1.5, "confidence": 0.96},
                {"word": "welcome", "start": 1.6, "end": 2.0, "confidence": 0.97},
                {"word": "to", "start": 2.1, "end": 2.2, "confidence": 0.99},
                {"word": "our", "start": 2.3, "end": 2.4, "confidence": 0.98},
                {"word": "presentation", "start": 2.5, "end": 3.0, "confidence": 0.96},
                {"word": "about", "start": 3.1, "end": 3.3, "confidence": 0.97},
                {"word": "Anuj", "start": 21.8, "end": 22.0, "confidence": 0.97},
                {"word": "ji's", "start": 22.0, "end": 22.3, "confidence": 0.95},
                {"word": "favorite", "start": 22.4, "end": 22.8, "confidence": 0.96},
                {"word": "destination", "start": 22.9, "end": 23.4, "confidence": 0.97},
                {"word": "Bali", "start": 23.5, "end": 23.8, "confidence": 0.98}
            ],
            "segments": [{
                "id": 0,
                "start": 0.0,
                "end": 24.0,
                "text": " Hello Anuj ji, welcome to our presentation about Anuj ji's favorite destination Bali.",
                "words": []  # Already extracted above
            }]
        }
    
    def refine_word_boundaries(self, 
                             audio_path: Path,
                             words: List[Dict],
                             context_ms: int = 50) -> List[Dict]:
        """
        Refine word boundaries using VAD and energy analysis
        
        Args:
            audio_path: Path to audio file
            words: List of word dictionaries with start/end times
            context_ms: Context window in milliseconds
            
        Returns:
            Refined word list
        """
        if not VAD_AVAILABLE:
            return words
            
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
            
        # Resample to 16kHz for VAD
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # Process each word
        refined_words = []
        for word in words:
            start_sample = int(word['start'] * sr)
            end_sample = int(word['end'] * sr)
            
            # Add context
            context_samples = int(context_ms * sr / 1000)
            start_with_context = max(0, start_sample - context_samples)
            end_with_context = min(len(audio), end_sample + context_samples)
            
            # Extract segment
            segment = audio[start_with_context:end_with_context]
            
            # Find voice activity boundaries
            refined_start, refined_end = self._find_voice_boundaries(
                segment, sr, context_samples
            )
            
            # Update timing
            refined_word = word.copy()
            refined_word['start'] = (start_with_context + refined_start) / sr
            refined_word['end'] = (start_with_context + refined_end) / sr
            refined_words.append(refined_word)
            
        return refined_words
    
    def _find_voice_boundaries(self, 
                              audio_segment: np.ndarray,
                              sample_rate: int,
                              context_samples: int) -> Tuple[int, int]:
        """Find precise voice boundaries in audio segment"""
        # Convert to 16-bit PCM for VAD
        audio_16bit = (audio_segment * 32767).astype(np.int16)
        
        # Frame size for VAD (10, 20, or 30 ms)
        frame_duration_ms = 10
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Process frames
        voice_flags = []
        for i in range(0, len(audio_16bit) - frame_size, frame_size):
            frame = audio_16bit[i:i + frame_size].tobytes()
            is_speech = self.vad.is_speech(frame, sample_rate)
            voice_flags.append(is_speech)
        
        # Find first and last voice frame
        if any(voice_flags):
            first_voice = voice_flags.index(True) * frame_size
            last_voice = (len(voice_flags) - voice_flags[::-1].index(True)) * frame_size
            
            # Ensure we don't cut into the word
            first_voice = max(0, first_voice - frame_size)
            last_voice = min(len(audio_16bit), last_voice + frame_size)
            
            return first_voice, last_voice
        else:
            # No voice detected, return original boundaries
            return context_samples, len(audio_16bit) - context_samples
    
    def extract_phonemes(self, word: str, audio_segment: np.ndarray, sr: int) -> List[Dict]:
        """
        Extract phoneme-level timing (simplified version)
        In production, would use forced alignment tools like Montreal Forced Aligner
        
        Args:
            word: The word to analyze
            audio_segment: Audio segment containing the word
            sr: Sample rate
            
        Returns:
            List of phoneme timings
        """
        # This is a simplified phoneme extraction
        # Real implementation would use tools like:
        # - Montreal Forced Aligner
        # - Kaldi
        # - DeepSpeech with phoneme output
        
        word_duration = len(audio_segment) / sr
        
        # Simple heuristic: divide word equally by estimated phonemes
        # This is NOT accurate - just for demonstration
        phoneme_count = max(1, len(word) // 2)  # Very rough estimate
        phoneme_duration = word_duration / phoneme_count
        
        phonemes = []
        for i in range(phoneme_count):
            phonemes.append({
                "phoneme": f"ph{i}",  # Placeholder
                "start": i * phoneme_duration,
                "end": (i + 1) * phoneme_duration
            })
            
        return phonemes


def test_word_aligner():
    """Test word alignment functionality"""
    aligner = WordAligner()
    
    # Test with actual video file if it exists
    video_file = Path("VIDEO-2025-07-05-16-44-05.mp4")
    test_audio = Path("test_audio.wav")
    
    if video_file.exists():
        # Extract audio for testing
        import subprocess
        cmd = ["ffmpeg", "-i", str(video_file), "-ac", "1", "-ar", "16000", 
               "-t", "5", "-y", str(test_audio)]
        subprocess.run(cmd, capture_output=True)
        
        if test_audio.exists():
            result = aligner.align_words(test_audio)
            test_audio.unlink()  # Clean up
        else:
            # Fall back to mock
            result = aligner._mock_alignment()
    else:
        # Use mock alignment
        result = aligner._mock_alignment()
    
    print("Alignment result:")
    print(f"- Text: {result['text']}")
    print(f"- Words found: {len(result['words'])}")
    
    if result['words']:
        print("\nFirst few words:")
        for word in result['words'][:5]:
            print(f"  - {word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")


if __name__ == "__main__":
    test_word_aligner()