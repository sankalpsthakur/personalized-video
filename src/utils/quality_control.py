#!/usr/bin/env python3
"""
Quality control module for automated validation
Ensures production-ready output quality
"""

import subprocess
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import warnings
warnings.filterwarnings('ignore')


class QualityControl:
    def __init__(self, 
                 target_lufs: float = -16.0,
                 max_true_peak: float = -1.0,
                 min_silence_db: float = -60.0):
        """
        Initialize quality control system
        
        Args:
            target_lufs: Target loudness for broadcast
            max_true_peak: Maximum true peak level in dBFS
            min_silence_db: Minimum level for silence detection
        """
        self.target_lufs = target_lufs
        self.max_true_peak = max_true_peak
        self.min_silence_db = min_silence_db
        
    def check_audio_levels(self, audio_path: Path) -> Dict:
        """
        Check audio levels and loudness compliance
        
        Returns:
            Dictionary with level measurements and compliance status
        """
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=48000, mono=False)
        
        # Convert to stereo if mono
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        
        # Import pyloudnorm for LUFS measurement
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        
        # Measure loudness
        if audio.shape[0] == 2:  # Stereo
            loudness = meter.integrated_loudness(audio.T)
        else:
            loudness = meter.integrated_loudness(audio)
        
        # Check true peak
        true_peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        
        # Check for clipping
        clipping_samples = np.sum(np.abs(audio) > 0.99)
        
        # Dynamic range
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        dynamic_range_db = 20 * np.log10(peak / (rms + 1e-10))
        
        # Silence detection
        silence_threshold = 10 ** (self.min_silence_db / 20)
        silence_ratio = np.sum(np.abs(audio) < silence_threshold) / audio.size
        
        return {
            'loudness_lufs': float(loudness),
            'loudness_compliant': abs(loudness - self.target_lufs) < 2.0,
            'true_peak_db': float(true_peak_db),
            'true_peak_compliant': true_peak_db < self.max_true_peak,
            'clipping_samples': int(clipping_samples),
            'has_clipping': clipping_samples > 0,
            'dynamic_range_db': float(dynamic_range_db),
            'silence_ratio': float(silence_ratio),
            'sample_rate': sr,
            'duration_seconds': len(audio.T) / sr
        }
    
    def check_video_quality(self, video_path: Path) -> Dict:
        """
        Check video quality metrics
        
        Returns:
            Dictionary with video quality measurements
        """
        # Get video info using ffprobe
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
        
        quality_info = {
            'format': info['format']['format_name'],
            'duration': float(info['format']['duration']),
            'size_mb': float(info['format']['size']) / (1024 * 1024),
            'bitrate_kbps': float(info['format']['bit_rate']) / 1000
        }
        
        if video_stream:
            quality_info['video'] = {
                'codec': video_stream['codec_name'],
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),
                'bitrate_kbps': float(video_stream.get('bit_rate', 0)) / 1000,
                'profile': video_stream.get('profile', 'unknown')
            }
        
        if audio_stream:
            quality_info['audio'] = {
                'codec': audio_stream['codec_name'],
                'sample_rate': int(audio_stream['sample_rate']),
                'channels': int(audio_stream['channels']),
                'bitrate_kbps': float(audio_stream.get('bit_rate', 0)) / 1000
            }
        
        # Check frame accuracy with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample a few frames to check for corruption
            sample_frames = [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
            corrupted_frames = 0
            
            for frame_idx in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    corrupted_frames += 1
            
            quality_info['video']['total_frames'] = frame_count
            quality_info['video']['corrupted_sample_frames'] = corrupted_frames
            quality_info['video']['integrity_ok'] = corrupted_frames == 0
            
            cap.release()
        
        return quality_info
    
    def check_sync(self, video_path: Path, tolerance_ms: float = 40) -> Dict:
        """
        Check audio-video synchronization
        
        Args:
            video_path: Path to video file
            tolerance_ms: Sync tolerance in milliseconds
            
        Returns:
            Dictionary with sync analysis
        """
        # This is a simplified check - production would use more sophisticated methods
        # like detecting claps or sync markers
        
        # Get stream timestamps
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time",
            "-of", "json",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        video_packets = json.loads(result.stdout).get('packets', [])[:10]  # First 10 packets
        
        cmd[2] = "a:0"  # Audio stream
        result = subprocess.run(cmd, capture_output=True, text=True)
        audio_packets = json.loads(result.stdout).get('packets', [])[:10]
        
        if video_packets and audio_packets:
            # Check initial sync
            video_start = float(video_packets[0].get('pts_time', 0))
            audio_start = float(audio_packets[0].get('pts_time', 0))
            sync_offset_ms = abs(video_start - audio_start) * 1000
            
            return {
                'sync_offset_ms': sync_offset_ms,
                'sync_ok': sync_offset_ms < tolerance_ms,
                'video_start_time': video_start,
                'audio_start_time': audio_start
            }
        
        return {
            'sync_offset_ms': 0,
            'sync_ok': True,
            'note': 'Could not determine sync accurately'
        }
    
    def check_replacement_quality(self,
                                original_path: Path,
                                replaced_path: Path,
                                replacement_times: List[Tuple[float, float]]) -> Dict:
        """
        Check quality of audio replacements
        
        Args:
            original_path: Original audio file
            replaced_path: Audio with replacements
            replacement_times: List of (start, end) times for replacements
            
        Returns:
            Dictionary with replacement quality metrics
        """
        # Load both audio files
        original, sr = librosa.load(str(original_path), sr=48000, mono=True)
        replaced, sr = librosa.load(str(replaced_path), sr=48000, mono=True)
        
        # Ensure same length
        min_len = min(len(original), len(replaced))
        original = original[:min_len]
        replaced = replaced[:min_len]
        
        results = {
            'replacements': [],
            'overall_quality': True
        }
        
        for start_time, end_time in replacement_times:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Check transition smoothness (look at boundaries)
            transition_samples = int(0.05 * sr)  # 50ms
            
            # Before transition
            if start_sample > transition_samples:
                before_orig = original[start_sample - transition_samples:start_sample]
                before_repl = replaced[start_sample - transition_samples:start_sample]
                before_diff = np.mean(np.abs(before_orig - before_repl))
            else:
                before_diff = 0
            
            # After transition
            if end_sample + transition_samples < len(original):
                after_orig = original[end_sample:end_sample + transition_samples]
                after_repl = replaced[end_sample:end_sample + transition_samples]
                after_diff = np.mean(np.abs(after_orig - after_repl))
            else:
                after_diff = 0
            
            # Check for clicks/pops at boundaries
            if start_sample > 0 and start_sample < len(replaced):
                start_discontinuity = abs(replaced[start_sample] - replaced[start_sample - 1])
            else:
                start_discontinuity = 0
                
            if end_sample > 0 and end_sample < len(replaced) - 1:
                end_discontinuity = abs(replaced[end_sample] - replaced[end_sample + 1])
            else:
                end_discontinuity = 0
            
            # Measure loudness consistency
            replacement_audio = replaced[start_sample:end_sample]
            if len(replacement_audio) > 0:
                replacement_rms = np.sqrt(np.mean(replacement_audio**2))
                
                # Compare with surrounding context
                context_size = int(2.0 * sr)  # 2 seconds
                context_start = max(0, start_sample - context_size)
                context_end = min(len(original), end_sample + context_size)
                context_audio = original[context_start:context_end]
                context_rms = np.sqrt(np.mean(context_audio**2))
                
                loudness_ratio = replacement_rms / (context_rms + 1e-10)
                loudness_db_diff = 20 * np.log10(loudness_ratio + 1e-10)
            else:
                loudness_db_diff = 0
            
            replacement_result = {
                'start_time': start_time,
                'end_time': end_time,
                'transition_quality': {
                    'before_diff': float(before_diff),
                    'after_diff': float(after_diff),
                    'smooth_transitions': before_diff < 0.01 and after_diff < 0.01
                },
                'discontinuities': {
                    'start': float(start_discontinuity),
                    'end': float(end_discontinuity),
                    'no_clicks': start_discontinuity < 0.1 and end_discontinuity < 0.1
                },
                'loudness_consistency': {
                    'db_difference': float(loudness_db_diff),
                    'consistent': abs(loudness_db_diff) < 3.0
                }
            }
            
            # Overall quality for this replacement
            replacement_result['quality_ok'] = (
                replacement_result['transition_quality']['smooth_transitions'] and
                replacement_result['discontinuities']['no_clicks'] and
                replacement_result['loudness_consistency']['consistent']
            )
            
            results['replacements'].append(replacement_result)
            
            if not replacement_result['quality_ok']:
                results['overall_quality'] = False
        
        return results
    
    def generate_qc_report(self,
                         video_path: Path,
                         output_path: Path,
                         replacement_times: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """
        Generate comprehensive QC report
        
        Args:
            video_path: Path to video file to check
            output_path: Path to save JSON report
            replacement_times: Optional list of replacement timestamps
            
        Returns:
            Complete QC report dictionary
        """
        report = {
            'file': str(video_path),
            'checks': {}
        }
        
        # Extract audio for analysis
        audio_path = video_path.with_suffix('.qc_audio.wav')
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-c:a", "pcm_s16le",
            "-ar", "48000", "-ac", "2",
            "-y", str(audio_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Run all checks
        try:
            report['checks']['audio_levels'] = self.check_audio_levels(audio_path)
            report['checks']['video_quality'] = self.check_video_quality(video_path)
            report['checks']['sync'] = self.check_sync(video_path)
            
            # Overall pass/fail
            report['passed'] = (
                report['checks']['audio_levels']['loudness_compliant'] and
                report['checks']['audio_levels']['true_peak_compliant'] and
                not report['checks']['audio_levels']['has_clipping'] and
                report['checks']['video_quality'].get('video', {}).get('integrity_ok', True) and
                report['checks']['sync']['sync_ok']
            )
            
            # Add recommendations
            report['recommendations'] = []
            
            if not report['checks']['audio_levels']['loudness_compliant']:
                current = report['checks']['audio_levels']['loudness_lufs']
                report['recommendations'].append(
                    f"Adjust loudness from {current:.1f} LUFS to {self.target_lufs} LUFS"
                )
            
            if report['checks']['audio_levels']['has_clipping']:
                report['recommendations'].append(
                    "Audio clipping detected - reduce levels or apply limiting"
                )
            
            if not report['checks']['sync']['sync_ok']:
                offset = report['checks']['sync']['sync_offset_ms']
                report['recommendations'].append(
                    f"Audio-video sync offset of {offset:.1f}ms detected"
                )
            
        finally:
            # Cleanup
            if audio_path.exists():
                audio_path.unlink()
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def test_quality_control():
    """Test quality control functionality"""
    qc = QualityControl()
    
    # Test with sample audio
    test_audio = Path("test_audio.wav")
    if test_audio.exists():
        levels = qc.check_audio_levels(test_audio)
        print("Audio levels:", json.dumps(levels, indent=2))
    
    print("Quality control system initialized successfully")


if __name__ == "__main__":
    test_quality_control()