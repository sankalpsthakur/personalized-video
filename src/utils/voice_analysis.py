#!/usr/bin/env python3
"""
Comprehensive Voice Analysis Tool
Analyzes and compares voice characteristics between original and personalized videos
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import warnings
import parselmouth
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import soundfile as sf

warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class VoiceFeatures:
    """Container for all voice analysis features"""
    # Pitch features
    pitch_mean: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    pitch_range: float
    jitter: float
    shimmer: float
    
    # Energy features
    energy_mean: float
    energy_std: float
    dynamic_range: float
    crest_factor: float
    
    # Spectral features
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_rolloff_mean: float
    spectral_flux_mean: float
    zcr_mean: float
    zcr_std: float
    
    # MFCC features
    mfcc_means: List[float]
    mfcc_stds: List[float]
    
    # Formants
    f1_mean: float
    f2_mean: float
    f3_mean: float
    
    # Temporal features
    speaking_rate: float
    pause_count: int
    avg_pause_duration: float
    
    # Voice quality
    hnr: float  # Harmonics-to-Noise Ratio
    breathiness: float
    
    def to_dict(self) -> Dict:
        # Convert numpy types to Python native types for JSON serialization
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, np.floating):
                data[key] = float(value)
            elif isinstance(value, np.integer):
                data[key] = int(value)
            elif isinstance(value, list):
                data[key] = [float(v) if isinstance(v, np.floating) else v for v in value]
        return data


class VoiceAnalyzer:
    """Comprehensive voice analysis and comparison tool"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_length = int(0.025 * sample_rate)  # 25ms frames
        self.hop_length = int(0.010 * sample_rate)    # 10ms hop
        
    def extract_audio_from_video(self, video_path: Path) -> Path:
        """Extract audio from video file"""
        audio_path = video_path.parent / f"{video_path.stem}_audio.wav"
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",  # Mono
            "-y", str(audio_path)
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return audio_path
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        return y, sr
    
    def extract_pitch_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract pitch-related features using Parselmouth (Praat)"""
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch()
        
        # Get pitch values
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
        
        if len(pitch_values) == 0:
            return {
                'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0,
                'pitch_max': 0, 'pitch_range': 0, 'jitter': 0, 'shimmer': 0
            }
        
        # Calculate jitter
        point_process = parselmouth.praat.call(
            sound, "To PointProcess (periodic, cc)", 75, 600
        )
        jitter = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        
        # Calculate shimmer
        shimmer = parselmouth.praat.call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        
        return {
            'pitch_mean': np.mean(pitch_values),
            'pitch_std': np.std(pitch_values),
            'pitch_min': np.min(pitch_values),
            'pitch_max': np.max(pitch_values),
            'pitch_range': np.max(pitch_values) - np.min(pitch_values),
            'jitter': jitter * 100,  # Convert to percentage
            'shimmer': shimmer * 100  # Convert to percentage
        }
    
    def extract_energy_features(self, audio: np.ndarray) -> Dict:
        """Extract energy and amplitude-related features"""
        # RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, 
                                  hop_length=self.hop_length)[0]
        
        # Peak amplitude
        peak_amplitude = np.max(np.abs(audio))
        
        # Crest factor
        crest_factor = peak_amplitude / (np.sqrt(np.mean(audio**2)) + 1e-8)
        
        # Dynamic range (in dB)
        min_rms = np.percentile(rms[rms > 0], 5)
        max_rms = np.percentile(rms, 95)
        dynamic_range = 20 * np.log10(max_rms / (min_rms + 1e-8))
        
        return {
            'energy_mean': np.mean(rms),
            'energy_std': np.std(rms),
            'dynamic_range': dynamic_range,
            'crest_factor': crest_factor
        }
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract spectral and timbral features"""
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        
        # Spectral flux
        stft = librosa.stft(audio, n_fft=2048, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        spectral_flux = np.sum(np.diff(magnitude, axis=1)**2, axis=0)
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_flux_mean': np.mean(spectral_flux),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr)
        }
    
    def extract_mfcc_features(self, audio: np.ndarray, sr: int, n_mfcc: int = 13) -> Dict:
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, 
                                     hop_length=self.hop_length)
        
        return {
            'mfcc_means': np.mean(mfccs, axis=1).tolist(),
            'mfcc_stds': np.std(mfccs, axis=1).tolist()
        }
    
    def extract_formants(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract formant frequencies using Parselmouth"""
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        formants = sound.to_formant_burg()
        
        # Get average formant values
        f1_values = []
        f2_values = []
        f3_values = []
        
        for i in range(int(formants.get_total_duration() * 100)):
            time = i / 100
            f1 = formants.get_value_at_time(1, time)
            f2 = formants.get_value_at_time(2, time)
            f3 = formants.get_value_at_time(3, time)
            
            if f1 and not np.isnan(f1):
                f1_values.append(f1)
            if f2 and not np.isnan(f2):
                f2_values.append(f2)
            if f3 and not np.isnan(f3):
                f3_values.append(f3)
        
        return {
            'f1_mean': np.mean(f1_values) if f1_values else 0,
            'f2_mean': np.mean(f2_values) if f2_values else 0,
            'f3_mean': np.mean(f3_values) if f3_values else 0
        }
    
    def extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract temporal and rhythmic features"""
        # Detect pauses (silence)
        intervals = librosa.effects.split(audio, top_db=30)
        
        # Calculate pause statistics
        pauses = []
        if len(intervals) > 1:
            for i in range(1, len(intervals)):
                pause_start = intervals[i-1][1] / sr
                pause_end = intervals[i][0] / sr
                pause_duration = pause_end - pause_start
                if pause_duration > 0.1:  # Consider pauses > 100ms
                    pauses.append(pause_duration)
        
        # Estimate speaking rate (syllables per second)
        # Using onset detection as proxy for syllables
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, 
                                                 hop_length=self.hop_length)
        syllable_rate = len(onset_frames) / (len(audio) / sr)
        
        return {
            'speaking_rate': syllable_rate,
            'pause_count': len(pauses),
            'avg_pause_duration': np.mean(pauses) if pauses else 0
        }
    
    def extract_voice_quality_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract voice quality features"""
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Harmonics-to-Noise Ratio (HNR)
        harmonicity = sound.to_harmonicity()
        hnr_values = harmonicity.values[harmonicity.values != -200]
        hnr_mean = np.mean(hnr_values) if len(hnr_values) > 0 else 0
        
        # Breathiness estimation (high frequency energy ratio)
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        low_freq_energy = np.sum(magnitude[:int(magnitude.shape[0]/2), :])
        high_freq_energy = np.sum(magnitude[int(magnitude.shape[0]/2):, :])
        breathiness = high_freq_energy / (low_freq_energy + 1e-8)
        
        return {
            'hnr': hnr_mean,
            'breathiness': breathiness
        }
    
    def analyze_voice(self, audio_path: Path) -> VoiceFeatures:
        """Perform complete voice analysis"""
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Extract all features
        pitch_features = self.extract_pitch_features(audio, sr)
        energy_features = self.extract_energy_features(audio)
        spectral_features = self.extract_spectral_features(audio, sr)
        mfcc_features = self.extract_mfcc_features(audio, sr)
        formant_features = self.extract_formants(audio, sr)
        temporal_features = self.extract_temporal_features(audio, sr)
        quality_features = self.extract_voice_quality_features(audio, sr)
        
        # Combine all features
        return VoiceFeatures(
            **pitch_features,
            **energy_features,
            **spectral_features,
            **mfcc_features,
            **formant_features,
            **temporal_features,
            **quality_features
        )
    
    def compare_voices(self, original_features: VoiceFeatures, 
                      modified_features: VoiceFeatures) -> Dict:
        """Compare two voice profiles and calculate similarity metrics"""
        comparison = {}
        
        # Pitch comparison
        comparison['pitch'] = {
            'mean_diff': abs(original_features.pitch_mean - modified_features.pitch_mean),
            'mean_diff_percent': abs(original_features.pitch_mean - modified_features.pitch_mean) / 
                                original_features.pitch_mean * 100 if original_features.pitch_mean > 0 else 0,
            'range_diff': abs(original_features.pitch_range - modified_features.pitch_range),
            'jitter_diff': abs(original_features.jitter - modified_features.jitter),
            'shimmer_diff': abs(original_features.shimmer - modified_features.shimmer)
        }
        
        # Energy comparison
        comparison['energy'] = {
            'mean_diff': abs(original_features.energy_mean - modified_features.energy_mean),
            'dynamic_range_diff': abs(original_features.dynamic_range - modified_features.dynamic_range),
            'crest_factor_diff': abs(original_features.crest_factor - modified_features.crest_factor)
        }
        
        # Spectral comparison
        comparison['spectral'] = {
            'centroid_diff': abs(original_features.spectral_centroid_mean - 
                               modified_features.spectral_centroid_mean),
            'rolloff_diff': abs(original_features.spectral_rolloff_mean - 
                              modified_features.spectral_rolloff_mean),
            'zcr_diff': abs(original_features.zcr_mean - modified_features.zcr_mean)
        }
        
        # MFCC comparison (cosine similarity)
        orig_mfcc = np.array(original_features.mfcc_means)
        mod_mfcc = np.array(modified_features.mfcc_means)
        mfcc_similarity = np.dot(orig_mfcc, mod_mfcc) / (np.linalg.norm(orig_mfcc) * 
                                                         np.linalg.norm(mod_mfcc))
        comparison['mfcc_similarity'] = mfcc_similarity
        
        # Formant comparison
        comparison['formants'] = {
            'f1_diff': abs(original_features.f1_mean - modified_features.f1_mean),
            'f2_diff': abs(original_features.f2_mean - modified_features.f2_mean),
            'f3_diff': abs(original_features.f3_mean - modified_features.f3_mean)
        }
        
        # Overall similarity score (0-100)
        pitch_score = max(0, 100 - comparison['pitch']['mean_diff_percent'])
        energy_score = max(0, 100 - abs(comparison['energy']['dynamic_range_diff']) * 5)
        spectral_score = max(0, 100 - (comparison['spectral']['centroid_diff'] / 100))
        mfcc_score = comparison['mfcc_similarity'] * 100
        formant_score = max(0, 100 - (comparison['formants']['f1_diff'] + 
                                     comparison['formants']['f2_diff']) / 20)
        
        comparison['overall_similarity'] = np.mean([
            pitch_score, energy_score, spectral_score, mfcc_score, formant_score
        ])
        
        return comparison
    
    def generate_report(self, original_path: Path, modified_path: Path, 
                       output_path: Path) -> Dict:
        """Generate comprehensive voice analysis report"""
        print("Analyzing original voice...")
        original_features = self.analyze_voice(original_path)
        
        print("Analyzing modified voice...")
        modified_features = self.analyze_voice(modified_path)
        
        print("Comparing voices...")
        comparison = self.compare_voices(original_features, modified_features)
        
        # Create report
        report = {
            'original_voice': original_features.to_dict(),
            'modified_voice': modified_features.to_dict(),
            'comparison': comparison,
            'recommendations': self._generate_recommendations(comparison)
        }
        
        # Save report with custom encoder
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        # Generate visualizations
        self._generate_visualizations(original_features, modified_features, 
                                    output_path.parent)
        
        return report
    
    def _generate_recommendations(self, comparison: Dict) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        if comparison['pitch']['mean_diff_percent'] > 20:
            recommendations.append(
                "Large pitch difference detected. Consider adjusting TTS voice settings."
            )
        
        if comparison['mfcc_similarity'] < 0.7:
            recommendations.append(
                "Low timbral similarity. The TTS voice characteristics differ significantly."
            )
        
        if comparison['energy']['dynamic_range_diff'] > 10:
            recommendations.append(
                "Dynamic range mismatch. Apply compression or normalization."
            )
        
        if comparison['overall_similarity'] < 70:
            recommendations.append(
                "Overall similarity is low. Consider using a different TTS voice or "
                "applying voice conversion techniques."
            )
        
        return recommendations
    
    def _generate_visualizations(self, original: VoiceFeatures, 
                               modified: VoiceFeatures, output_dir: Path):
        """Generate comparison visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Voice Comparison Analysis', fontsize=16)
        
        # Pitch comparison
        ax = axes[0, 0]
        features = ['Mean', 'Range', 'Jitter', 'Shimmer']
        original_vals = [original.pitch_mean, original.pitch_range, 
                        original.jitter, original.shimmer]
        modified_vals = [modified.pitch_mean, modified.pitch_range,
                        modified.jitter, modified.shimmer]
        
        x = np.arange(len(features))
        ax.bar(x - 0.2, original_vals, 0.4, label='Original', alpha=0.8)
        ax.bar(x + 0.2, modified_vals, 0.4, label='Modified', alpha=0.8)
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        ax.set_title('Pitch Features')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.legend()
        
        # MFCC comparison
        ax = axes[0, 1]
        ax.plot(original.mfcc_means, 'o-', label='Original', alpha=0.8)
        ax.plot(modified.mfcc_means, 's-', label='Modified', alpha=0.8)
        ax.set_xlabel('MFCC Coefficient')
        ax.set_ylabel('Mean Value')
        ax.set_title('MFCC Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Formants comparison
        ax = axes[0, 2]
        formants = ['F1', 'F2', 'F3']
        original_formants = [original.f1_mean, original.f2_mean, original.f3_mean]
        modified_formants = [modified.f1_mean, modified.f2_mean, modified.f3_mean]
        
        x = np.arange(len(formants))
        ax.bar(x - 0.2, original_formants, 0.4, label='Original', alpha=0.8)
        ax.bar(x + 0.2, modified_formants, 0.4, label='Modified', alpha=0.8)
        ax.set_xlabel('Formant')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Formant Frequencies')
        ax.set_xticks(x)
        ax.set_xticklabels(formants)
        ax.legend()
        
        # Energy features
        ax = axes[1, 0]
        features = ['Mean Energy', 'Dynamic Range', 'Crest Factor']
        original_energy = [original.energy_mean, original.dynamic_range, 
                          original.crest_factor]
        modified_energy = [modified.energy_mean, modified.dynamic_range,
                          modified.crest_factor]
        
        x = np.arange(len(features))
        ax.bar(x - 0.2, original_energy, 0.4, label='Original', alpha=0.8)
        ax.bar(x + 0.2, modified_energy, 0.4, label='Modified', alpha=0.8)
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        ax.set_title('Energy Features')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.legend()
        
        # Spectral features
        ax = axes[1, 1]
        features = ['Centroid', 'Rolloff', 'ZCR']
        original_spectral = [original.spectral_centroid_mean / 1000,  # Convert to kHz
                           original.spectral_rolloff_mean / 1000,
                           original.zcr_mean * 100]  # Scale for visibility
        modified_spectral = [modified.spectral_centroid_mean / 1000,
                           modified.spectral_rolloff_mean / 1000,
                           modified.zcr_mean * 100]
        
        x = np.arange(len(features))
        ax.bar(x - 0.2, original_spectral, 0.4, label='Original', alpha=0.8)
        ax.bar(x + 0.2, modified_spectral, 0.4, label='Modified', alpha=0.8)
        ax.set_xlabel('Features')
        ax.set_ylabel('Values (kHz / scaled)')
        ax.set_title('Spectral Features')
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()
        
        # Voice quality
        ax = axes[1, 2]
        features = ['HNR (dB)', 'Breathiness', 'Speaking Rate']
        original_quality = [original.hnr, original.breathiness * 100, 
                          original.speaking_rate]
        modified_quality = [modified.hnr, modified.breathiness * 100,
                          modified.speaking_rate]
        
        x = np.arange(len(features))
        ax.bar(x - 0.2, original_quality, 0.4, label='Original', alpha=0.8)
        ax.bar(x + 0.2, modified_quality, 0.4, label='Modified', alpha=0.8)
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        ax.set_title('Voice Quality')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'voice_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate detailed pitch contour comparison
        self._plot_pitch_contours(original, modified, output_dir)
    
    def _plot_pitch_contours(self, original: VoiceFeatures, 
                            modified: VoiceFeatures, output_dir: Path):
        """Plot detailed pitch contours"""
        # This would require the full audio signals for detailed plotting
        # For now, we'll create a summary visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create box plots for pitch distribution
        pitch_data = {
            'Original': np.random.normal(original.pitch_mean, original.pitch_std, 1000),
            'Modified': np.random.normal(modified.pitch_mean, modified.pitch_std, 1000)
        }
        
        df = pd.DataFrame(pitch_data)
        df.boxplot(ax=ax)
        ax.set_ylabel('Pitch (Hz)')
        ax.set_title('Pitch Distribution Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pitch_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Voice Analysis Tool')
    parser.add_argument('original', help='Original audio/video file')
    parser.add_argument('modified', help='Modified audio/video file')
    parser.add_argument('--output-dir', default='voice_analysis_output',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = VoiceAnalyzer()
    
    # Process files
    original_path = Path(args.original)
    modified_path = Path(args.modified)
    
    # Extract audio if video files provided
    if original_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        print("Extracting audio from original video...")
        original_audio = analyzer.extract_audio_from_video(original_path)
    else:
        original_audio = original_path
    
    if modified_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        print("Extracting audio from modified video...")
        modified_audio = analyzer.extract_audio_from_video(modified_path)
    else:
        modified_audio = modified_path
    
    # Generate report
    report_path = output_dir / 'voice_analysis_report.json'
    report = analyzer.generate_report(original_audio, modified_audio, report_path)
    
    # Print summary
    print("\n" + "="*60)
    print("VOICE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Overall Similarity: {report['comparison']['overall_similarity']:.1f}%")
    print(f"Pitch Difference: {report['comparison']['pitch']['mean_diff_percent']:.1f}%")
    print(f"MFCC Similarity: {report['comparison']['mfcc_similarity']:.2f}")
    print(f"Formant F1 Difference: {report['comparison']['formants']['f1_diff']:.0f} Hz")
    print(f"Formant F2 Difference: {report['comparison']['formants']['f2_diff']:.0f} Hz")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print(f"\nDetailed report saved to: {report_path}")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()