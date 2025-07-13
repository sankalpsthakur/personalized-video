"""
Sync Loss Evaluation Module
Provides comprehensive synchronization analysis and metrics for lip sync models
"""

import os
import cv2
import numpy as np
import torch
import librosa
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
import json

logger = logging.getLogger(__name__)


@dataclass
class SyncLossMetrics:
    """Comprehensive sync loss metrics"""
    # Frame-level metrics (milliseconds)
    avg_offset_ms: float = 0.0
    max_offset_ms: float = 0.0
    std_offset_ms: float = 0.0
    
    # Correlation metrics
    audio_visual_correlation: float = 0.0
    temporal_consistency: float = 0.0
    
    # Quality scores (0-1, higher is better)
    sync_accuracy_score: float = 0.0
    jitter_score: float = 0.0  # Lower jitter = better
    
    # Detection metrics
    frames_analyzed: int = 0
    sync_violations: int = 0  # Frames with >40ms offset
    
    # Model-specific
    model_name: str = ""
    processing_time: float = 0.0
    quantization_level: str = "fp32"


@dataclass
class FrameLevelSync:
    """Frame-level synchronization data"""
    frame_idx: int
    timestamp_ms: float
    audio_energy: float
    mouth_openness: float
    sync_offset_ms: float
    lip_movement_intensity: float


class SyncLossEvaluator:
    """Advanced synchronization loss evaluation and analysis"""
    
    def __init__(self, sample_rate: int = 16000, frame_rate: float = 30.0):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.frame_duration_ms = 1000.0 / frame_rate
        
        # Sync thresholds (milliseconds)
        self.perfect_sync_threshold = 20.0  # <20ms = perfect
        self.good_sync_threshold = 40.0     # <40ms = good
        self.poor_sync_threshold = 80.0     # >80ms = poor
        
        logger.info(f"SyncLossEvaluator initialized: {sample_rate}Hz, {frame_rate}fps")
    
    def evaluate_video_sync(self, video_path: str, audio_path: str, 
                          model_name: str = "unknown", 
                          quantization: str = "fp32") -> SyncLossMetrics:
        """
        Comprehensive sync loss evaluation for a processed video
        
        Args:
            video_path: Path to processed video
            audio_path: Path to reference audio
            model_name: Name of lip sync model used
            quantization: Quantization level (fp32, fp16, int8)
            
        Returns:
            Detailed sync loss metrics
        """
        try:
            logger.info(f"Evaluating sync for {model_name} ({quantization})")
            
            # Extract frame-level data
            frame_data = self._extract_frame_level_sync(video_path, audio_path)
            
            if not frame_data:
                logger.warning("No frame data extracted, returning empty metrics")
                return SyncLossMetrics(model_name=model_name, quantization_level=quantization)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_sync_metrics(frame_data, model_name, quantization)
            
            logger.info(f"Sync evaluation complete: {metrics.avg_offset_ms:.1f}ms avg offset")
            return metrics
            
        except Exception as e:
            logger.error(f"Sync evaluation failed: {e}")
            return SyncLossMetrics(model_name=model_name, quantization_level=quantization)
    
    def _extract_frame_level_sync(self, video_path: str, audio_path: str) -> List[FrameLevelSync]:
        """Extract frame-by-frame synchronization data"""
        frame_data = []
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            audio_features = self._extract_audio_features(audio)
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing {frame_count} frames for sync analysis")
            
            frame_idx = 0
            while frame_idx < frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp_ms = frame_idx * self.frame_duration_ms
                
                # Extract visual features
                mouth_openness = self._calculate_mouth_openness(frame)
                lip_movement = self._calculate_lip_movement_intensity(frame)
                
                # Get corresponding audio features
                audio_idx = int(timestamp_ms * self.sample_rate / 1000)
                if audio_idx < len(audio_features):
                    audio_energy = audio_features[audio_idx]
                else:
                    audio_energy = 0.0
                
                # Calculate sync offset
                sync_offset = self._calculate_sync_offset(
                    mouth_openness, audio_energy, timestamp_ms, audio_features
                )
                
                frame_data.append(FrameLevelSync(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    audio_energy=audio_energy,
                    mouth_openness=mouth_openness,
                    sync_offset_ms=sync_offset,
                    lip_movement_intensity=lip_movement
                ))
                
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    logger.debug(f"Processed {frame_idx}/{frame_count} frames")
            
            cap.release()
            logger.info(f"Extracted sync data for {len(frame_data)} frames")
            return frame_data
            
        except Exception as e:
            logger.error(f"Frame-level sync extraction failed: {e}")
            return []
    
    def _extract_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract audio energy features for sync analysis"""
        try:
            # Calculate frame-wise energy
            hop_length = int(self.sample_rate / self.frame_rate)
            frame_energy = librosa.feature.rms(
                y=audio, 
                hop_length=hop_length,
                frame_length=hop_length*2
            )[0]
            
            # Smooth energy for better sync detection
            frame_energy = signal.medfilt(frame_energy, kernel_size=3)
            
            # Normalize
            if np.max(frame_energy) > 0:
                frame_energy = frame_energy / np.max(frame_energy)
            
            return frame_energy
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return np.zeros(100)  # Fallback
    
    def _calculate_mouth_openness(self, frame: np.ndarray) -> float:
        """Calculate mouth openness from video frame"""
        try:
            # Simplified mouth detection using face landmarks
            # In practice, would use MediaPipe or similar
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple mouth region estimation (lower third of face)
            h, w = gray.shape
            mouth_region = gray[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
            
            # Calculate variance as proxy for mouth openness
            mouth_variance = np.var(mouth_region)
            
            # Normalize to 0-1 range
            mouth_openness = min(mouth_variance / 1000.0, 1.0)
            
            return mouth_openness
            
        except Exception as e:
            logger.debug(f"Mouth openness calculation failed: {e}")
            return 0.0
    
    def _calculate_lip_movement_intensity(self, frame: np.ndarray) -> float:
        """Calculate lip movement intensity"""
        try:
            # Simplified lip movement detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Focus on mouth region
            h, w = gray.shape
            lip_region = gray[int(h*0.65):int(h*0.85), int(w*0.35):int(w*0.65)]
            
            # Calculate edge intensity as proxy for lip movement
            edges = cv2.Canny(lip_region, 50, 150)
            movement_intensity = np.sum(edges) / edges.size
            
            # Normalize
            movement_intensity = min(movement_intensity / 50.0, 1.0)
            
            return movement_intensity
            
        except Exception as e:
            logger.debug(f"Lip movement calculation failed: {e}")
            return 0.0
    
    def _calculate_sync_offset(self, mouth_openness: float, audio_energy: float, 
                             timestamp_ms: float, audio_features: np.ndarray) -> float:
        """Calculate synchronization offset in milliseconds"""
        try:
            # Find the optimal audio-visual alignment using cross-correlation
            
            # Create a small window around current timestamp
            window_ms = 200  # ±100ms window
            center_frame = int(timestamp_ms / self.frame_duration_ms)
            window_frames = int(window_ms / self.frame_duration_ms)
            
            start_frame = max(0, center_frame - window_frames // 2)
            end_frame = min(len(audio_features), center_frame + window_frames // 2)
            
            if end_frame <= start_frame:
                return 0.0
            
            # Get audio segment
            audio_segment = audio_features[start_frame:end_frame]
            
            # Create visual signal (simplified)
            visual_signal = np.array([mouth_openness] * len(audio_segment))
            
            # Calculate cross-correlation
            if len(audio_segment) > 1:
                correlation = signal.correlate(audio_segment, visual_signal, mode='same')
                peak_idx = np.argmax(correlation)
                
                # Convert to milliseconds offset
                offset_frames = peak_idx - len(correlation) // 2
                offset_ms = offset_frames * self.frame_duration_ms
                
                return offset_ms
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Sync offset calculation failed: {e}")
            return 0.0
    
    def _calculate_sync_metrics(self, frame_data: List[FrameLevelSync], 
                               model_name: str, quantization: str) -> SyncLossMetrics:
        """Calculate comprehensive sync loss metrics"""
        try:
            if not frame_data:
                return SyncLossMetrics(model_name=model_name, quantization_level=quantization)
            
            # Extract offset data
            offsets = [abs(f.sync_offset_ms) for f in frame_data]
            
            # Basic statistics
            avg_offset = np.mean(offsets)
            max_offset = np.max(offsets)
            std_offset = np.std(offsets)
            
            # Audio-visual correlation
            audio_energies = [f.audio_energy for f in frame_data]
            mouth_openness = [f.mouth_openness for f in frame_data]
            
            if len(set(audio_energies)) > 1 and len(set(mouth_openness)) > 1:
                correlation, _ = pearsonr(audio_energies, mouth_openness)
                av_correlation = abs(correlation)
            else:
                av_correlation = 0.0
            
            # Temporal consistency (smoothness of sync)
            if len(offsets) > 1:
                offset_diff = np.diff(offsets)
                temporal_consistency = 1.0 / (1.0 + np.std(offset_diff))
            else:
                temporal_consistency = 1.0
            
            # Sync accuracy score (0-1, higher is better)
            perfect_frames = sum(1 for o in offsets if o <= self.perfect_sync_threshold)
            sync_accuracy = perfect_frames / len(offsets)
            
            # Jitter score (lower is better)
            jitter = np.std(offsets) / (avg_offset + 1.0)
            
            # Sync violations
            violations = sum(1 for o in offsets if o > self.good_sync_threshold)
            
            return SyncLossMetrics(
                avg_offset_ms=avg_offset,
                max_offset_ms=max_offset,
                std_offset_ms=std_offset,
                audio_visual_correlation=av_correlation,
                temporal_consistency=temporal_consistency,
                sync_accuracy_score=sync_accuracy,
                jitter_score=jitter,
                frames_analyzed=len(frame_data),
                sync_violations=violations,
                model_name=model_name,
                quantization_level=quantization
            )
            
        except Exception as e:
            logger.error(f"Sync metrics calculation failed: {e}")
            return SyncLossMetrics(model_name=model_name, quantization_level=quantization)
    
    def compare_models(self, results: List[SyncLossMetrics]) -> Dict[str, Any]:
        """Compare sync loss across multiple models"""
        try:
            if not results:
                return {}
            
            comparison = {
                "summary": {
                    "total_models": len(results),
                    "best_sync_model": "",
                    "best_avg_offset": float('inf'),
                    "best_accuracy_model": "",
                    "best_accuracy_score": 0.0
                },
                "rankings": {
                    "by_avg_offset": [],
                    "by_accuracy": [],
                    "by_temporal_consistency": []
                },
                "detailed_metrics": {}
            }
            
            # Find best models
            for result in results:
                model_id = f"{result.model_name}_{result.quantization_level}"
                
                if result.avg_offset_ms < comparison["summary"]["best_avg_offset"]:
                    comparison["summary"]["best_avg_offset"] = result.avg_offset_ms
                    comparison["summary"]["best_sync_model"] = model_id
                
                if result.sync_accuracy_score > comparison["summary"]["best_accuracy_score"]:
                    comparison["summary"]["best_accuracy_score"] = result.sync_accuracy_score
                    comparison["summary"]["best_accuracy_model"] = model_id
                
                comparison["detailed_metrics"][model_id] = {
                    "avg_offset_ms": result.avg_offset_ms,
                    "sync_accuracy": result.sync_accuracy_score,
                    "temporal_consistency": result.temporal_consistency,
                    "violations": result.sync_violations,
                    "frames_analyzed": result.frames_analyzed
                }
            
            # Create rankings
            comparison["rankings"]["by_avg_offset"] = sorted(
                results, key=lambda x: x.avg_offset_ms
            )
            comparison["rankings"]["by_accuracy"] = sorted(
                results, key=lambda x: x.sync_accuracy_score, reverse=True
            )
            comparison["rankings"]["by_temporal_consistency"] = sorted(
                results, key=lambda x: x.temporal_consistency, reverse=True
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {}
    
    def generate_sync_report(self, metrics: SyncLossMetrics, output_dir: str = "sync_reports"):
        """Generate detailed sync loss report"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            model_id = f"{metrics.model_name}_{metrics.quantization_level}"
            report_path = os.path.join(output_dir, f"sync_report_{model_id}.json")
            
            report = {
                "model_info": {
                    "name": metrics.model_name,
                    "quantization": metrics.quantization_level,
                    "processing_time": metrics.processing_time
                },
                "sync_metrics": {
                    "average_offset_ms": metrics.avg_offset_ms,
                    "maximum_offset_ms": metrics.max_offset_ms,
                    "offset_std_dev": metrics.std_offset_ms,
                    "sync_accuracy_score": metrics.sync_accuracy_score,
                    "audio_visual_correlation": metrics.audio_visual_correlation,
                    "temporal_consistency": metrics.temporal_consistency,
                    "jitter_score": metrics.jitter_score
                },
                "quality_assessment": {
                    "frames_analyzed": metrics.frames_analyzed,
                    "sync_violations": metrics.sync_violations,
                    "violation_rate": metrics.sync_violations / max(metrics.frames_analyzed, 1),
                    "overall_grade": self._calculate_overall_grade(metrics)
                },
                "recommendations": self._generate_recommendations(metrics)
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Sync report saved: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None
    
    def _calculate_overall_grade(self, metrics: SyncLossMetrics) -> str:
        """Calculate overall sync quality grade"""
        if metrics.avg_offset_ms <= self.perfect_sync_threshold and metrics.sync_accuracy_score >= 0.9:
            return "A+ (Excellent)"
        elif metrics.avg_offset_ms <= self.good_sync_threshold and metrics.sync_accuracy_score >= 0.8:
            return "A (Very Good)"
        elif metrics.avg_offset_ms <= self.poor_sync_threshold and metrics.sync_accuracy_score >= 0.7:
            return "B (Good)"
        elif metrics.avg_offset_ms <= 120 and metrics.sync_accuracy_score >= 0.5:
            return "C (Acceptable)"
        else:
            return "D (Poor)"
    
    def _generate_recommendations(self, metrics: SyncLossMetrics) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if metrics.avg_offset_ms > self.good_sync_threshold:
            recommendations.append("Consider using a higher quality model or reducing quantization")
        
        if metrics.temporal_consistency < 0.7:
            recommendations.append("Temporal consistency could be improved with post-processing smoothing")
        
        if metrics.jitter_score > 0.5:
            recommendations.append("High jitter detected - consider temporal filtering")
        
        if metrics.sync_violations > metrics.frames_analyzed * 0.1:
            recommendations.append("High violation rate - model may not be suitable for this content type")
        
        if metrics.audio_visual_correlation < 0.3:
            recommendations.append("Low audio-visual correlation - check input audio quality")
        
        if not recommendations:
            recommendations.append("Sync quality is good - no major improvements needed")
        
        return recommendations
    
    def visualize_sync_analysis(self, frame_data: List[FrameLevelSync], 
                               output_path: str = "sync_analysis.png"):
        """Create visualization of sync analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            timestamps = [f.timestamp_ms / 1000 for f in frame_data]  # Convert to seconds
            
            # Plot 1: Audio vs Visual signals
            audio_energies = [f.audio_energy for f in frame_data]
            mouth_openness = [f.mouth_openness for f in frame_data]
            
            axes[0, 0].plot(timestamps, audio_energies, label='Audio Energy', alpha=0.7)
            axes[0, 0].plot(timestamps, mouth_openness, label='Mouth Openness', alpha=0.7)
            axes[0, 0].set_title('Audio vs Visual Signals')
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('Normalized Amplitude')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Sync Offset Over Time
            offsets = [f.sync_offset_ms for f in frame_data]
            axes[0, 1].plot(timestamps, offsets, color='red', alpha=0.7)
            axes[0, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Perfect Sync')
            axes[0, 1].axhline(y=self.good_sync_threshold, color='orange', linestyle='--', alpha=0.5, label='Good Sync Threshold')
            axes[0, 1].axhline(y=-self.good_sync_threshold, color='orange', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Sync Offset Over Time')
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Sync Offset (ms)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Offset Distribution
            abs_offsets = [abs(o) for o in offsets]
            axes[1, 0].hist(abs_offsets, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].axvline(x=self.perfect_sync_threshold, color='green', linestyle='--', label='Perfect Sync')
            axes[1, 0].axvline(x=self.good_sync_threshold, color='orange', linestyle='--', label='Good Sync')
            axes[1, 0].set_title('Sync Offset Distribution')
            axes[1, 0].set_xlabel('Absolute Offset (ms)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Lip Movement Intensity
            lip_movements = [f.lip_movement_intensity for f in frame_data]
            axes[1, 1].plot(timestamps, lip_movements, color='purple', alpha=0.7)
            axes[1, 1].set_title('Lip Movement Intensity')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Movement Intensity')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Sync analysis visualization saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None


# Global evaluator instance
sync_evaluator = SyncLossEvaluator()