"""
Enhanced Smart Lip Sync Selector with Quantization and Sync Loss Optimization
Extends the original selector to include quantized models and sync loss as primary criteria
"""

import os
import torch
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

from .advanced_smart_selector import (
    AdvancedSmartLipSyncSelector, SystemCapabilities, VideoCharacteristics, ProcessingOptions
)
from .sync_evaluator import SyncLossEvaluator, SyncLossMetrics
from .quantized_models import QuantizedModelFactory, QuantizationConfig

logger = logging.getLogger(__name__)


@dataclass 
class EnhancedProcessingOptions(ProcessingOptions):
    """Extended processing options with sync and quantization preferences"""
    # Sync loss requirements
    max_sync_loss_ms: float = 40.0  # Maximum acceptable sync loss
    sync_priority: bool = True  # Prioritize sync over other factors
    target_sync_accuracy: float = 0.8  # Target sync accuracy score (0-1)
    
    # Quantization preferences
    allow_quantization: bool = True
    preferred_precision: Optional[str] = None  # fp32, fp16, int8, dynamic, auto
    min_precision: str = "fp16"  # Minimum acceptable precision
    
    # Performance requirements
    min_fps: float = 24.0  # Minimum processing FPS
    max_model_size_mb: float = 2000.0  # Maximum model size
    
    # Sync optimization
    enable_sync_correction: bool = True  # Enable post-processing sync correction
    adaptive_quantization: bool = True  # Adapt quantization based on content
    sync_loss_weight: float = 0.4  # Weight of sync loss in scoring (0-1)


@dataclass
class ModelVariant:
    """Information about a model variant (original + quantized versions)"""
    base_name: str
    precision: str
    model_manager: Any
    estimated_sync_loss_ms: float = 35.0
    sync_accuracy_score: float = 0.85
    model_size_mb: float = 500.0
    avg_fps: float = 30.0
    vram_gb: float = 8.0
    supports_real_time: bool = True
    supports_emotions: bool = False
    supports_3d: bool = False


class EnhancedSmartLipSyncSelector(AdvancedSmartLipSyncSelector):
    """Enhanced selector with quantization and sync loss optimization"""
    
    def __init__(self):
        super().__init__()
        self.sync_evaluator = SyncLossEvaluator()
        self.quantized_factory = QuantizedModelFactory()
        self.model_variants = {}
        self.sync_history = {}  # Track sync performance history
        
        # Initialize quantized models
        self._initialize_quantized_models()
        
        logger.info("Enhanced smart selector initialized with quantization support")
    
    def _initialize_quantized_models(self):
        """Initialize all quantized model variants"""
        try:
            base_models = ["musetalk", "latentsync", "vasa1", "emo", "gaussian_splatting"]
            precision_levels = ["fp32", "fp16", "int8", "dynamic"]
            
            for model_name in base_models:
                self.model_variants[model_name] = {}
                
                for precision in precision_levels:
                    variant_id = f"{model_name}_{precision}"
                    
                    # Create model variant info
                    variant = self._create_model_variant(model_name, precision)
                    if variant:
                        self.model_variants[model_name][precision] = variant
                        logger.debug(f"Initialized variant: {variant_id}")
            
            total_variants = sum(len(variants) for variants in self.model_variants.values())
            logger.info(f"Initialized {total_variants} model variants across {len(base_models)} base models")
            
        except Exception as e:
            logger.error(f"Quantized model initialization failed: {e}")
    
    def _create_model_variant(self, model_name: str, precision: str) -> Optional[ModelVariant]:
        """Create a model variant with estimated performance metrics"""
        try:
            # Base metrics for each model (from documentation and benchmarks)
            base_metrics = {
                "musetalk": {
                    "base_sync_loss": 25.0, "sync_accuracy": 0.90, "base_fps": 30.0,
                    "vram_gb": 6.0, "model_size_mb": 800, "real_time": True, "emotions": False, "3d": False
                },
                "latentsync": {
                    "base_sync_loss": 15.0, "sync_accuracy": 0.98, "base_fps": 22.0,
                    "vram_gb": 12.0, "model_size_mb": 1500, "real_time": False, "emotions": False, "3d": False
                },
                "vasa1": {
                    "base_sync_loss": 20.0, "sync_accuracy": 0.93, "base_fps": 40.0,
                    "vram_gb": 12.0, "model_size_mb": 1200, "real_time": True, "emotions": True, "3d": False
                },
                "emo": {
                    "base_sync_loss": 18.0, "sync_accuracy": 0.95, "base_fps": 25.0,
                    "vram_gb": 16.0, "model_size_mb": 1800, "real_time": False, "emotions": True, "3d": False
                },
                "gaussian_splatting": {
                    "base_sync_loss": 30.0, "sync_accuracy": 0.88, "base_fps": 100.0,
                    "vram_gb": 8.0, "model_size_mb": 600, "real_time": True, "emotions": False, "3d": True
                }
            }
            
            if model_name not in base_metrics:
                return None
            
            base = base_metrics[model_name]
            
            # Apply quantization effects
            precision_factors = self._get_precision_factors(precision)
            
            # Calculate adjusted metrics
            sync_loss = base["base_sync_loss"] * precision_factors["sync_loss_factor"]
            sync_accuracy = base["sync_accuracy"] * precision_factors["accuracy_factor"]
            fps = base["base_fps"] * precision_factors["speed_factor"]
            model_size = base["model_size_mb"] * precision_factors["size_factor"]
            vram = base["vram_gb"] * precision_factors["vram_factor"]
            
            # Try to create actual quantized model manager
            try:
                config = QuantizationConfig(precision=precision)
                model_manager = self.quantized_factory.create_quantized_model(model_name, config)
            except:
                model_manager = None  # Fallback for testing
            
            return ModelVariant(
                base_name=model_name,
                precision=precision,
                model_manager=model_manager,
                estimated_sync_loss_ms=sync_loss,
                sync_accuracy_score=sync_accuracy,
                model_size_mb=model_size,
                avg_fps=fps,
                vram_gb=vram,
                supports_real_time=base["real_time"] and fps >= 24.0,
                supports_emotions=base["emotions"],
                supports_3d=base["3d"]
            )
            
        except Exception as e:
            logger.error(f"Model variant creation failed for {model_name}_{precision}: {e}")
            return None
    
    def _get_precision_factors(self, precision: str) -> Dict[str, float]:
        """Get performance impact factors for different precision levels"""
        factors = {
            "fp32": {
                "sync_loss_factor": 1.0,    # Baseline
                "accuracy_factor": 1.0,     # Best accuracy
                "speed_factor": 1.0,        # Baseline speed
                "size_factor": 1.0,         # Largest size
                "vram_factor": 1.0          # Most VRAM
            },
            "fp16": {
                "sync_loss_factor": 1.1,    # Slight sync loss increase
                "accuracy_factor": 0.98,    # Minimal accuracy loss
                "speed_factor": 1.5,        # ~50% faster
                "size_factor": 0.5,         # Half the size
                "vram_factor": 0.6          # 40% less VRAM
            },
            "int8": {
                "sync_loss_factor": 1.25,   # More sync loss
                "accuracy_factor": 0.92,    # Some accuracy loss
                "speed_factor": 2.0,        # ~2x faster
                "size_factor": 0.25,        # Quarter size
                "vram_factor": 0.4          # 60% less VRAM
            },
            "dynamic": {
                "sync_loss_factor": 1.15,   # Moderate sync loss
                "accuracy_factor": 0.95,    # Minimal accuracy loss
                "speed_factor": 1.3,        # ~30% faster
                "size_factor": 0.7,         # 30% smaller
                "vram_factor": 0.8          # 20% less VRAM
            }
        }
        
        return factors.get(precision, factors["fp32"])
    
    def select_optimal_variant(self, video_path: str, 
                              options: EnhancedProcessingOptions) -> Tuple[str, str]:
        """
        Select optimal model and precision variant based on enhanced criteria
        
        Returns:
            Tuple of (model_name, precision)
        """
        try:
            logger.info("Selecting optimal model variant with sync loss optimization")
            
            # Analyze video characteristics
            video_chars = self.analyze_video(video_path)
            
            # Score all available variants
            variant_scores = {}
            
            for model_name, precisions in self.model_variants.items():
                for precision, variant in precisions.items():
                    if variant is None:
                        continue
                    
                    variant_id = f"{model_name}_{precision}"
                    score = self._score_variant_enhanced(variant, video_chars, options)
                    variant_scores[variant_id] = {
                        "score": score,
                        "variant": variant,
                        "model": model_name,
                        "precision": precision
                    }
                    
                    logger.debug(f"Variant {variant_id}: score = {score:.3f}")
            
            # Select best variant
            if not variant_scores:
                logger.warning("No variants available, using fallback")
                return "audio_only", "fp32"
            
            best_variant = max(variant_scores.items(), key=lambda x: x[1]["score"])
            best_id = best_variant[0]
            best_info = best_variant[1]
            
            selected_model = best_info["model"]
            selected_precision = best_info["precision"]
            
            logger.info(f"Selected optimal variant: {selected_model} ({selected_precision}) "
                       f"with score {best_info['score']:.3f}")
            
            return selected_model, selected_precision
            
        except Exception as e:
            logger.error(f"Variant selection failed: {e}")
            return "audio_only", "fp32"
    
    def _score_variant_enhanced(self, variant: ModelVariant, 
                               video_chars: VideoCharacteristics,
                               options: EnhancedProcessingOptions) -> float:
        """Enhanced scoring with sync loss as primary factor"""
        try:
            score = 0.0
            
            # 1. Sync Loss Score (Primary factor - 40% weight by default)
            sync_weight = options.sync_loss_weight
            if options.sync_priority:
                sync_weight *= 1.5  # Boost sync importance
            
            # Score based on sync loss (lower is better)
            if variant.estimated_sync_loss_ms <= options.max_sync_loss_ms:
                sync_score = 100.0 * (1.0 - variant.estimated_sync_loss_ms / 100.0)
                sync_score = max(sync_score, 0.0)
            else:
                sync_score = -50.0  # Heavy penalty for exceeding sync threshold
            
            score += sync_score * sync_weight
            
            # 2. Sync Accuracy Score (15% weight)
            accuracy_weight = 0.15
            if variant.sync_accuracy_score >= options.target_sync_accuracy:
                accuracy_score = variant.sync_accuracy_score * 100.0
            else:
                accuracy_score = variant.sync_accuracy_score * 50.0  # Penalty
            
            score += accuracy_score * accuracy_weight
            
            # 3. Performance Score (20% weight)
            perf_weight = 0.20
            if variant.avg_fps >= options.min_fps:
                fps_score = min(variant.avg_fps / 60.0, 1.0) * 100.0
            else:
                fps_score = -30.0  # Penalty for insufficient FPS
            
            score += fps_score * perf_weight
            
            # 4. Hardware Compatibility (15% weight)
            hw_weight = 0.15
            hw_score = 0.0
            
            if self.system_caps.has_cuda and self.system_caps.available_vram_gb >= variant.vram_gb:
                hw_score = 80.0  # Good GPU match
            elif self.system_caps.has_mps and variant.vram_gb <= 8.0:
                hw_score = 60.0  # Apple Silicon compatible
            elif variant.vram_gb <= 2.0:
                hw_score = 40.0  # CPU-friendly model
            else:
                hw_score = -20.0  # Hardware mismatch
            
            score += hw_score * hw_weight
            
            # 5. Feature Requirements (10% weight)
            feature_weight = 0.10
            feature_score = 50.0  # Base score
            
            if options.require_real_time and variant.supports_real_time:
                feature_score += 30.0
            elif options.require_real_time and not variant.supports_real_time:
                feature_score -= 40.0
            
            if options.enable_emotions and variant.supports_emotions:
                feature_score += 20.0
            elif options.enable_emotions and not variant.supports_emotions:
                feature_score -= 15.0
            
            if options.enable_3d and variant.supports_3d:
                feature_score += 20.0
            elif options.enable_3d and not variant.supports_3d:
                feature_score -= 15.0
            
            score += feature_score * feature_weight
            
            # 6. Quantization Preference Bonus/Penalty
            if options.allow_quantization:
                if variant.precision == options.preferred_precision:
                    score += 10.0  # Preference bonus
                
                # Check minimum precision requirement
                precision_order = {"fp32": 4, "fp16": 3, "dynamic": 2, "int8": 1}
                min_level = precision_order.get(options.min_precision, 4)
                variant_level = precision_order.get(variant.precision, 4)
                
                if variant_level < min_level:
                    score -= 30.0  # Below minimum precision
            else:
                if variant.precision != "fp32":
                    score -= 20.0  # Quantization not allowed
            
            # 7. Model Size Constraint
            if variant.model_size_mb > options.max_model_size_mb:
                score -= 25.0  # Size penalty
            
            # 8. Historical Performance Bonus
            variant_id = f"{variant.base_name}_{variant.precision}"
            if variant_id in self.sync_history:
                historical_sync = self.sync_history[variant_id].get("avg_sync_loss", 50.0)
                if historical_sync < options.max_sync_loss_ms:
                    score += 5.0  # Historical performance bonus
            
            return max(score, 0.0)  # Ensure non-negative score
            
        except Exception as e:
            logger.error(f"Enhanced variant scoring failed: {e}")
            return 0.0
    
    def process_video_enhanced(self, video_path: str, audio_path: str, output_path: str,
                              options: Optional[EnhancedProcessingOptions] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Process video with enhanced model selection and sync optimization
        
        Returns:
            Tuple of (success, detailed_results)
        """
        try:
            if options is None:
                options = EnhancedProcessingOptions()
            
            start_time = time.time()
            
            # Select optimal variant
            model_name, precision = self.select_optimal_variant(video_path, options)
            variant_id = f"{model_name}_{precision}"
            
            logger.info(f"Processing with enhanced selection: {variant_id}")
            
            # Get the variant
            variant = None
            if model_name in self.model_variants and precision in self.model_variants[model_name]:
                variant = self.model_variants[model_name][precision]
            
            # Process video
            processing_success = False
            if variant and variant.model_manager:
                try:
                    processing_success = variant.model_manager.process_video(
                        video_path, audio_path, output_path, precision
                    )
                except:
                    logger.warning(f"Quantized processing failed, trying fallback")
            
            # Fallback to original selector if needed
            if not processing_success:
                logger.info("Falling back to original selector")
                original_options = ProcessingOptions(
                    quality_priority=options.quality_priority,
                    max_cost_usd=options.max_cost_usd,
                    require_real_time=options.require_real_time,
                    enable_emotions=options.enable_emotions,
                    enable_3d=options.enable_3d
                )
                processing_success, fallback_method = self.process_video(
                    video_path, audio_path, output_path, original_options
                )
                variant_id = fallback_method
            
            processing_time = time.time() - start_time
            
            # Evaluate sync loss if processing succeeded
            sync_metrics = None
            if processing_success and os.path.exists(output_path):
                try:
                    sync_metrics = self.sync_evaluator.evaluate_video_sync(
                        output_path, audio_path, model_name, precision
                    )
                    
                    # Update sync history
                    self._update_sync_history(variant_id, sync_metrics)
                    
                except Exception as e:
                    logger.warning(f"Sync evaluation failed: {e}")
            
            # Compile results
            results = {
                "success": processing_success,
                "selected_model": model_name,
                "selected_precision": precision,
                "variant_id": variant_id,
                "processing_time": processing_time,
                "sync_metrics": sync_metrics,
                "estimated_sync_loss": variant.estimated_sync_loss_ms if variant else None,
                "meets_sync_requirements": (
                    sync_metrics.avg_offset_ms <= options.max_sync_loss_ms 
                    if sync_metrics else False
                )
            }
            
            if sync_metrics:
                logger.info(f"Actual sync loss: {sync_metrics.avg_offset_ms:.1f}ms "
                           f"(estimated: {variant.estimated_sync_loss_ms:.1f}ms)")
            
            return processing_success, results
            
        except Exception as e:
            logger.error(f"Enhanced video processing failed: {e}")
            return False, {"error": str(e)}
    
    def _update_sync_history(self, variant_id: str, sync_metrics: SyncLossMetrics):
        """Update historical sync performance data"""
        try:
            if variant_id not in self.sync_history:
                self.sync_history[variant_id] = {
                    "samples": 0,
                    "total_sync_loss": 0.0,
                    "avg_sync_loss": 0.0,
                    "best_sync_loss": float('inf'),
                    "worst_sync_loss": 0.0
                }
            
            history = self.sync_history[variant_id]
            history["samples"] += 1
            history["total_sync_loss"] += sync_metrics.avg_offset_ms
            history["avg_sync_loss"] = history["total_sync_loss"] / history["samples"]
            history["best_sync_loss"] = min(history["best_sync_loss"], sync_metrics.avg_offset_ms)
            history["worst_sync_loss"] = max(history["worst_sync_loss"], sync_metrics.avg_offset_ms)
            
            logger.debug(f"Updated sync history for {variant_id}: "
                        f"avg={history['avg_sync_loss']:.1f}ms over {history['samples']} samples")
            
        except Exception as e:
            logger.error(f"Sync history update failed: {e}")
    
    def get_sync_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive sync performance report"""
        try:
            report = {
                "total_variants": sum(len(precisions) for precisions in self.model_variants.values()),
                "tested_variants": len(self.sync_history),
                "best_performers": {},
                "worst_performers": {},
                "precision_analysis": {},
                "model_analysis": {}
            }
            
            if not self.sync_history:
                return report
            
            # Find best and worst performers
            sorted_by_sync = sorted(
                self.sync_history.items(), 
                key=lambda x: x[1]["avg_sync_loss"]
            )
            
            report["best_performers"] = dict(sorted_by_sync[:3])  # Top 3
            report["worst_performers"] = dict(sorted_by_sync[-3:])  # Bottom 3
            
            # Analyze by precision
            precision_stats = {}
            for variant_id, stats in self.sync_history.items():
                precision = variant_id.split('_')[-1]
                if precision not in precision_stats:
                    precision_stats[precision] = {
                        "count": 0,
                        "total_sync_loss": 0.0,
                        "avg_sync_loss": 0.0
                    }
                
                precision_stats[precision]["count"] += 1
                precision_stats[precision]["total_sync_loss"] += stats["avg_sync_loss"]
            
            for precision, stats in precision_stats.items():
                stats["avg_sync_loss"] = stats["total_sync_loss"] / stats["count"]
            
            report["precision_analysis"] = precision_stats
            
            # Analyze by model
            model_stats = {}
            for variant_id, stats in self.sync_history.items():
                model = '_'.join(variant_id.split('_')[:-1])  # Remove precision suffix
                if model not in model_stats:
                    model_stats[model] = {
                        "count": 0,
                        "total_sync_loss": 0.0,
                        "avg_sync_loss": 0.0
                    }
                
                model_stats[model]["count"] += 1
                model_stats[model]["total_sync_loss"] += stats["avg_sync_loss"]
            
            for model, stats in model_stats.items():
                stats["avg_sync_loss"] = stats["total_sync_loss"] / stats["count"]
            
            report["model_analysis"] = model_stats
            
            return report
            
        except Exception as e:
            logger.error(f"Sync performance report generation failed: {e}")
            return {}
    
    def recommend_optimal_settings(self, use_case: str = "general") -> EnhancedProcessingOptions:
        """Recommend optimal settings for different use cases"""
        try:
            use_case_configs = {
                "general": EnhancedProcessingOptions(
                    max_sync_loss_ms=40.0,
                    sync_priority=True,
                    preferred_precision="fp16",
                    min_fps=24.0
                ),
                "high_quality": EnhancedProcessingOptions(
                    max_sync_loss_ms=20.0,
                    sync_priority=True,
                    preferred_precision="fp32",
                    min_fps=20.0,
                    sync_loss_weight=0.6
                ),
                "real_time": EnhancedProcessingOptions(
                    max_sync_loss_ms=60.0,
                    sync_priority=False,
                    preferred_precision="int8",
                    min_fps=30.0,
                    require_real_time=True
                ),
                "mobile": EnhancedProcessingOptions(
                    max_sync_loss_ms=50.0,
                    preferred_precision="int8",
                    max_model_size_mb=200.0,
                    min_fps=24.0
                ),
                "precision": EnhancedProcessingOptions(
                    max_sync_loss_ms=15.0,
                    sync_priority=True,
                    preferred_precision="fp32",
                    allow_quantization=False,
                    sync_loss_weight=0.8
                )
            }
            
            config = use_case_configs.get(use_case, use_case_configs["general"])
            logger.info(f"Recommended settings for {use_case}: max_sync_loss={config.max_sync_loss_ms}ms, "
                       f"precision={config.preferred_precision}")
            
            return config
            
        except Exception as e:
            logger.error(f"Settings recommendation failed: {e}")
            return EnhancedProcessingOptions()


# Global enhanced selector instance
enhanced_selector = EnhancedSmartLipSyncSelector()