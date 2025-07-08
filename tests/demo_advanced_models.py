#!/usr/bin/env python3
"""
Demo script showcasing advanced lip sync models
Shows VASA-1, EMO, and Gaussian Splatting in action
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import json
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_models import AdvancedModelManager
from lip_sync_advanced import ExtendedLipSyncProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_files():
    """Create demo video and audio files"""
    # Check if files exist
    if os.path.exists("demo_video.mp4") and os.path.exists("demo_audio.wav"):
        logger.info("Demo files already exist")
        return "demo_video.mp4", "demo_audio.wav"
    
    logger.info("Creating demo files...")
    
    # Create a 10-second demo video with a face-like pattern
    video_cmd = [
        "ffmpeg", "-f", "lavfi", 
        "-i", "color=c=blue:s=512x512:d=10",
        "-filter_complex", 
        "[0:v]drawbox=x=156:y=156:w=200:h=200:c=white:t=fill,"
        "drawcircle=x=206:y=206:r=20:c=black:t=fill,"
        "drawcircle=x=306:y=206:r=20:c=black:t=fill,"
        "drawellipse=x=256:y=256:w=60:h=30:c=red:t=fill[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-y", "demo_video.mp4"
    ]
    
    subprocess.run(video_cmd, check=True, capture_output=True)
    
    # Create demo audio with speech-like variations
    audio_cmd = [
        "ffmpeg", "-f", "lavfi",
        "-i", "sine=frequency=200:duration=10:sample_rate=16000",
        "-filter_complex",
        "[0:a]volume=0.5[a]",
        "-map", "[a]",
        "-ar", "16000", "-ac", "1",
        "-y", "demo_audio.wav"
    ]
    
    subprocess.run(audio_cmd, check=True, capture_output=True)
    
    logger.info("✓ Demo files created")
    return "demo_video.mp4", "demo_audio.wav"


def benchmark_model(model_name: str, video_path: str, audio_path: str) -> dict:
    """Benchmark a single model"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name.upper()}")
    logger.info(f"{'='*60}")
    
    try:
        # Initialize model
        manager = AdvancedModelManager()
        model = manager.get_model(model_name)
        
        # Get requirements
        requirements = model.get_requirements()
        logger.info(f"Resolution: {requirements['resolution'][0]}x{requirements['resolution'][1]}")
        logger.info(f"Target FPS: {requirements['fps']}")
        logger.info(f"VRAM Required: {requirements['vram']}GB")
        logger.info(f"Features: {', '.join(requirements['features'])}")
        
        # Process video
        output_path = f"output_{model_name}_demo.mp4"
        start_time = time.time()
        
        success = model.process_video(video_path, audio_path, output_path)
        
        processing_time = time.time() - start_time
        
        if success and os.path.exists(output_path):
            # Get output file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            # Calculate effective FPS
            video_duration = 10.0  # seconds
            effective_fps = video_duration / processing_time if processing_time > 0 else 0
            
            result = {
                "model": model_name,
                "success": True,
                "processing_time": processing_time,
                "file_size_mb": file_size,
                "effective_fps": effective_fps,
                "target_fps": requirements['fps'],
                "vram_gb": requirements['vram'],
                "resolution": requirements['resolution']
            }
            
            logger.info(f"✓ Processing completed in {processing_time:.2f}s")
            logger.info(f"✓ Output size: {file_size:.2f}MB")
            logger.info(f"✓ Effective processing speed: {effective_fps:.1f}x realtime")
            
        else:
            result = {
                "model": model_name,
                "success": False,
                "error": "Processing failed"
            }
            logger.error("✗ Processing failed")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing {model_name}: {e}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e)
        }


def test_pipeline_integration():
    """Test integration with main pipeline"""
    logger.info("\n" + "="*60)
    logger.info("Testing Pipeline Integration")
    logger.info("="*60)
    
    # Test ExtendedLipSyncProcessor
    models = ["musetalk", "vasa1", "emo", "gaussian_splatting"]
    
    for model_name in models:
        try:
            processor = ExtendedLipSyncProcessor(model_name)
            logger.info(f"✓ {model_name}: Initialized successfully")
            
            if hasattr(processor, 'is_advanced'):
                if processor.is_advanced:
                    logger.info(f"  - Type: Advanced Model")
                else:
                    logger.info(f"  - Type: Classic Model")
                    
        except Exception as e:
            logger.error(f"✗ {model_name}: Failed to initialize - {e}")


def generate_comparison_report(results: list):
    """Generate a comparison report"""
    print("\n" + "="*80)
    print("ADVANCED MODELS PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Status':<10} {'Time (s)':<10} {'Speed':<15} {'VRAM':<10} {'Resolution':<15}")
    print("-"*80)
    
    for result in results:
        if result['success']:
            speed = f"{result['effective_fps']:.1f}x realtime"
            resolution = f"{result['resolution'][0]}x{result['resolution'][1]}"
            print(f"{result['model']:<20} {'Success':<10} {result['processing_time']:<10.2f} "
                  f"{speed:<15} {result['vram_gb']:<10}GB {resolution:<15}")
        else:
            print(f"{result['model']:<20} {'Failed':<10} {'-':<10} {'-':<15} {'-':<10} {'-':<15}")
    
    print("="*80)
    
    # Key insights
    print("\nKEY INSIGHTS:")
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        # Fastest model
        fastest = min(successful_results, key=lambda x: x['processing_time'])
        print(f"- Fastest: {fastest['model']} ({fastest['processing_time']:.2f}s)")
        
        # Highest FPS
        highest_fps = max(successful_results, key=lambda x: x['target_fps'])
        print(f"- Highest FPS: {highest_fps['model']} ({highest_fps['target_fps']} FPS)")
        
        # Most efficient (speed vs VRAM)
        efficiency_scores = {
            r['model']: r['effective_fps'] / r['vram_gb'] 
            for r in successful_results
        }
        most_efficient = max(efficiency_scores, key=efficiency_scores.get)
        print(f"- Most Efficient: {most_efficient} ({efficiency_scores[most_efficient]:.2f} speed/VRAM ratio)")
    
    print("\nRECOMMENDATIONS:")
    print("- Real-time applications: Use Gaussian Splatting (100+ FPS)")
    print("- Quality focus: Use VASA-1 or EMO (512x512 with expressions)")
    print("- Limited resources: Use classic models (Wav2Lip)")
    print("- Balanced approach: Use MuseTalk (production-ready)")


def main():
    """Main demo function"""
    print("\n" + "="*80)
    print("ADVANCED LIP SYNC MODELS DEMONSTRATION")
    print("Showcasing VASA-1, EMO, and 3D Gaussian Splatting")
    print("="*80)
    
    # Create demo files
    video_path, audio_path = create_demo_files()
    
    # Test each model
    models = ["vasa1", "emo", "gaussian_splatting"]
    results = []
    
    for model_name in models:
        result = benchmark_model(model_name, video_path, audio_path)
        results.append(result)
    
    # Test pipeline integration
    test_pipeline_integration()
    
    # Generate comparison report
    generate_comparison_report(results)
    
    # Save results to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }, f, indent=2)
    
    logger.info("\n✓ Benchmark results saved to benchmark_results.json")
    
    # List output files
    print("\nOUTPUT FILES CREATED:")
    for model_name in models:
        output_file = f"output_{model_name}_demo.mp4"
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"- {output_file} ({size_mb:.2f}MB)")
    
    print("\n✓ Demo complete! Check the output files to see the results.")


if __name__ == "__main__":
    main()