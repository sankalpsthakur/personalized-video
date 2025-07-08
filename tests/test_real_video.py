#!/usr/bin/env python3
"""Test advanced models on real video"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lip_sync.advanced_models import AdvancedModelManager
import time

def test_model_on_real_video(model_name, video_path, audio_path):
    """Test a single model on real video"""
    print(f"\n[{model_name.upper()}]")
    
    manager = AdvancedModelManager()
    model = manager.get_model(model_name)
    req = model.get_requirements()
    
    print(f"Target: {req['resolution'][0]}x{req['resolution'][1]} @ {req['fps']} FPS")
    print("Processing...")
    
    output_path = f"output_{model_name}_real.mp4"
    start = time.time()
    
    try:
        success = model.process_video(video_path, audio_path, output_path)
        elapsed = time.time() - start
        
        if success and os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            speed = 30.6 / elapsed  # Video is 30.6 seconds
            
            print(f"✓ Success!")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Speed: {speed:.2f}x realtime")
            print(f"  Output: {size_mb:.2f}MB")
            return True
        else:
            print("✗ Failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("TESTING ADVANCED MODELS ON REAL VIDEO")
    print("Video: VIDEO-2025-07-05-16-44-05.mp4 (30.6s, 1080x1920)")
    print("="*80)
    
    video = "fixtures/test_video.mp4"
    audio = "fixtures/extracted_audio.wav"
    
    # Test VASA-1
    test_model_on_real_video("vasa1", video, audio)
    
    # Test EMO
    test_model_on_real_video("emo", video, audio)
    
    # Test Gaussian Splatting
    test_model_on_real_video("gaussian_splatting", video, audio)
    
    print("\n" + "="*80)
    print("Testing complete. Check output files:")
    for model in ["vasa1", "emo", "gaussian_splatting"]:
        output = f"output_{model}_real.mp4"
        if os.path.exists(output):
            size = os.path.getsize(output) / (1024 * 1024)
            print(f"  - {output} ({size:.2f}MB)")
    print("="*80)