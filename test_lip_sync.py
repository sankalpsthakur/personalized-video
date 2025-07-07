#!/usr/bin/env python3
"""
Test script for lip sync functionality
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import json
import cv2
import numpy as np
from datetime import datetime


def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    print("-" * 50)
    
    dependencies = {
        "torch": "PyTorch",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "gtts": "Google TTS",
        "whisper": "OpenAI Whisper",
        "mediapipe": "MediaPipe (face detection)",
        "imageio": "ImageIO",
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(module)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def test_model_availability():
    """Test which lip sync models are available"""
    print("\nTesting model availability...")
    print("-" * 50)
    
    try:
        from lip_sync import LipSyncProcessor
        processor = LipSyncProcessor()
        models = processor.get_available_models()
        
        print(f"{'Model':<12} {'Available':<10} {'Quality':<10} {'VRAM Req':<10}")
        print("-" * 42)
        
        available_count = 0
        for model in models:
            status = "Yes" if model["available"] else "No"
            if model["available"]:
                available_count += 1
            print(f"{model['name']:<12} {status:<10} {model['quality']:<10} {model['vram_required']}GB")
        
        print(f"\n{available_count}/{len(models)} models available")
        return available_count > 0
        
    except Exception as e:
        print(f"Error testing models: {e}")
        return False


def test_face_detection():
    """Test face detection functionality"""
    print("\nTesting face detection...")
    print("-" * 50)
    
    try:
        from lip_sync import LipSyncProcessor
        
        # Create test image with a simple face-like pattern
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Draw a simple face
        # Face circle
        cv2.circle(test_image, (320, 240), 100, (200, 150, 100), -1)
        # Eyes
        cv2.circle(test_image, (290, 220), 20, (50, 50, 50), -1)
        cv2.circle(test_image, (350, 220), 20, (50, 50, 50), -1)
        # Mouth
        cv2.ellipse(test_image, (320, 280), (40, 20), 0, 0, 180, (50, 50, 50), -1)
        
        # Save test image
        cv2.imwrite("test_face.jpg", test_image)
        
        # Test detection
        processor = LipSyncProcessor()
        bbox = processor.detect_face(test_image)
        
        if bbox:
            print("✓ Face detection working")
            print(f"  Detected at: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
            
            # Draw bounding box on image
            x, y, w, h = bbox
            cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite("test_face_detected.jpg", test_image)
            print("  Saved visualization to test_face_detected.jpg")
            
            return True
        else:
            print("✗ No face detected in test image")
            print("  Note: Simple geometric face may not be detected by advanced models")
            return False
            
    except Exception as e:
        print(f"Error testing face detection: {e}")
        return False


def create_test_video():
    """Create a simple test video with audio"""
    print("\nCreating test video...")
    print("-" * 50)
    
    try:
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        # Generate test audio using gTTS
        try:
            from gtts import gTTS
            text = "Hello, my name is John and I am going to Paris"
            tts = gTTS(text=text, lang='en')
            audio_path = temp_dir / "test_audio.mp3"
            tts.save(str(audio_path))
            print("✓ Generated test audio with gTTS")
        except:
            # Fallback: create silent audio
            audio_path = temp_dir / "test_audio.wav"
            cmd = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=3",
                   "-y", str(audio_path)]
            subprocess.run(cmd, capture_output=True)
            print("✓ Generated silent test audio")
        
        # Create video frames
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        
        width, height = 640, 480
        fps = 30
        duration = 3  # seconds
        total_frames = fps * duration
        
        for i in range(total_frames):
            # Create frame with moving circle (simulating face)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Animated face position
            center_x = int(width/2 + 50 * np.sin(2 * np.pi * i / fps))
            center_y = int(height/2)
            
            # Draw face
            cv2.circle(frame, (center_x, center_y), 80, (200, 150, 100), -1)
            cv2.circle(frame, (center_x-25, center_y-20), 15, (50, 50, 50), -1)
            cv2.circle(frame, (center_x+25, center_y-20), 15, (50, 50, 50), -1)
            
            # Animated mouth (simulate talking)
            mouth_height = int(15 + 10 * np.sin(8 * np.pi * i / fps))
            cv2.ellipse(frame, (center_x, center_y+30), (30, mouth_height), 
                       0, 0, 180, (50, 50, 50), -1)
            
            # Add frame number
            cv2.putText(frame, f"Frame {i+1}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            frame_path = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
        
        print(f"✓ Generated {total_frames} video frames")
        
        # Combine frames into video
        video_path = "test_video.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-i", str(audio_path),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-shortest",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print(f"✓ Created test video: {video_path}")
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)
            
            return video_path
        else:
            print(f"✗ Failed to create video: {result.stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"Error creating test video: {e}")
        return None


def test_lip_sync_models(video_path):
    """Test lip sync with each available model"""
    print("\nTesting lip sync models...")
    print("-" * 50)
    
    if not video_path or not os.path.exists(video_path):
        print("✗ No test video available")
        return False
    
    try:
        from lip_sync import LipSyncProcessor, print_model_comparison
        
        # Show model comparison
        print_model_comparison()
        
        # Test each model
        models_to_test = ["musetalk", "wav2lip", "latentsync"]
        
        for model_name in models_to_test:
            print(f"\nTesting {model_name}...")
            
            try:
                # Initialize processor
                processor = LipSyncProcessor(model_type=model_name)
                
                # Check if model is available
                models = processor.get_available_models()
                model_info = next(m for m in models if m["name"] == model_name)
                
                if not model_info["available"]:
                    print(f"  ✗ {model_name} model files not found")
                    continue
                
                if model_info.get("vram_sufficient") is False:
                    print(f"  ✗ Insufficient VRAM for {model_name}")
                    continue
                
                # Test lip sync on a segment
                output_path = f"test_output_{model_name}.mp4"
                segments = [{
                    "start": 0.5,
                    "end": 2.5,
                    "audio": video_path  # Using same audio for simplicity
                }]
                
                print(f"  Processing test segment...")
                success = processor.apply_lip_sync(video_path, segments, output_path)
                
                if success and os.path.exists(output_path):
                    size = os.path.getsize(output_path) / 1024 / 1024
                    print(f"  ✓ {model_name} test completed ({size:.1f} MB)")
                else:
                    print(f"  ✗ {model_name} test failed")
                    
            except Exception as e:
                print(f"  ✗ Error testing {model_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error in lip sync tests: {e}")
        return False


def test_full_pipeline():
    """Test the full personalization pipeline"""
    print("\nTesting full pipeline...")
    print("-" * 50)
    
    video_path = "test_video.mp4"
    if not os.path.exists(video_path):
        print("✗ Test video not found")
        return False
    
    try:
        # Run pipeline with lip sync
        cmd = [
            sys.executable, "personalization_pipeline.py", video_path,
            "--customer-name", "Alice",
            "--destination", "Tokyo",
            "--output-dir", "test_output",
            "--lip-sync-model", "musetalk"
        ]
        
        print("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Pipeline completed successfully")
            
            # Check output
            output_dir = Path("test_output")
            if output_dir.exists():
                files = list(output_dir.glob("*.mp4"))
                if files:
                    print(f"  Output: {files[0]}")
                    size = files[0].stat().st_size / 1024 / 1024
                    print(f"  Size: {size:.1f} MB")
                    return True
        else:
            print("✗ Pipeline failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error testing pipeline: {e}")
    
    return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Lip Sync Test Suite")
    print("=" * 50)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Track test results
    results = {}
    
    # Test 1: Dependencies
    results["Dependencies"] = test_dependencies()
    if not results["Dependencies"]:
        print("\n⚠️  Cannot continue without dependencies")
        return
    
    # Test 2: Model availability
    results["Model Availability"] = test_model_availability()
    
    # Test 3: Face detection
    results["Face Detection"] = test_face_detection()
    
    # Test 4: Create test video
    video_path = create_test_video()
    results["Test Video Creation"] = video_path is not None
    
    # Test 5: Lip sync models
    if video_path:
        results["Lip Sync Models"] = test_lip_sync_models(video_path)
    
    # Test 6: Full pipeline
    if video_path:
        results["Full Pipeline"] = test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test:<25} [{status}]")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    # Cleanup
    print("\nCleaning up test files...")
    test_files = ["test_face.jpg", "test_face_detected.jpg", "test_video.mp4",
                  "test_output_musetalk.mp4", "test_output_wav2lip.mp4", 
                  "test_output_latentsync.mp4"]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  Removed {file}")


if __name__ == "__main__":
    main()