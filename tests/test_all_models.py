#!/usr/bin/env python3
"""
Test all lip sync models with a sample video
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_simple_test_video(duration=5):
    """Create a simple test video with a talking face animation"""
    logger.info("Creating test video...")
    
    try:
        import cv2
        import numpy as np
        from gtts import gTTS
    except ImportError:
        logger.error("Required packages not installed")
        return None
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Generate test speech
    text = "Hello, my name is John and I am going to Paris for vacation"
    tts = gTTS(text=text, lang='en')
    audio_path = temp_dir / "test_audio.mp3"
    tts.save(str(audio_path))
    
    # Create animated face frames
    fps = 30
    width, height = 640, 480
    total_frames = fps * duration
    
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir()
    
    for i in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Animated face position
        t = i / fps
        center_x = int(width/2 + 30 * np.sin(2 * np.pi * t / 3))
        center_y = int(height/2)
        
        # Draw face
        cv2.circle(frame, (center_x, center_y), 100, (200, 180, 160), -1)
        
        # Eyes
        cv2.circle(frame, (center_x-30, center_y-30), 12, (50, 50, 50), -1)
        cv2.circle(frame, (center_x+30, center_y-30), 12, (50, 50, 50), -1)
        
        # Animated mouth (simulate talking)
        mouth_height = int(20 + 15 * abs(np.sin(10 * np.pi * t)))
        cv2.ellipse(frame, (center_x, center_y+40), (40, mouth_height), 
                   0, 0, 180, (150, 50, 50), -1)
        
        # Save frame
        cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), frame)
    
    # Create video
    video_path = "test_input.mp4"
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
        logger.info(f"✓ Created test video: {video_path}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return video_path
    else:
        logger.error(f"Failed to create video: {result.stderr.decode()}")
        return None


def test_model(model_name, video_path):
    """Test a specific lip sync model"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name.upper()} model")
    logger.info(f"{'='*60}")
    
    output_path = f"test_output_{model_name}.mp4"
    
    cmd = [
        sys.executable, "personalization_pipeline.py",
        video_path,
        "--customer-name", "Alice",
        "--destination", "Tokyo",
        "--output-dir", f"output_{model_name}",
        "--lip-sync-model", model_name
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        logger.info(f"✓ {model_name} completed in {elapsed_time:.1f}s")
        
        # Check output
        output_dir = Path(f"output_{model_name}")
        if output_dir.exists():
            videos = list(output_dir.glob("*.mp4"))
            if videos:
                output_file = videos[0]
                size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"  Output: {output_file} ({size_mb:.1f} MB)")
                
                # Move to test outputs directory
                test_outputs = Path("test_outputs")
                test_outputs.mkdir(exist_ok=True)
                
                final_path = test_outputs / f"{model_name}_output.mp4"
                output_file.rename(final_path)
                logger.info(f"  Moved to: {final_path}")
    else:
        logger.error(f"✗ {model_name} failed:")
        if result.stdout:
            logger.error(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")


def main():
    """Test all models"""
    print("\n" + "="*60)
    print("Lip Sync Models Test Suite")
    print("="*60)
    
    # Create test video if it doesn't exist
    test_video = "test_input.mp4"
    if not os.path.exists(test_video):
        test_video = create_simple_test_video()
        if not test_video:
            logger.error("Failed to create test video")
            return
    else:
        logger.info(f"Using existing test video: {test_video}")
    
    # Check available models first
    logger.info("\nChecking available models...")
    result = subprocess.run(
        [sys.executable, "personalization_pipeline.py", "--list-models"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
    
    # Test each model
    models = ["musetalk", "wav2lip", "latentsync"]
    
    logger.info("\nStarting model tests...")
    
    for model in models:
        test_model(model, test_video)
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    
    test_outputs = Path("test_outputs")
    if test_outputs.exists():
        outputs = list(test_outputs.glob("*.mp4"))
        logger.info(f"Generated {len(outputs)} output videos:")
        for output in outputs:
            size_mb = output.stat().st_size / (1024 * 1024)
            logger.info(f"  - {output.name}: {size_mb:.1f} MB")
    
    logger.info("\nTo compare results:")
    logger.info("  1. Check test_outputs/ directory")
    logger.info("  2. Compare visual quality")
    logger.info("  3. Check processing times in logs")


if __name__ == "__main__":
    main()