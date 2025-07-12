#!/usr/bin/env python3
"""
Video Personalization Pipeline - Main Entry Point
Simple interface for personalizing videos with template-based TTS
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import VideoPersonalizationPipeline

def validate_video_file(video_path: str) -> bool:
    """Validate that the video file exists and is accessible"""
    video_file = Path(video_path)
    
    if not video_file.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return False
    
    if not video_file.is_file():
        print(f"❌ Error: Path is not a file: {video_path}")
        return False
    
    # Check file extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    if video_file.suffix.lower() not in valid_extensions:
        print(f"⚠️  Warning: Unsupported file extension: {video_file.suffix}")
        print(f"Supported formats: {', '.join(valid_extensions)}")
    
    # Check file size
    file_size_mb = video_file.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:  # 500MB threshold
        print(f"⚠️  Warning: Large file size: {file_size_mb:.1f}MB")
        print("Processing may take longer for large videos")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Personalize videos using template-based TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4 --customer-name "Sarah Johnson" --destination "Tokyo"
  python main.py video.mp4 --customer-name "John Smith" --destination "Paris" --lip-sync
  python main.py video.mp4 --customer-name "Alice" --destination "London" --output-dir results/
        """
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--customer-name", required=True, help="Customer name to use in video")
    parser.add_argument("--destination", required=True, help="Destination to use in video") 
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--lip-sync", action="store_true", help="Apply lip sync (slower but better quality)")
    parser.add_argument("--no-lip-sync", action="store_true", help="Skip lip sync (faster)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level (default: INFO)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate inputs without processing")
    
    args = parser.parse_args()
    
    # Validate video file first
    if not validate_video_file(args.video_path):
        sys.exit(1)
    
    # Determine lip sync setting
    if args.no_lip_sync:
        apply_lip_sync = False
    elif args.lip_sync:
        apply_lip_sync = True
    else:
        # Default: no lip sync for speed
        apply_lip_sync = False
    
    # Variables for replacement
    variables = {
        "customer_name": args.customer_name,
        "destination": args.destination
    }
    
    print("🎬 Video Personalization Pipeline")
    print(f"📹 Input: {args.video_path}")
    print(f"📂 File size: {Path(args.video_path).stat().st_size / (1024*1024):.1f} MB")
    print(f"👤 Customer: {args.customer_name}")
    print(f"🌍 Destination: {args.destination}")
    print(f"🎭 Lip Sync: {'Enabled' if apply_lip_sync else 'Disabled'}")
    print(f"📊 Log Level: {args.log_level}")
    print("-" * 60)
    
    if args.validate_only:
        print("✅ Validation complete. Use --lip-sync or --no-lip-sync to process.")
        return
    
    try:
        # Initialize pipeline with logging
        pipeline = VideoPersonalizationPipeline(
            output_dir=args.output_dir,
            log_level=args.log_level
        )
        
        # Process video
        output_path = pipeline.create_personalized_video(
            video_path=args.video_path,
            variables=variables,
            apply_lip_sync=apply_lip_sync
        )
        
        print("\n" + "="*60)
        print("✅ SUCCESS! Personalized video created:")
        print(f"📁 {output_path}")
        print(f"📊 Output size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        print("📋 Check logs/ directory for detailed processing information")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("  - Check that FFmpeg is installed and accessible")
        print("  - Ensure the video file is not corrupted")
        print("  - Try with --log-level DEBUG for more details")
        print("  - Check logs/ directory for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()