#!/usr/bin/env python3
"""
Download script for all advanced lip sync models
Downloads MuseTalk, LatentSync, VASA-1, EMO, and Gaussian Splatting models
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def setup_logging():
    """Setup logging for model downloads"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('model_download.log')
        ]
    )


def download_musetalk_models():
    """Download MuseTalk models"""
    print("\n📥 Downloading MuseTalk models...")
    
    try:
        from src.lip_sync.musetalk_model import musetalk_model
        
        success = musetalk_model.download_models()
        
        if success:
            print("✅ MuseTalk models downloaded successfully")
            return True
        else:
            print("❌ Failed to download MuseTalk models")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading MuseTalk: {e}")
        return False


def download_latentsync_models():
    """Download LatentSync models"""
    print("\n📥 Downloading LatentSync models...")
    
    try:
        from src.lip_sync.latentsync_model import latentsync_model
        
        success = latentsync_model.download_models()
        
        if success:
            print("✅ LatentSync models downloaded successfully")
            return True
        else:
            print("❌ Failed to download LatentSync models")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading LatentSync: {e}")
        return False


def download_vasa1_models():
    """Download VASA-1 models"""
    print("\n📥 Downloading VASA-1 models...")
    
    try:
        from src.lip_sync.vasa1_model import vasa1_model
        
        success = vasa1_model.download_models()
        
        if success:
            print("✅ VASA-1 models downloaded successfully")
            return True
        else:
            print("❌ Failed to download VASA-1 models")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading VASA-1: {e}")
        return False


def download_emo_models():
    """Download EMO models"""
    print("\n📥 Downloading EMO models...")
    
    try:
        from src.lip_sync.emo_model import emo_model
        
        success = emo_model.download_models()
        
        if success:
            print("✅ EMO models downloaded successfully")
            return True
        else:
            print("❌ Failed to download EMO models")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading EMO: {e}")
        return False


def download_gaussian_splatting_models():
    """Download Gaussian Splatting models"""
    print("\n📥 Downloading Gaussian Splatting models...")
    
    try:
        from src.lip_sync.gaussian_splatting_model import gaussian_splatting_model
        
        success = gaussian_splatting_model.download_models()
        
        if success:
            print("✅ Gaussian Splatting models downloaded successfully")
            return True
        else:
            print("❌ Failed to download Gaussian Splatting models")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading Gaussian Splatting: {e}")
        return False


def check_system_requirements():
    """Check system requirements for all models"""
    print("\n🔍 Checking system requirements...")
    
    import torch
    
    # Check CUDA
    has_cuda = torch.cuda.is_available()
    print(f"CUDA Available: {has_cuda}")
    
    if has_cuda:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU VRAM: {vram_gb:.1f} GB")
        
        # Check requirements for each model
        models_requirements = {
            "MuseTalk": 6.0,
            "LatentSync": 12.0,
            "VASA-1": 12.0,
            "EMO": 16.0,
            "Gaussian Splatting": 8.0
        }
        
        print("\nModel compatibility:")
        compatible_models = []
        
        for model_name, required_vram in models_requirements.items():
            compatible = vram_gb >= required_vram
            status = "✅" if compatible else "❌"
            print(f"  {status} {model_name}: {required_vram}GB required")
            
            if compatible:
                compatible_models.append(model_name)
        
        print(f"\nCompatible models: {len(compatible_models)}/5")
        return compatible_models
    
    else:
        print("❌ No CUDA GPU detected. Advanced models require GPU acceleration.")
        print("Only cloud-based methods will be available.")
        return []


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing required dependencies...")
    
    dependencies = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "mediapipe>=0.10.0",
        "librosa>=0.10.0",
        "huggingface_hub>=0.16.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "requests>=2.31.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {dep} installed")
            else:
                print(f"❌ Failed to install {dep}: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error installing {dep}: {e}")


def get_model_info():
    """Get information about all models"""
    
    model_info = {
        "MuseTalk": {
            "description": "Real-time high quality lip sync with latent space inpainting",
            "fps": "30+ FPS",
            "vram": "6GB minimum, 8GB recommended",
            "resolution": "256x256 face region",
            "quality": "9.2/10",
            "features": ["Real-time", "High quality"],
            "source": "https://github.com/TMElyralab/MuseTalk"
        },
        "LatentSync": {
            "description": "Stable Diffusion-based lip sync for highest quality",
            "fps": "20-24 FPS",
            "vram": "12GB minimum, 20GB recommended", 
            "resolution": "512x512",
            "quality": "9.8/10",
            "features": ["Highest quality", "Stable Diffusion"],
            "source": "https://github.com/bytedance/LatentSync"
        },
        "VASA-1": {
            "description": "Microsoft's expressive talking face generation",
            "fps": "40 FPS (1.33x real-time)",
            "vram": "12GB minimum, 16GB recommended",
            "resolution": "512x512", 
            "quality": "9.5/10",
            "features": ["Real-time", "Emotions", "Expressions"],
            "source": "Microsoft Research VASA-1"
        },
        "EMO": {
            "description": "Emote Portrait Alive with emotional expressions",
            "fps": "25 FPS",
            "vram": "16GB minimum, 24GB recommended",
            "resolution": "512x512",
            "quality": "9.7/10", 
            "features": ["Emotions", "Singing", "Expressions"],
            "source": "https://github.com/HumanAIGC/EMO"
        },
        "Gaussian Splatting": {
            "description": "Ultra-fast 3D-aware lip synchronization",
            "fps": "100+ FPS",
            "vram": "8GB minimum, 12GB recommended",
            "resolution": "512x512 (scalable to 4K)",
            "quality": "9.0/10",
            "features": ["Ultra-fast", "3D-aware", "Real-time"],
            "source": "Based on GaussianTalker research"
        }
    }
    
    print("\n📊 Advanced Lip Sync Models Information:")
    print("=" * 80)
    
    for model_name, info in model_info.items():
        print(f"\n🎭 {model_name}")
        print(f"   Description: {info['description']}")
        print(f"   Performance: {info['fps']}")
        print(f"   VRAM: {info['vram']}")
        print(f"   Resolution: {info['resolution']}")
        print(f"   Quality: {info['quality']}")
        print(f"   Features: {', '.join(info['features'])}")
        print(f"   Source: {info['source']}")


def main():
    """Main download function"""
    
    parser = argparse.ArgumentParser(description="Download advanced lip sync models")
    parser.add_argument("--models", nargs="+", 
                       choices=["musetalk", "latentsync", "vasa1", "emo", "gaussian", "all"],
                       default=["all"],
                       help="Models to download")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check system requirements")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install required dependencies")
    parser.add_argument("--info", action="store_true",
                       help="Show model information")
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("🚀 Advanced Lip Sync Models Download Script")
    print("=" * 60)
    
    if args.info:
        get_model_info()
        return
    
    if args.install_deps:
        install_dependencies()
    
    # Check system requirements
    compatible_models = check_system_requirements()
    
    if args.check_only:
        return
    
    if not compatible_models and not args.install_deps:
        print("\n⚠️  No compatible models found. Consider upgrading GPU or using cloud methods.")
        return
    
    # Download models
    models_to_download = args.models
    if "all" in models_to_download:
        models_to_download = ["musetalk", "latentsync", "vasa1", "emo", "gaussian"]
    
    download_functions = {
        "musetalk": download_musetalk_models,
        "latentsync": download_latentsync_models,
        "vasa1": download_vasa1_models,
        "emo": download_emo_models,
        "gaussian": download_gaussian_splatting_models
    }
    
    successful_downloads = 0
    total_downloads = len(models_to_download)
    
    print(f"\n📥 Downloading {total_downloads} model(s)...")
    
    for model_name in models_to_download:
        if model_name in download_functions:
            success = download_functions[model_name]()
            if success:
                successful_downloads += 1
        else:
            print(f"❌ Unknown model: {model_name}")
    
    # Summary
    print(f"\n📈 Download Summary:")
    print(f"Successfully downloaded: {successful_downloads}/{total_downloads}")
    
    if successful_downloads == total_downloads:
        print("✅ All models downloaded successfully!")
        print("\nNext steps:")
        print("1. Run: python test_advanced_lip_sync.py")
        print("2. Use in your code: from src.lip_sync.advanced_smart_selector import advanced_smart_selector")
    else:
        print("⚠️  Some downloads failed. Check the logs for details.")
        print("Note: Some models may require manual download or API keys.")
    
    print(f"\nDetailed logs saved to: model_download.log")


if __name__ == "__main__":
    main()