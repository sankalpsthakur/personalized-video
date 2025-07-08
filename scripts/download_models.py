#!/usr/bin/env python3
"""
Download required models for lip sync functionality
"""

import os
import sys
import requests
from pathlib import Path
import zipfile
import tarfile
import shutil
from tqdm import tqdm


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))


def setup_musetalk_models():
    """Download and setup MuseTalk models"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("MuseTalk Model Setup")
    print("=" * 60)
    print()
    
    # Import helper functions
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.lip_sync.models import create_musetalk_placeholder
        create_musetalk_placeholder(models_dir)
        print("✓ Created MuseTalk placeholder structure")
    except:
        pass
    
    print("MuseTalk requires several pre-trained models to function.")
    print()
    print("For actual MuseTalk models:")
    print("1. Visit: https://github.com/TMElyralab/MuseTalk")
    print("2. Download model checkpoints")
    print("3. Place in models/musetalk/")
    print()
    
    # Create subdirectories
    subdirs = ["musetalk", "whisper", "syncnet", "face_detection"]
    for subdir in subdirs:
        (models_dir / subdir).mkdir(exist_ok=True)
    
    # Download Whisper tiny model (this is publicly available)
    whisper_url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
    whisper_path = models_dir / "whisper" / "tiny.pt"
    
    if not whisper_path.exists():
        print("\nDownloading Whisper tiny model (39MB)...")
        try:
            download_file(whisper_url, whisper_path, "Whisper tiny")
            print("✓ Whisper model downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download Whisper model: {e}")
    else:
        print("✓ Whisper model already exists")
    
    print()
    print("Setting up Wav2Lip...")
    print("-" * 30)
    try:
        from src.lip_sync.models import download_wav2lip_model
        download_wav2lip_model(models_dir)
        print("✓ Wav2Lip setup complete")
    except Exception as e:
        print(f"Wav2Lip setup failed: {e}")
        print("Manual setup:")
        print("1. Clone: git clone https://github.com/Rudrabha/Wav2Lip.git")
        print("2. Download wav2lip_gan.pth")
        print("3. Place in models/wav2lip/")
    
    print()
    print("Setting up LatentSync...")
    print("-" * 30)
    try:
        from src.lip_sync.models import setup_latentsync_env
        setup_latentsync_env(models_dir)
        print("✓ LatentSync placeholders created")
    except Exception as e:
        print(f"LatentSync setup failed: {e}")
    
    print()
    print("For LatentSync (requires 20GB+ VRAM):")
    print("1. Install: pip install diffusers transformers accelerate")
    print("2. Visit: https://github.com/bytedance/LatentSync")
    print("3. Download stable_syncnet.pt")
    print("4. Place in models/latentsync/")
    print()
    
    # Create README in models directory
    readme_content = """# Model Directory

This directory contains pre-trained models for lip sync functionality.

## Required Models

### MuseTalk Models
- musetalk/musetalk.json
- musetalk/pytorch_model.bin
- whisper/tiny.pt (or base.pt)
- syncnet/syncnet.pth (optional)

### Alternative: Wav2Lip Models
- wav2lip/wav2lip_gan.pth

## Download Instructions

Please run `python download_models.py` for detailed instructions on obtaining these models.
"""
    
    with open(models_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print()
    print("✓ Model directory structure created")
    print("✓ README.md created in models directory")
    print()
    print("Next steps:")
    print("1. Download the required models as instructed above")
    print("2. Place them in the appropriate directories")
    print("3. Run the personalization pipeline with lip sync enabled")
    print()


def check_models():
    """Check if required models are present"""
    models_dir = Path("models")
    
    required_files = {
        "MuseTalk": [
            "musetalk/musetalk.json",
            "musetalk/pytorch_model.bin",
            "whisper/tiny.pt"
        ],
        "Wav2Lip": [
            "wav2lip/wav2lip_gan.pth"
        ]
    }
    
    print("\nChecking for installed models...")
    print("-" * 40)
    
    for model_name, files in required_files.items():
        print(f"\n{model_name}:")
        all_present = True
        for file in files:
            path = models_dir / file
            if path.exists():
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (missing)")
                all_present = False
        
        if all_present:
            print(f"  → {model_name} is ready to use")
    
    print()


if __name__ == "__main__":
    print("Video Personalization - Model Setup")
    print("=" * 60)
    
    # Check if tqdm is available
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        os.system(f"{sys.executable} -m pip install tqdm")
        from tqdm import tqdm
    
    setup_musetalk_models()
    check_models()
    
    print("Setup complete!")
    print()
    print("To test lip sync, run:")
    print("  python personalization_pipeline.py <video_file> --customer-name 'John' --destination 'Paris'")
    print()
    print("To disable lip sync (audio-only replacement), add --no-lip-sync")