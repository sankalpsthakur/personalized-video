# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a video personalization pipeline that automatically replaces spoken content in videos with custom variables using template-based text-to-speech generation and optional lip synchronization. The system specializes in creating personalized travel advisor videos for customers.

## Core Architecture

### Main Components
- **`src/pipeline.py`**: Core `VideoPersonalizationPipeline` class that orchestrates the entire process
- **`src/templates.py`**: Contains transcript templates with variable placeholders (`{customer_name}`, `{destination}`)
- **`src/lip_sync/`**: Multiple lip sync implementations with intelligent selection
- **`main.py`**: Command-line interface for the pipeline

### Processing Flow
1. **Duration Analysis**: Extract original video duration using FFmpeg
2. **Template Processing**: Replace variables in transcript template
3. **TTS Generation**: Multi-tier TTS system (Edge-TTS → ElevenLabs → gTTS fallback)
4. **Duration Matching**: Gentle speed adjustment to match original timing (±20% max)
5. **Video Combination**: Either simple audio replacement or full lip sync

### Lip Sync System
The project includes multiple lip sync approaches:
- **Smart Selector** (`smart_selector.py`): Intelligent method selection based on quality/cost/time constraints
- **Working Lip Sync** (`working_lipsync.py`): Reliable fallback implementation
- **Cloud/Replicate** (`cloud_client.py`, `replicate_client.py`): API-based solutions
- **Local Models** (`easy_wav2lip.py`, `improved_local.py`): Self-hosted implementations

## Development Commands

### Running the Pipeline
```bash
# Basic usage (audio replacement only)
python main.py video.mp4 --customer-name "Sarah Johnson" --destination "Tokyo"

# With lip sync (higher quality, slower)
python main.py video.mp4 --customer-name "Sarah Johnson" --destination "Tokyo" --lip-sync

# Python API usage
from src import VideoPersonalizationPipeline
pipeline = VideoPersonalizationPipeline(output_dir="output")
output_path = pipeline.create_personalized_video(
    video_path="video.mp4",
    variables={"customer_name": "Sarah", "destination": "Tokyo"},
    apply_lip_sync=False
)
```

### Testing
```bash
# Run individual test scripts (multiple available)
python test_working_pipeline.py
python test_sarah_tokyo.py
python test_lip_sync_comparison.py

# No formal test framework - uses individual test scripts for different scenarios
```

### Dependencies
```bash
# Install requirements
pip install -r requirements.txt

# Key dependencies: torch, FFmpeg, librosa, edge-tts, various lip sync models
```

## Important File Paths

- **Template video**: `/Users/sankalpthakur/Projects/Projects - Emtribe/personalise_video/src/video_template.mp4`
- **Transcripts**: Located in `src/templates.py`
- **Test scripts**: Multiple `test_*.py` files in root directory
- **Output**: Generated in `output/` directory with logs in `output/logs/`

## Key Configuration

### Template Customization
Edit `src/templates.py` to modify:
- `TRANSCRIPT_TEMPLATE`: The base transcript with `{variable}` placeholders
- `DEFAULT_VARIABLES`: Default values for variables
- `REQUIRED_VARIABLES`: List of mandatory variables

### Environment Variables
- `ELEVENLABS_API_KEY`: For premium TTS quality (optional)

## Common Development Patterns

- The pipeline uses comprehensive logging with stage timing and statistics
- All processing creates timestamped logs in `output/logs/`
- FFmpeg is heavily used for audio/video processing
- Multiple fallback mechanisms ensure robustness (TTS engines, lip sync methods)
- Processing statistics are saved as JSON for analysis

## Architecture Notes

- The system prioritizes complete TTS regeneration over audio stitching to avoid artifacts
- Lip sync is optional and uses intelligent method selection based on constraints
- Duration matching uses gentle speed adjustment with librosa for quality
- All temporary files are managed in isolated temp directories