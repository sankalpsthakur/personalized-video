# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a video personalization pipeline that automatically finds and replaces spoken words in videos with personalized content using AI-powered speech recognition, audio processing, and realistic lip synchronization. The system supports multiple state-of-the-art lip sync models and provides both CLI and REST API interfaces.

Additionally, the repository contains a parallel **Veo3 Production Pipeline** (`src/veo3_pipeline/`) that implements a FLUX Kontext → Veo 3 → ElevenLabs workflow for generating character-consistent animated videos with voice cloning from a single reference image.

## Key Commands

### Setup and Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# For Python 3.13+: pip install -r requirements-py313.txt

# Download model files
python scripts/download_models.py           # Basic models (MuseTalk, Wav2Lip)
python scripts/download_advanced_models.py  # Advanced models (VASA-1, EMO, Gaussian)
```

### Running the Pipeline
```bash
# Basic usage with default model (MuseTalk)
PYTHONPATH=/path/to/project python -m src.core.pipeline video.mp4 \
  --customer-name "John" --destination "Paris" --output-dir output/

# Without lip sync (audio-only replacement)
PYTHONPATH=/path/to/project python -m src.core.pipeline video.mp4 \
  --customer-name "John" --destination "Paris" --output-dir output/ --no-lip-sync

# With specific lip sync model
PYTHONPATH=/path/to/project python -m src.core.pipeline video.mp4 \
  --lip-sync-model wav2lip  # Options: musetalk, wav2lip, latentsync, vasa1, emo, gaussian_splatting

# List available models
PYTHONPATH=/path/to/project python -m src.core.pipeline --list-models
```

### Running Tests
```bash
# Run all tests
PYTHONPATH=/path/to/project pytest tests/

# Run specific test file
PYTHONPATH=/path/to/project pytest tests/test_lip_sync.py

# Run with verbose output
PYTHONPATH=/path/to/project pytest -v tests/

# Run specific test function
PYTHONPATH=/path/to/project pytest tests/test_lip_sync.py::test_model_initialization

# Test real video processing
PYTHONPATH=/path/to/project python tests/test_real_video.py
```

### API Server
```bash
# Start the API server
PYTHONPATH=/path/to/project python -m src.api.server

# Submit job via API
curl -X POST http://localhost:5000/personalize \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4", "replacements": {"customer_name": "John", "destination": "Paris"}}'

# Check job status
curl http://localhost:5000/status/{job_id}

# Download result
curl http://localhost:5000/download/{job_id} -o personalized_video.mp4
```

### Benchmarking
```bash
# Benchmark all models
PYTHONPATH=/path/to/project python scripts/benchmark_models.py
```

### Veo3 Pipeline Usage
```bash
# Run complete character video generation demo
PYTHONPATH=/path/to/project python src/veo3_pipeline/example_usage.py

# Direct pipeline usage
PYTHONPATH=/path/to/project python -c "
from src.veo3_pipeline import Veo3Pipeline
pipeline = Veo3Pipeline()
project_id = pipeline.create_project('TestCharacter')
print(f'Created project: {project_id}')
"
```

## Architecture Overview

### Core Pipeline Flow
1. **Video Input** → Extract audio for processing (16kHz for Whisper, 48kHz for output)
2. **Speech Recognition** → Use Whisper to transcribe with word-level timestamps
3. **Variable Detection** → Find target phrases in transcription using configurable patterns
4. **Audio Replacement** → Generate TTS audio and splice with 50ms buffer + 20ms crossfade
5. **Lip Sync** (optional) → Apply selected model to sync mouth movements
6. **Output** → Final personalized video with replaced audio/visuals

### Module Structure

**src/core/** - Main pipeline components
- `pipeline.py` - Main orchestrator (VideoPersonalizationPipeline class)
  - `process()` - Main entry point (line 468)
  - `create_audio_with_replacements()` - Audio splicing logic (line 287)
  - `concatenate_audio_segments()` - Advanced crossfade blending (line 347)
- `audio_processor.py` - Audio extraction, processing, normalization
- `video_processor.py` - Video manipulation, frame extraction, quality checks
- `word_alignment.py` - Whisper integration for word-level timestamps

**src/lip_sync/** - Lip synchronization models
- `lip_sync.py` - LipSyncProcessor class, manages all models
- `models.py` - Individual model implementations (Wav2LipInference, MuseTalkInference, etc.)
- `advanced_models.py` - VASA-1, EMO, Gaussian Splatting implementations
- `advanced.py` - Integration layer for advanced models

**src/api/** - REST API components
- `server.py` - Flask server with job queue (PersonalizationJob class)
- `client.py` - Example client implementation

**src/utils/** - Utilities
- `quality_control.py` - Video/audio quality validation
- `logging_config.py` - Centralized logging configuration

### Key Integration Points

1. **Model Selection**: Models selected via `--lip-sync-model` parameter, instantiated in `pipeline.py:67-73`. Falls back gracefully if models unavailable.

2. **Audio Replacement**: 
   - Buffer time (50ms) added in `create_audio_with_replacements()` (line 306)
   - Crossfade blending (20ms) in `concatenate_audio_segments()` (line 369)
   - TTS generation in `generate_replacement_audio()` (line 188)

3. **Lip Sync Application**: 
   - Called in `create_final_video()` (line 414)
   - Only processes segments with replacements (efficiency optimization)
   - Supports face detection fallback (MediaPipe → OpenCV)

## Important Implementation Details

### Default Search Patterns
The pipeline searches for these phrases by default (defined in pipeline.py:76):
```python
self.search_patterns = {
    "customer_name": ["Anuji", "Anuj ji", "Anuj"],
    "destination": ["Bali"]
}
```

### Audio Processing Details
- **Buffer Time**: 50ms around replacements to ensure clean cuts
- **Crossfade**: 100ms overlap between segments (increased from 20ms for better continuity)
- **Sample Rate**: 48kHz mono for final output
- **Normalization**: Prevents clipping, targets 0.95 max amplitude
- **Spectral Matching**: Voice conversion applied to match original speaker characteristics
- **Prosody Matching**: Context-aware TTS generation analyzes surrounding audio

### Model Requirements
- **MuseTalk**: 6GB VRAM, requires Whisper model
- **Wav2Lip**: 4GB VRAM, fastest option
- **LatentSync**: 20GB+ VRAM, highest quality
- **VASA-1**: 16GB VRAM, real-time capable
- **EMO**: 12GB VRAM, emotion-aware
- **Gaussian Splatting**: 24GB VRAM, 3D rendering

### Recent Fixes Applied

1. **Audio Overlay Problem**: Fixed with buffer time + crossfade blending
2. **Lip Sync Integration**: Fixed method name mismatch and added error handling
3. **Python 3.13 Compatibility**: Added fallbacks for incompatible packages
4. **Transcription Accuracy**: Added "Anurji" (single word) to search patterns for Whisper
5. **Voice Conversion Implementation** (NEW):
   - Extract voice features (pitch, spectral envelope, MFCC) from speaker
   - Apply real-time voice conversion to TTS output
   - Spectral envelope matching for timbre transformation
   - Pitch shifting to match original speaker
   - Gaussian smoothing for natural transitions
6. **Improved Continuity**:
   - Increased crossfade from 50ms to 100ms
   - Added logarithmic fade curves for natural perception
   - Implemented spectral smoothing at transitions
   - Added prosody-aware TTS generation
   - Applied voice conversion to all TTS segments

### Audio Quality Metrics

**Previous Performance** (gTTS only):
- Overall Similarity: 86.1%
- Overall Continuity: 53.3%
- Spectral Continuity: 12.8%

**Current Performance** (with enhancements):
- Overall Similarity: 89.5%
- Overall Continuity: 51.4%
- Spectral Continuity: 16.7%

**Expected Performance** (with voice conversion):
- Overall Similarity: 90%+
- Overall Continuity: 75%+
- Spectral Continuity: 50-60%+
- Energy Continuity: 75%+

**Target**: 90%+ for all metrics

**Voice Conversion Features**:
1. **extract_voice_features()**: Analyzes speaker characteristics from input video
2. **apply_voice_conversion()**: Real-time transformation of TTS to match speaker
3. **wsola_stretch()**: Time-stretching for precise duration matching
4. Works with Python 3.13 using librosa/scipy

**Achieving 90%+ Continuity**: 
For full 90%+ spectral continuity, neural voice cloning is required:
1. **Coqui TTS with XTTS-v2** (requires Python < 3.13)
2. **OpenVoice** (when Python 3.13 support is added)
3. **Cloud APIs** (ElevenLabs, Azure Speech)
4. **Docker solution** with Python 3.11 for Coqui TTS

## Environment Dependencies

- **FFmpeg**: Required for all audio/video processing. Must be installed separately.
- **CUDA**: Required for GPU-accelerated lip sync models
- **Whisper Models**: Downloaded automatically on first use

## Configuration Points

- **TTS Provider**: gTTS by default, configurable in `generate_replacement_audio()`
- **Audio Format**: 48kHz mono throughout pipeline
- **Temp Files**: Auto-cleanup in `__del__` method
- **Job Queue**: In-memory queue for API server, could be replaced with Redis/RabbitMQ

## Veo3 Pipeline Architecture

### Workflow Overview
1. **FLUX Kontext** (`api_clients.py:FluxKontextClient`) - Generate/edit master stills with layered prompting
2. **Character Consistency** (`character_consistency.py:CharacterConsistencyManager`) - Profile management, turntable generation, LoRA prep
3. **Veo 3 Animation** (`api_clients.py:Veo3Client`) - Async video generation with 6-part structured prompts
4. **ElevenLabs Voice** (`api_clients.py:ElevenLabsClient`) - Professional voice cloning and TTS
5. **Post-Production** (`post_production.py:PostProductionPipeline`) - Sync, color grade, master to broadcast standards

### Key Integration Points

**Project Structure** (created by `Veo3Pipeline.create_project()`):
```
veo3_projects/{project_id}/
├── kontext/     # Master stills, edit logs
├── veo3/        # Animated videos, prompts  
├── audio/       # Voice files, transcripts
├── exports/     # Final deliverables (4K, HD, Mobile, Archive)
├── metadata/    # Complete project tracking
└── character_profiles/  # Visual features, identity markers
```

**API Configuration**: Set environment variables:
- `FLUX_KONTEXT_API_KEY`
- `VEO3_API_KEY`
- `ELEVENLABS_API_KEY`

**Export Presets** (in `post_production.py:PostProductionPipeline.EXPORT_PRESETS`):
- Web 4K: H.265, 50Mbps
- Web HD: H.264, 10Mbps  
- Master: ProRes 4444
- Mobile: H.264, 5Mbps

**Quality Metrics**: VMAF, STOI/PESQ, MOS estimation tracked automatically