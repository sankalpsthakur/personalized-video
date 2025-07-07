# Video Personalization Pipeline

Automatically find and replace spoken words in videos with personalized content using AI-powered speech recognition, audio processing, and realistic lip synchronization.

## Features

- **Automatic Speech Recognition**: Uses OpenAI Whisper for accurate word-level transcription
- **Variable Detection**: Automatically finds target phrases in the video
- **Audio Replacement**: Replaces specific words while maintaining audio continuity
- **Lip Synchronization**: Uses MuseTalk for realistic mouth movements matching replaced audio
- **Text-to-Speech**: Integrated Google TTS for generating replacement audio
- **REST API**: Easy integration with any system via HTTP API
- **Batch Processing**: Process multiple personalizations efficiently

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sankalpsthakur/personalized-video.git
cd personalized-video
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download lip sync models:
```bash
python download_models.py
```

### Basic Usage

#### Command Line

```bash
# Personalize a video with lip sync
python personalization_pipeline.py video.mp4 \
  --customer-name "John Smith" \
  --destination "Paris" \
  --output-dir output/

# Disable lip sync (audio-only replacement)
python personalization_pipeline.py video.mp4 \
  --customer-name "John Smith" \
  --destination "Paris" \
  --output-dir output/ \
  --no-lip-sync
```

#### API Server

1. Start the server:
```bash
python api_server.py
```

2. Submit a personalization job:
```bash
curl -X POST http://localhost:5000/personalize \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "replacements": {
      "customer_name": "Alice Johnson",
      "destination": "Tokyo"
    }
  }'
```

## Configuration

### Default Variables

The pipeline searches for these default phrases:
- Customer name: "Anuji", "Anuj ji", "Anuj"
- Destination: "Bali"

To customize, edit the `search_patterns` in `personalization_pipeline.py`:

```python
self.search_patterns = {
    "customer_name": ["Your Name", "Another Variant"],
    "destination": ["Your Location"],
    "custom_var": ["Custom Phrase"]
}
```

### Adding Text-to-Speech

By default, the pipeline replaces audio with silence. To use actual TTS:

1. Install a TTS provider:
```bash
pip install gtts  # For Google TTS
# or
pip install elevenlabs  # For ElevenLabs
```

2. Modify `generate_replacement_audio` in `personalization_pipeline.py` to use TTS instead of silence.

## Project Structure

```
├── personalization_pipeline.py  # Main pipeline orchestrator
├── api_server.py               # REST API server
├── client_example.py           # Example API client
├── word_alignment.py           # Whisper-based word alignment
├── audio_processor.py          # Audio processing utilities
├── video_processor.py          # Video manipulation tools
├── quality_control.py          # QC and validation
└── requirements.txt            # Python dependencies
```

## API Reference

### POST /personalize

Submit a video for personalization.

**Request:**
```json
{
  "video_path": "/path/to/video.mp4",
  "replacements": {
    "customer_name": "John Doe",
    "destination": "London"
  }
}
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Video personalization job created"
}
```

### GET /status/{job_id}

Check job status.

### GET /download/{job_id}

Download the personalized video.

## Requirements

- Python 3.8+
- FFmpeg (must be installed separately)
- 2GB+ RAM for Whisper model

## Lip Sync Technology

The system supports multiple state-of-the-art lip synchronization models:

### Available Models

| Model | Quality | FPS | Face Size | VRAM Required | Use Case |
|-------|---------|-----|-----------|---------------|----------|
| **MuseTalk** | High | 30+ | 256x256 | 6GB | Best balance of quality and performance |
| **Wav2Lip** | Medium | 25 | 96x96 | 4GB | Fastest, good for real-time applications |
| **LatentSync** | Highest | 24 | 512x512 | 20GB+ | Best quality, requires high-end GPU |

### Model Selection

Choose a model based on your needs:

```bash
# List available models
python personalization_pipeline.py --list-models

# Use specific model
python personalization_pipeline.py video.mp4 --lip-sync-model musetalk  # Default
python personalization_pipeline.py video.mp4 --lip-sync-model wav2lip   # Faster
python personalization_pipeline.py video.mp4 --lip-sync-model latentsync # Best quality
```

### Features

- **Multi-Model Support**: Switch between models via CLI
- **Face Detection**: Multiple fallback options (FaceNet, MediaPipe, OpenCV)
- **Selective Processing**: Only processes video segments with replacements for efficiency
- **Automatic Model Selection**: Falls back to simpler models if GPU resources are limited

### Requirements by Model

**MuseTalk:**
- 6GB+ VRAM (GPU)
- Models: musetalk.json, pytorch_model.bin, whisper model
- Best for: General use, good quality/speed balance

**Wav2Lip:**
- 4GB+ VRAM (GPU)
- Models: wav2lip_gan.pth
- Best for: Real-time processing, lower-end GPUs

**LatentSync:**
- 20GB+ VRAM (GPU)
- Models: stable_syncnet.pt, Stable Diffusion models
- Best for: Highest quality output, film production

## Testing

Test the lip sync functionality:

```bash
# Run comprehensive test suite
python test_lip_sync.py

# Test individual components
python lip_sync.py  # Shows model comparison
python download_models.py  # Check model installation
```

## Limitations

- Best results with clear speech and minimal background noise
- Processing time depends on video length and system resources
- Lip sync requires visible face in frame
- GPU recommended for real-time lip sync performance
- Model quality depends on available VRAM and model files

## License

MIT License - See LICENSE file for details