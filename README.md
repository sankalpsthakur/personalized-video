# Video Personalization Pipeline

Automatically find and replace spoken words in videos with personalized content using AI-powered speech recognition and audio processing.

## Features

- **Automatic Speech Recognition**: Uses OpenAI Whisper for accurate word-level transcription
- **Variable Detection**: Automatically finds target phrases in the video
- **Audio Replacement**: Replaces specific words while maintaining audio continuity
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

### Basic Usage

#### Command Line

```bash
# Personalize a video
python personalization_pipeline.py video.mp4 \
  --customer-name "John Smith" \
  --destination "Paris" \
  --output-dir output/
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

## Limitations

- Currently replaces audio with silence (TTS integration available but not enabled by default)
- Best results with clear speech and minimal background noise
- Processing time depends on video length and system resources

## License

MIT License - See LICENSE file for details