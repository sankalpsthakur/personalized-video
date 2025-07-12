# Video Personalization Pipeline

Automatically personalize videos by replacing spoken content with custom variables using template-based text-to-speech generation and optional lip synchronization.

## 🚀 Features

- **Template-based personalization**: Replace `{customer_name}` and `{destination}` variables in transcript
- **Professional TTS**: Multi-tier system (Edge-TTS, ElevenLabs, gTTS) with automatic fallback  
- **Natural speech flow**: Complete transcript regeneration ensures consistent voice
- **Optional lip sync**: High-quality lip synchronization using state-of-the-art models
- **Duration matching**: Smart speed adjustment to match original video timing

## 📋 Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd personalise_video
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Personalize video (audio replacement only - fast)
python main.py input_video.mp4 \
  --customer-name "Sarah Johnson" \
  --destination "Tokyo"

# With lip sync (slower but higher quality)
python main.py input_video.mp4 \
  --customer-name "Sarah Johnson" \
  --destination "Tokyo" \
  --lip-sync
```

### Python API

```python
from src import VideoPersonalizationPipeline

# Initialize pipeline
pipeline = VideoPersonalizationPipeline(output_dir="output")

# Personalize video
output_path = pipeline.create_personalized_video(
    video_path="input_video.mp4",
    variables={
        "customer_name": "Sarah Johnson", 
        "destination": "Tokyo"
    },
    apply_lip_sync=False  # Set to True for lip sync
)

print(f"Personalized video: {output_path}")
```

## 🏗️ How It Works

### Template System

The pipeline uses an exact transcript template with variable placeholders:

```
"Hi, {customer_name}. I'm Kshitij, your dedicated travel advisor from Thirty Sundays. 
I'll be helping you plan your {destination} trip. I just wanted to put a face to the 
name so that you know who you're speaking with..."
```

### TTS Generation

1. **Edge-TTS** (Microsoft): Professional quality, free
2. **ElevenLabs**: Premium voices (requires API key)  
3. **gTTS** (Google): Basic quality fallback

### Processing Steps

1. Extract original video duration for timing reference
2. Replace template variables with provided values
3. Generate complete TTS audio using best available engine
4. Apply gentle speed adjustment to match original timing (±20% max)
5. Optionally apply lip sync to entire video
6. Output final personalized video

## ⚙️ Configuration

### Environment Variables (Optional)

```bash
# For premium TTS quality
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

### Custom Templates

Edit `src/templates.py` to customize the transcript:

```python
TRANSCRIPT_TEMPLATE = "Hello {customer_name}, welcome to {company}..."

DEFAULT_VARIABLES = {
    "customer_name": "Anuj Ji",
    "company": "Thirty Sundays"
}
```

## 🎯 Quality Features

- **No audio artifacts**: Complete TTS regeneration eliminates stitching issues
- **Consistent voice**: Same TTS engine throughout entire video
- **Duration preservation**: Smart timing adjustment maintains lip sync compatibility
- **Professional quality**: Multi-tier TTS system ensures best possible voice quality

## 🛠️ Requirements

- Python 3.8+ (Python 3.13+ supported)
- FFmpeg (must be installed separately)
- 2GB+ RAM for processing
- GPU recommended for lip sync (4-6GB VRAM)

## 📁 Project Structure

```
├── src/
│   ├── pipeline.py          # Main VideoPersonalizationPipeline class
│   ├── templates.py         # Transcript templates and variables
│   ├── lip_sync/           # Lip synchronization models
│   └── utils/              # Logging and utilities
├── main.py                 # Command-line interface
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 📄 License

MIT License - See LICENSE file for details