# Veo3 Production Pipeline

Production-ready video generation pipeline using FLUX Kontext → Veo 3 → ElevenLabs workflow for creating character-consistent animated videos with voice cloning.

## Overview

This pipeline implements a professional workflow for generating high-quality character videos:

1. **FLUX Kontext** - Generate or edit master character stills
2. **Veo 3** - Animate stills with cinematic camera movement
3. **ElevenLabs** - Clone voices and generate synchronized narration
4. **Post-Production** - Professional sync, color grading, and mastering

## Key Features

### Character Consistency
- Character profile management with visual feature tracking
- Turntable generation for multiple angle views
- LoRA dataset preparation for future fine-tuning
- Identity validation across generated content

### Production Quality
- 4K export with multiple format presets
- Broadcast-standard audio mastering (EBU R128)
- Professional color grading with skin tone protection
- Frame-accurate audio-video synchronization

### Scalability
- Async API integration for parallel processing
- Batch processing capabilities
- Comprehensive metadata tracking
- Version control for all assets

## Installation

```bash
# Install dependencies
pip install -r requirements_veo3.txt

# Set API keys
export FLUX_KONTEXT_API_KEY="your_key"
export VEO3_API_KEY="your_key"
export ELEVENLABS_API_KEY="your_key"
```

## Quick Start

```python
from veo3_pipeline import Veo3Pipeline, KontextConfig

# Initialize pipeline
pipeline = Veo3Pipeline()

# Create project
project_id = pipeline.create_project("MyCharacter")

# Generate master still
kontext_config = KontextConfig(
    base_prompt="A professional woman in business attire",
    style_hints="photorealistic, studio lighting"
)
still_result = pipeline.generate_master_still(project_id, kontext_config)

# Animate with Veo 3
animation_result = pipeline.animate_with_veo3(
    project_id,
    still_result["image_path"],
    prompt_structure={
        "subject": "professional woman",
        "action": "confidently presents to camera",
        "camera_motion": "subtle zoom in"
    }
)

# Add voice
voice_result = pipeline.clone_voice_elevenlabs(
    project_id,
    script_text="Welcome to our presentation..."
)

# Post-production
post_result = pipeline.post_production(project_id)
```

## Architecture

### Core Modules

1. **veo3_pipeline.py** - Main pipeline orchestration
2. **api_clients.py** - API client implementations with retry logic
3. **character_consistency.py** - Character profile and consistency management
4. **post_production.py** - Professional video finishing tools
5. **example_usage.py** - Complete usage examples

### Project Structure

```
veo3_projects/
├── {project_id}/
│   ├── kontext/         # Master stills and edit logs
│   ├── veo3/           # Animated videos and prompts
│   ├── audio/          # Voice files and transcripts
│   ├── exports/        # Final deliverables
│   ├── metadata/       # Project metadata
│   └── character_profiles/  # Character consistency data
```

## API Integration

### FLUX Kontext
- Supports Dev/Pro/Max model versions
- Layered prompting with negative prompts
- Reference image editing capabilities
- Automatic edit log versioning

### Veo 3
- 6-part structured prompting schema
- Multi-angle generation strategy
- Async processing for efficiency
- Conditioning frame support

### ElevenLabs
- Professional voice cloning (1-3 min samples)
- Audio-to-audio conversion
- WebSocket streaming support
- Voice settings customization

## Post-Production Features

### Audio Mastering
- EBU R128 loudness normalization
- Target LUFS: -14 dB (streaming standard)
- True peak limiting: -1 dB
- Dynamic range control

### Color Grading
- 3D LUT support
- Parametric adjustments
- Skin tone protection
- Minimal frame variation

### Export Presets
- **Web 4K**: H.265, 50Mbps, 3840x2160
- **Web HD**: H.264, 10Mbps, 1920x1080
- **Master**: ProRes 4444, lossless
- **Mobile**: H.264, 5Mbps, 1280x720

## Quality Metrics

The pipeline tracks objective quality metrics:
- Frame stability (VMAF)
- Audio intelligibility (STOI/PESQ)
- Overall MOS estimation
- Sync accuracy validation

## Best Practices

### Data Management
1. Version control all prompts and settings
2. Archive lossless masters
3. Track character profiles across projects
4. Maintain edit logs for reproducibility

### Performance Optimization
1. Use async APIs for parallel processing
2. Cache character embeddings
3. Batch similar operations
4. Monitor API rate limits

### Legal Compliance
1. Verify voice cloning rights
2. Disclose synthetic content
3. Respect platform ToS
4. Maintain consent records

## Roadmap

### Near-term (Available Now)
- Basic character consistency via careful prompting
- Multi-angle composite strategy
- Professional voice cloning

### Mid-term (Coming Months)
- Veo3 fine-tuning hooks
- LoRA integration for character lock-in
- Real-time preview streaming

### Long-term
- Full 360° character turntables
- Dynamic scene adaptation
- Multi-character interaction

## Troubleshooting

### Common Issues

1. **Character drift between angles**
   - Use more specific identity markers in prompts
   - Generate additional reference angles
   - Wait for Veo3 fine-tuning support

2. **Audio sync issues**
   - Enable drift correction
   - Use manual sync markers for critical points
   - Check source frame rates match

3. **Quality degradation**
   - Use lossless intermediate formats
   - Avoid aggressive color grading
   - Export at native resolution first

## Contributing

This pipeline is designed for extensibility. Key extension points:
- Custom API clients in `api_clients.py`
- New export presets in `post_production.py`
- Character profile enhancements
- Quality metric implementations

## License

This pipeline is provided as-is for demonstration purposes. Ensure you have appropriate licenses for all APIs and content you generate.