# Lip Sync Models - Quick Reference Guide

## 🚀 Quick Selection Guide

### Choose Your Model:

```
┌─────────────────────────────────────────────────────────────┐
│ Need Real-time Processing?                                  │
│ ↓ YES                           ↓ NO                        │
│                                                             │
│ 🏃 Wav2Lip                      Need Best Quality?          │
│ • 25+ FPS                       ↓ YES        ↓ NO          │
│ • 4GB VRAM                                                  │
│ • Good enough quality           🎨 LatentSync  🎯 MuseTalk │
│                                 • Best quality  • Balanced  │
│                                 • 20GB+ VRAM   • 6GB VRAM  │
│                                 • Slow         • Fast       │
└─────────────────────────────────────────────────────────────┘
```

## 📊 At a Glance

| | MuseTalk | Wav2Lip | LatentSync |
|---|:---:|:---:|:---:|
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **VRAM** | 6GB | 4GB | 20GB+ |
| **Setup** | Easy | Easiest | Complex |

## 🎯 Usage Commands

### MuseTalk (Recommended for most users)
```bash
python personalization_pipeline.py video.mp4 --lip-sync-model musetalk
```

### Wav2Lip (Fast & lightweight)
```bash
python personalization_pipeline.py video.mp4 --lip-sync-model wav2lip
```

### LatentSync (Highest quality)
```bash
python personalization_pipeline.py video.mp4 --lip-sync-model latentsync
```

## 💡 Pro Tips

1. **First Time?** Start with MuseTalk
2. **Low GPU?** Use Wav2Lip
3. **Production?** Use LatentSync
4. **Testing?** Run benchmark: `python benchmark_models.py`

## ⚙️ Model Settings

### MuseTalk
- Face Size: 256×256
- FPS: 30+
- Best for: YouTube, social media

### Wav2Lip  
- Face Size: 96×96
- FPS: 25+
- Best for: Live streaming, demos

### LatentSync
- Face Size: 512×512
- FPS: 20-24
- Best for: Films, commercials

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use Wav2Lip or reduce video resolution |
| Slow processing | Check GPU is being used, try Wav2Lip |
| Poor quality | Switch to MuseTalk or LatentSync |
| No face detected | Ensure face is clearly visible |

---
*For detailed comparison, see [MODEL_COMPARISON_REFERENCE.md](MODEL_COMPARISON_REFERENCE.md)*