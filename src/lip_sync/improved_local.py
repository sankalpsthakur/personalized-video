"""
Improved local lip sync implementation
Uses better audio analysis and mouth shape mapping for more realistic sync
"""

import cv2
import numpy as np
import librosa
import logging
from pathlib import Path
import subprocess
import os

logger = logging.getLogger(__name__)

class ImprovedLocalLipSync:
    """Improved local lip sync with better audio analysis"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def extract_audio_features(self, audio_path: str, video_fps: float, frame_count: int):
        """Extract detailed audio features for better lip sync"""
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Calculate hop length for frame-by-frame analysis
        hop_length = sr // int(video_fps)
        
        features = {
            'energy': [],
            'mfcc': [],
            'spectral_centroid': [],
            'zero_crossing_rate': []
        }
        
        for i in range(frame_count):
            start_sample = i * hop_length
            end_sample = min(start_sample + hop_length, len(audio))
            
            if start_sample < len(audio):
                frame_audio = audio[start_sample:end_sample]
                
                # Energy (RMS)
                energy = np.sqrt(np.mean(frame_audio**2))
                features['energy'].append(energy)
                
                # MFCC (first coefficient indicates formant structure)
                if len(frame_audio) > 0:
                    mfccs = librosa.feature.mfcc(y=frame_audio, sr=sr, n_mfcc=13)
                    features['mfcc'].append(mfccs.mean(axis=1))
                    
                    # Spectral centroid (brightness)
                    centroid = librosa.feature.spectral_centroid(y=frame_audio, sr=sr)
                    features['spectral_centroid'].append(centroid.mean())
                    
                    # Zero crossing rate (consonant detection)
                    zcr = librosa.feature.zero_crossing_rate(frame_audio)
                    features['zero_crossing_rate'].append(zcr.mean())
                else:
                    features['mfcc'].append(np.zeros(13))
                    features['spectral_centroid'].append(0.0)
                    features['zero_crossing_rate'].append(0.0)
            else:
                features['energy'].append(0.0)
                features['mfcc'].append(np.zeros(13))
                features['spectral_centroid'].append(0.0)
                features['zero_crossing_rate'].append(0.0)
        
        # Normalize features
        max_energy = max(features['energy']) if features['energy'] else 1.0
        features['energy'] = [e / max_energy for e in features['energy']]
        
        max_centroid = max(features['spectral_centroid']) if features['spectral_centroid'] else 1.0
        features['spectral_centroid'] = [c / max_centroid for c in features['spectral_centroid']]
        
        return features
    
    def determine_mouth_shape(self, energy: float, centroid: float, zcr: float, mfcc: np.ndarray):
        """Determine mouth shape based on audio features"""
        
        # Basic mouth states based on audio characteristics
        if energy < 0.1:
            return "closed"  # Silence
        elif zcr > 0.15:
            return "open_wide"  # Consonants (s, t, k sounds)
        elif centroid > 0.6:
            return "open_narrow"  # High frequency sounds (i, e)
        elif energy > 0.6:
            return "open_round"  # Loud sounds (a, o)
        else:
            return "open_medium"  # Default speech
    
    def apply_mouth_shape(self, frame: np.ndarray, mouth_region: tuple, shape: str, intensity: float):
        """Apply specific mouth shape to the frame"""
        
        x, y, w, h = mouth_region
        
        # Define mouth area (bottom third of face)
        mouth_y = y + int(h * 0.7)
        mouth_h = int(h * 0.3)
        mouth_x = x + int(w * 0.25)
        mouth_w = int(w * 0.5)
        
        # Ensure mouth region is within frame bounds
        frame_h, frame_w = frame.shape[:2]
        mouth_y = max(0, min(mouth_y, frame_h - mouth_h))
        mouth_x = max(0, min(mouth_x, frame_w - mouth_w))
        mouth_h = min(mouth_h, frame_h - mouth_y)
        mouth_w = min(mouth_w, frame_w - mouth_x)
        
        if mouth_w <= 0 or mouth_h <= 0:
            return frame
        
        # Extract mouth region
        mouth_roi = frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w].copy()
        
        if mouth_roi.size == 0:
            return frame
        
        # Apply shape-specific transformations
        if shape == "closed":
            # Slightly compress vertically
            new_h = max(1, int(mouth_h * 0.8))
            mouth_roi = cv2.resize(mouth_roi, (mouth_w, new_h))
            
            # Center the compressed mouth
            y_offset = (mouth_h - new_h) // 2
            frame[mouth_y + y_offset:mouth_y + y_offset + new_h, mouth_x:mouth_x+mouth_w] = mouth_roi
            
        elif shape == "open_wide":
            # Expand horizontally and vertically
            expand_factor = 1.0 + (intensity * 0.3)
            new_w = min(frame_w - mouth_x, int(mouth_w * expand_factor))
            new_h = min(frame_h - mouth_y, int(mouth_h * expand_factor))
            
            if new_w > 0 and new_h > 0:
                mouth_roi = cv2.resize(mouth_roi, (new_w, new_h))
                frame[mouth_y:mouth_y+new_h, mouth_x:mouth_x+new_w] = mouth_roi
            
        elif shape == "open_round":
            # Create circular opening effect
            new_h = int(mouth_h * (1.0 + intensity * 0.4))
            new_h = min(new_h, frame_h - mouth_y)
            
            if new_h > 0:
                mouth_roi = cv2.resize(mouth_roi, (mouth_w, new_h))
                
                # Darken center to simulate opening
                center_y, center_x = new_h // 2, mouth_w // 2
                radius = min(center_x, center_y) // 2
                
                mask = np.zeros((new_h, mouth_w), dtype=np.uint8)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
                # Apply darkening where mask is white
                for c in range(3):  # BGR channels
                    mouth_roi[:, :, c] = np.where(mask == 255, 
                                                  mouth_roi[:, :, c] * 0.6, 
                                                  mouth_roi[:, :, c])
                
                frame[mouth_y:mouth_y+new_h, mouth_x:mouth_x+mouth_w] = mouth_roi
                
        elif shape in ["open_narrow", "open_medium"]:
            # Vertical expansion with slight horizontal change
            v_factor = 1.0 + (intensity * 0.2)
            h_factor = 1.0 + (intensity * 0.1) if shape == "open_medium" else 1.0
            
            new_h = min(frame_h - mouth_y, int(mouth_h * v_factor))
            new_w = min(frame_w - mouth_x, int(mouth_w * h_factor))
            
            if new_w > 0 and new_h > 0:
                mouth_roi = cv2.resize(mouth_roi, (new_w, new_h))
                frame[mouth_y:mouth_y+new_h, mouth_x:mouth_x+new_w] = mouth_roi
        
        return frame
    
    def process_video(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Process video with improved lip sync"""
        
        try:
            logger.info("Using improved local lip sync...")
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video: {frame_count} frames at {fps} FPS, {width}x{height}")
            
            # Extract detailed audio features
            audio_features = self.extract_audio_features(audio_path, fps, frame_count)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_output = output_path + ".temp.mp4"
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            frame_idx = 0
            last_face = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get audio features for this frame
                if frame_idx < len(audio_features['energy']):
                    energy = audio_features['energy'][frame_idx]
                    centroid = audio_features['spectral_centroid'][frame_idx]
                    zcr = audio_features['zero_crossing_rate'][frame_idx]
                    mfcc = audio_features['mfcc'][frame_idx]
                else:
                    energy = centroid = zcr = 0.0
                    mfcc = np.zeros(13)
                
                # Detect face (every 10 frames for performance)
                if frame_idx % 10 == 0 or last_face is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        last_face = max(faces, key=lambda x: x[2] * x[3])
                
                # Apply lip sync if face detected
                if last_face is not None and energy > 0.05:  # Only sync for speech
                    mouth_shape = self.determine_mouth_shape(energy, centroid, zcr, mfcc)
                    frame = self.apply_mouth_shape(frame, last_face, mouth_shape, energy)
                
                out.write(frame)
                frame_idx += 1
                
                if frame_idx % 30 == 0:
                    logger.info(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
            
            cap.release()
            out.release()
            
            # Add audio back using FFmpeg
            logger.info("Adding audio to lip-synced video...")
            cmd = [
                "ffmpeg", "-i", temp_output, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", 
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                os.remove(temp_output)
                logger.info("✅ Improved local lip sync completed with audio")
                return True
            else:
                logger.error(f"Failed to add audio: {result.stderr}")
                os.rename(temp_output, output_path)
                logger.warning("✅ Improved local lip sync completed (video only)")
                return True
                
        except Exception as e:
            logger.error(f"Improved local lip sync failed: {e}")
            return False