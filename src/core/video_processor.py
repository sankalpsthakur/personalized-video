#!/usr/bin/env python3
"""
Video processing module for frame-accurate editing
Handles video cutting, stitching, and synchronization
"""

import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import tempfile
import shutil


class VideoProcessor:
    def __init__(self, working_dir: Optional[Path] = None):
        """
        Initialize video processor
        
        Args:
            working_dir: Working directory for temporary files
        """
        self.working_dir = working_dir or Path(tempfile.mkdtemp())
        self.working_dir.mkdir(exist_ok=True)
        
    def get_video_info(self, video_path: Path) -> Dict:
        """
        Get detailed video information using ffprobe
        
        Returns:
            Dictionary with video metadata
        """
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        # Extract key information
        video_stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
        
        return {
            'duration': float(info['format']['duration']),
            'video': {
                'codec': video_stream['codec_name'],
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),  # Convert "30/1" to 30
                'pixel_format': video_stream['pix_fmt']
            },
            'audio': {
                'codec': audio_stream['codec_name'] if audio_stream else None,
                'sample_rate': int(audio_stream['sample_rate']) if audio_stream else None,
                'channels': int(audio_stream['channels']) if audio_stream else None
            }
        }
    
    def extract_segment(self,
                       video_path: Path,
                       start_time: float,
                       end_time: float,
                       output_path: Path,
                       include_audio: bool = True) -> Path:
        """
        Extract a segment from video with frame accuracy
        
        Args:
            video_path: Input video path
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output path
            include_audio: Whether to include audio
            
        Returns:
            Path to extracted segment
        """
        # Handle infinite end time
        if end_time == float('inf'):
            # Extract just audio if output is .wav
            if str(output_path).endswith('.wav'):
                cmd = [
                    "ffmpeg", "-i", str(video_path),
                    "-vn", "-c:a", "pcm_s16le",
                    "-ar", "48000", "-ac", "2",
                    "-y", str(output_path)
                ]
            else:
                cmd = [
                    "ffmpeg", "-i", str(video_path),
                    "-c:v", "copy", "-c:a", "copy",
                    "-y", str(output_path)
                ]
        else:
            duration = end_time - start_time
            
            # Use -ss before -i for fast seeking
            cmd = [
                "ffmpeg", "-ss", str(start_time),
                "-i", str(video_path),
                "-t", str(duration),
            ]
            
            # Handle audio extraction
            if str(output_path).endswith('.wav'):
                cmd.extend(["-vn", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"])
            else:
                cmd.extend(["-c:v", "copy"])
                if include_audio:
                    cmd.extend(["-c:a", "copy"])
                else:
                    cmd.extend(["-an"])
            
            cmd.extend([
                "-avoid_negative_ts", "make_zero",
                "-y", str(output_path)
            ])
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def create_video_from_frames(self,
                               frame_pattern: str,
                               fps: float,
                               output_path: Path,
                               video_codec: str = "libx264",
                               preset: str = "medium",
                               crf: int = 18) -> Path:
        """
        Create video from image sequence
        
        Args:
            frame_pattern: Pattern for frame files (e.g., "frame_%04d.png")
            fps: Frame rate
            output_path: Output video path
            video_codec: Video codec to use
            preset: Encoding preset (ultrafast, fast, medium, slow)
            crf: Constant Rate Factor (0-51, lower is better quality)
            
        Returns:
            Path to created video
        """
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", video_codec,
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",  # Ensure compatibility
            "-y", str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def concatenate_videos(self,
                          video_paths: List[Path],
                          output_path: Path,
                          reencode: bool = False) -> Path:
        """
        Concatenate multiple videos with matching parameters
        
        Args:
            video_paths: List of video paths to concatenate
            output_path: Output video path
            reencode: Whether to re-encode (required if videos have different parameters)
            
        Returns:
            Path to concatenated video
        """
        # Create concat demuxer file
        concat_file = self.working_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for video_path in video_paths:
                f.write(f"file '{video_path.absolute()}'\n")
        
        if reencode:
            # Re-encode for compatibility
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-y", str(output_path)
            ]
        else:
            # Copy streams (faster, lossless)
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                "-y", str(output_path)
            ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        concat_file.unlink()
        
        return output_path
    
    def replace_audio_precise(self,
                            video_path: Path,
                            audio_replacements: List[Tuple[float, float, Path]],
                            output_path: Path) -> Path:
        """
        Replace audio at specific timestamps with frame-accurate sync
        
        Args:
            video_path: Input video path
            audio_replacements: List of (start_time, end_time, replacement_audio_path)
            output_path: Output video path
            
        Returns:
            Path to processed video
        """
        # Sort replacements by time
        audio_replacements.sort(key=lambda x: x[0])
        
        # Extract original audio
        original_audio = self.working_dir / "original_audio.wav"
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-c:a", "pcm_s16le",
            "-ar", "48000", "-ac", "2",
            "-y", str(original_audio)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Create audio segments
        segments = []
        last_end = 0.0
        
        for i, (start_time, end_time, replacement_path) in enumerate(audio_replacements):
            # Segment before replacement
            if start_time > last_end:
                segment_path = self.working_dir / f"segment_before_{i}.wav"
                cmd = [
                    "ffmpeg", "-i", str(original_audio),
                    "-ss", str(last_end), "-to", str(start_time),
                    "-c:a", "copy",
                    "-y", str(segment_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                segments.append(segment_path)
            
            # Add replacement
            segments.append(replacement_path)
            last_end = end_time
        
        # Final segment
        video_info = self.get_video_info(video_path)
        if last_end < video_info['duration']:
            final_segment = self.working_dir / "segment_final.wav"
            cmd = [
                "ffmpeg", "-i", str(original_audio),
                "-ss", str(last_end),
                "-c:a", "copy",
                "-y", str(final_segment)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            segments.append(final_segment)
        
        # Concatenate audio segments
        concat_audio = self.working_dir / "concatenated_audio.wav"
        concat_file = self.working_dir / "audio_concat.txt"
        with open(concat_file, "w") as f:
            for segment in segments:
                f.write(f"file '{segment.absolute()}'\n")
        
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c:a", "pcm_s16le",
            "-y", str(concat_audio)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Mux new audio with video
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(concat_audio),
            "-c:v", "copy",  # Copy video stream
            "-c:a", "aac", "-b:a", "192k",  # Encode audio
            "-map", "0:v:0", "-map", "1:a:0",  # Map video from first input, audio from second
            "-shortest",  # End at shortest stream
            "-y", str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Cleanup
        for segment in segments:
            if segment.exists() and segment.parent == self.working_dir:
                segment.unlink()
        original_audio.unlink()
        concat_audio.unlink()
        concat_file.unlink()
        
        return output_path
    
    def add_visual_overlay(self,
                         video_path: Path,
                         overlays: List[Dict],
                         output_path: Path) -> Path:
        """
        Add visual overlays (text, shapes) at specific timestamps
        
        Args:
            video_path: Input video path
            overlays: List of overlay definitions
                     Each overlay dict should have:
                     - type: 'text' or 'box'
                     - start_time: float
                     - end_time: float
                     - properties: dict with overlay-specific properties
            output_path: Output video path
            
        Returns:
            Path to processed video
        """
        # Build complex filter
        filter_parts = []
        
        for overlay in overlays:
            if overlay['type'] == 'text':
                props = overlay['properties']
                filter_parts.append(
                    f"drawtext="
                    f"text='{props.get('text', '')}'"
                    f":fontcolor={props.get('color', 'white')}"
                    f":fontsize={props.get('size', 40)}"
                    f":box={props.get('box', 1)}"
                    f":boxcolor={props.get('boxcolor', 'black@0.8')}"
                    f":x={props.get('x', '(w-text_w)/2')}"
                    f":y={props.get('y', 'h-100')}"
                    f":enable='between(t,{overlay['start_time']},{overlay['end_time']})'"
                )
            elif overlay['type'] == 'box':
                props = overlay['properties']
                filter_parts.append(
                    f"drawbox="
                    f"x={props.get('x', 0)}"
                    f":y={props.get('y', 0)}"
                    f":w={props.get('width', 100)}"
                    f":h={props.get('height', 50)}"
                    f":color={props.get('color', 'red@0.5')}"
                    f":thickness={props.get('thickness', 'fill')}"
                    f":enable='between(t,{overlay['start_time']},{overlay['end_time']})'"
                )
        
        if filter_parts:
            filter_complex = ",".join(filter_parts)
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vf", filter_complex,
                "-c:a", "copy",  # Copy audio
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-y", str(output_path)
            ]
        else:
            # No overlays, just copy
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-c", "copy",
                "-y", str(output_path)
            ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def cleanup(self):
        """Clean up working directory"""
        if self.working_dir.exists() and str(self.working_dir).startswith("/tmp"):
            shutil.rmtree(self.working_dir)


def test_video_processor():
    """Test video processor functionality"""
    processor = VideoProcessor()
    
    # Test video info extraction
    test_video = Path("test.mp4")
    if test_video.exists():
        info = processor.get_video_info(test_video)
        print("Video info:", json.dumps(info, indent=2))
    
    print("Video processor initialized successfully")


if __name__ == "__main__":
    test_video_processor()