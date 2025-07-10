#!/usr/bin/env python3
"""
Example usage of the Veo3 production pipeline
Demonstrates end-to-end character video generation
"""

import asyncio
from pathlib import Path
import logging
from datetime import datetime

# Import pipeline modules
from veo3_pipeline import (
    Veo3Pipeline, 
    KontextConfig, 
    Veo3Config, 
    ElevenLabsConfig
)
from api_clients import APIClientFactory
from character_consistency import CharacterConsistencyManager
from post_production import PostProductionPipeline, ColorGradeProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_character_video_demo():
    """
    Complete example: Create a character video with voice
    Following the production workflow:
    1. FLUX Kontext → Master still
    2. Veo 3 → Animated video
    3. ElevenLabs → Voice cloning
    4. Post-production → Final deliverable
    """
    
    # Initialize pipeline
    pipeline = Veo3Pipeline(project_dir="veo3_projects")
    
    # Create project
    character_name = "Elena_Robotics_Engineer"
    project_id = pipeline.create_project(character_name)
    logger.info(f"Created project: {project_id}")
    
    # ========================================
    # Step 1: Generate Master Still with FLUX Kontext
    # ========================================
    
    kontext_config = KontextConfig(
        model_version="dev",  # Use 'pro' or 'max' for higher quality
        base_prompt="A young female robotics engineer, professional attire, confident expression",
        style_hints="cinematic lighting, detailed, photorealistic, 8k quality",
        negative_prompt="no text, no glare, no artifacts, no blur",
        output_format="16:9"
    )
    
    # Generate master still
    logger.info("Generating master still with FLUX Kontext...")
    still_result = pipeline.generate_master_still(
        project_id=project_id,
        kontext_config=kontext_config
    )
    
    master_still_path = Path(still_result["image_path"])
    logger.info(f"Master still created: {master_still_path}")
    
    # ========================================
    # Step 2: Create Character Profile for Consistency
    # ========================================
    
    consistency_manager = CharacterConsistencyManager(pipeline.project_dir / project_id)
    
    # Create character profile
    character_profile = consistency_manager.create_character_profile(
        character_name="Elena",
        reference_images=[master_still_path],
        style_descriptors=[
            "brunette hair in professional bun",
            "blue eyes",
            "light skin tone",
            "wearing white lab coat",
            "silver earrings"
        ]
    )
    
    # Generate turntable prompts for multiple angles
    turntable_prompts = consistency_manager.generate_turntable_prompts(
        profile=character_profile,
        base_prompt="Elena the robotics engineer in her neon-lit laboratory",
        num_angles=8
    )
    
    # ========================================
    # Step 3: Animate with Veo 3
    # ========================================
    
    veo3_config = Veo3Config(
        duration_seconds=6.0,
        fps=24,
        quality="high",  # Use 'ultra' for final renders
        export_format="prores",
        camera_motion="steady dolly-in"
    )
    
    # Define structured prompt
    prompt_structure = {
        "subject": "Elena, young female robotics engineer",
        "context": "in a high-tech neon-lit laboratory",
        "action": "stands up from workbench, turns 90° left to face robot prototype",
        "style": "cinematic lighting, detailed, professional",
        "camera_motion": "steady dolly-in",
        "composition": "medium shot transitioning to close-up"
    }
    
    logger.info("Generating animated video with Veo 3...")
    animation_result = pipeline.animate_with_veo3(
        project_id=project_id,
        master_still_path=master_still_path,
        veo3_config=veo3_config,
        prompt_structure=prompt_structure
    )
    
    video_path = Path(animation_result["composite_path"])
    logger.info(f"Animation created: {video_path}")
    
    # ========================================
    # Step 4: Voice Cloning with ElevenLabs
    # ========================================
    
    # Prepare voice reference (in production, record 1-3 min of clean audio)
    reference_audio = Path("voice_samples/elena_reference.wav")
    
    elevenlabs_config = ElevenLabsConfig(
        model="eleven_turbo_v2",  # Or "eleven_multilingual_v2" for other languages
        voice_settings={
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.2,
            "use_speaker_boost": True
        }
    )
    
    # Script for the character
    script = """
    Welcome to the future of robotics. 
    I'm Elena, and today I'll show you how our latest AI prototype 
    can learn complex tasks in real-time.
    Watch as it adapts to new challenges autonomously.
    """
    
    logger.info("Cloning voice and generating narration...")
    voice_result = pipeline.clone_voice_elevenlabs(
        project_id=project_id,
        reference_audio_path=reference_audio if reference_audio.exists() else None,
        script_text=script,
        elevenlabs_config=elevenlabs_config
    )
    
    audio_path = Path(voice_result["audio_files"][0]["path"])
    logger.info(f"Voice audio created: {audio_path}")
    
    # ========================================
    # Step 5: Post-Production
    # ========================================
    
    post_pipeline = PostProductionPipeline(pipeline.project_dir / project_id)
    
    # Sync audio to video
    logger.info("Syncing audio to video...")
    synced_video = post_pipeline.sync_audio_video(
        video_path=video_path,
        audio_path=audio_path,
        drift_correction=True
    )
    
    # Apply cinematic color grade
    color_grade = ColorGradeProfile(
        name="Cinematic Tech",
        adjustments={
            "exposure": 0.2,
            "contrast": 1.1,
            "saturation": 0.9,
            "temperature": -5  # Slightly cooler for tech feel
        }
    )
    
    logger.info("Applying color grade...")
    graded_video = post_pipeline.apply_color_grade(
        video_path=synced_video,
        grade_profile=color_grade,
        preserve_skin_tones=True
    )
    
    # Master audio to broadcast standards
    logger.info("Mastering audio...")
    mastered_video = post_pipeline.master_audio(
        video_path=graded_video,
        target_lufs=-14.0,  # YouTube/streaming standard
        peak_limit=-1.0,
        dynamic_range=7.0
    )
    
    # Export deliverables
    logger.info("Exporting final deliverables...")
    exports = post_pipeline.export_deliverables(
        video_path=mastered_video,
        preset="web_4k",
        add_metadata=True
    )
    
    # ========================================
    # Step 6: Quality Validation
    # ========================================
    
    from post_production import QualityControl
    qc = QualityControl()
    
    # Validate sync
    sync_valid = qc.validate_sync(exports["web_4k"])
    logger.info(f"Audio sync validation: {'PASS' if sync_valid else 'FAIL'}")
    
    # Check quality metrics
    metrics = qc.measure_quality_metrics(exports["web_4k"])
    logger.info(f"Quality metrics: {metrics}")
    
    # ========================================
    # Final Summary
    # ========================================
    
    logger.info("\n" + "="*50)
    logger.info("VIDEO GENERATION COMPLETE")
    logger.info("="*50)
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Character: {character_name}")
    logger.info(f"Duration: {veo3_config.duration_seconds}s @ {veo3_config.fps}fps")
    logger.info("\nDeliverables:")
    for preset, path in exports.items():
        logger.info(f"  - {preset}: {path}")
    logger.info("\nAssets archived at:")
    logger.info(f"  - {pipeline.project_dir / project_id}")
    
    return exports


def example_batch_processing():
    """Example: Process multiple characters in batch"""
    
    characters = [
        {
            "name": "Dr_Sarah_Chen",
            "prompt": "Asian female scientist in modern laboratory",
            "script": "Quantum computing will revolutionize how we process information..."
        },
        {
            "name": "Marcus_Johnson",
            "prompt": "African American male teacher in classroom",
            "script": "Today's lesson explores the wonders of our solar system..."
        },
        {
            "name": "Sofia_Martinez",
            "prompt": "Latina female chef in professional kitchen",
            "script": "The secret to perfect paella lies in the sofrito base..."
        }
    ]
    
    pipeline = Veo3Pipeline()
    
    for character in characters:
        logger.info(f"\nProcessing character: {character['name']}")
        
        # Create project
        project_id = pipeline.create_project(character["name"])
        
        # Configure generation
        kontext_config = KontextConfig(
            base_prompt=character["prompt"],
            style_hints="photorealistic, professional lighting"
        )
        
        # Generate still
        still_result = pipeline.generate_master_still(
            project_id=project_id,
            kontext_config=kontext_config
        )
        
        # Continue with animation, voice, and post-production...
        logger.info(f"Completed: {character['name']}")


def example_api_integration():
    """Example: Direct API client usage"""
    
    # Create API clients
    flux_client = APIClientFactory.create_flux_client()
    veo3_client = APIClientFactory.create_veo3_client()
    elevenlabs_client = APIClientFactory.create_elevenlabs_client()
    
    # Generate image with FLUX
    image_result = flux_client.generate_image(
        prompt="A futuristic robot assistant, sleek design",
        negative_prompt="no text, no logos",
        model_version="pro",
        width=1920,
        height=1080
    )
    
    # Animate with Veo 3
    async def animate():
        video_result = await veo3_client.generate_video_async(
            conditioning_frame=Path(image_result["output_path"]),
            prompt="Robot assistant activates and greets viewer",
            duration=4.0,
            quality="high"
        )
        return video_result
    
    # Generate voice
    voice_audio = elevenlabs_client.generate_audio(
        text="Hello, I am your AI assistant. How may I help you today?",
        voice_id="your_voice_id",
        model_id="eleven_turbo_v2"
    )
    
    logger.info("API integration example completed")


if __name__ == "__main__":
    # Run the main demo
    asyncio.run(create_character_video_demo())
    
    # Uncomment to run other examples:
    # example_batch_processing()
    # example_api_integration()