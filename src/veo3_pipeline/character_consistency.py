#!/usr/bin/env python3
"""
Character consistency module for maintaining identity across angles and scenes
Implements turntable generation and LoRA/DreamBooth preparation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import cv2
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CharacterProfile:
    """Character identity profile"""
    character_id: str
    reference_images: List[Path]
    feature_embeddings: Dict[str, np.ndarray]
    style_descriptors: List[str]
    identity_markers: Dict[str, Any]
    

class CharacterConsistencyManager:
    """Manage character consistency across generation pipeline"""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.profiles_dir = project_dir / "character_profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        
    def create_character_profile(
        self,
        character_name: str,
        reference_images: List[Path],
        style_descriptors: Optional[List[str]] = None
    ) -> CharacterProfile:
        """Create a new character profile from reference images"""
        
        character_id = self._generate_character_id(character_name)
        profile_dir = self.profiles_dir / character_id
        profile_dir.mkdir(exist_ok=True)
        
        # Extract visual features
        feature_embeddings = {}
        for img_path in reference_images:
            features = self._extract_visual_features(img_path)
            feature_embeddings[img_path.stem] = features
            
        # Identify key markers (scars, tattoos, jewelry, etc.)
        identity_markers = self._identify_markers(reference_images)
        
        # Default style descriptors
        if style_descriptors is None:
            style_descriptors = [
                "consistent facial features",
                "matching skin tone",
                "identical eye color",
                "same hair style and color"
            ]
            
        profile = CharacterProfile(
            character_id=character_id,
            reference_images=reference_images,
            feature_embeddings=feature_embeddings,
            style_descriptors=style_descriptors,
            identity_markers=identity_markers
        )
        
        # Save profile
        self._save_profile(profile)
        
        logger.info(f"Created character profile: {character_id}")
        return profile
        
    def generate_turntable_prompts(
        self,
        profile: CharacterProfile,
        base_prompt: str,
        num_angles: int = 8
    ) -> List[Dict[str, str]]:
        """Generate prompts for turntable views"""
        
        angles = np.linspace(0, 360, num_angles, endpoint=False)
        turntable_prompts = []
        
        # Include character-specific descriptors
        character_desc = ", ".join(profile.style_descriptors)
        
        for i, angle in enumerate(angles):
            # Map angle to view description
            view_desc = self._angle_to_view_description(angle)
            
            # Add identity markers to prompt
            markers_desc = self._format_identity_markers(profile.identity_markers)
            
            prompt = {
                "angle": float(angle),
                "view": view_desc,
                "prompt": f"{base_prompt}, {view_desc} view, {character_desc}, {markers_desc}",
                "negative_prompt": "different person, inconsistent features, changing appearance"
            }
            
            turntable_prompts.append(prompt)
            
        return turntable_prompts
        
    def prepare_lora_dataset(
        self,
        profile: CharacterProfile,
        turntable_images: List[Path]
    ) -> Path:
        """Prepare dataset for LoRA fine-tuning"""
        
        dataset_dir = self.profiles_dir / profile.character_id / "lora_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        # Structure for LoRA training
        train_dir = dataset_dir / "train" / profile.character_id
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and preprocess images
        metadata = []
        for i, img_path in enumerate(turntable_images):
            # Standardize image
            processed_img = self._preprocess_for_training(img_path)
            
            # Save with consistent naming
            output_path = train_dir / f"{profile.character_id}_{i:03d}.png"
            cv2.imwrite(str(output_path), processed_img)
            
            # Create caption file
            caption_path = output_path.with_suffix(".txt")
            caption = f"a photo of {profile.character_id} person"
            caption_path.write_text(caption)
            
            metadata.append({
                "file_name": output_path.name,
                "caption": caption,
                "tags": profile.style_descriptors
            })
            
        # Save metadata
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "character_id": profile.character_id,
                "num_images": len(metadata),
                "images": metadata
            }, f, indent=2)
            
        logger.info(f"Prepared LoRA dataset at {dataset_dir}")
        return dataset_dir
        
    def validate_consistency(
        self,
        generated_images: List[Path],
        profile: CharacterProfile,
        threshold: float = 0.85
    ) -> Dict[str, float]:
        """Validate character consistency across generated images"""
        
        scores = {}
        reference_features = list(profile.feature_embeddings.values())[0]
        
        for img_path in generated_images:
            # Extract features from generated image
            generated_features = self._extract_visual_features(img_path)
            
            # Calculate similarity
            similarity = self._calculate_similarity(
                reference_features, 
                generated_features
            )
            
            scores[img_path.name] = similarity
            
            if similarity < threshold:
                logger.warning(
                    f"Low consistency score ({similarity:.2f}) for {img_path.name}"
                )
                
        # Overall consistency
        avg_score = np.mean(list(scores.values()))
        scores["average"] = avg_score
        
        return scores
        
    def _generate_character_id(self, name: str) -> str:
        """Generate unique character ID"""
        return f"{name.lower().replace(' ', '_')}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
    def _extract_visual_features(self, image_path: Path) -> np.ndarray:
        """Extract visual features from image"""
        # Placeholder - would use face recognition model
        img = cv2.imread(str(image_path))
        if img is None:
            return np.zeros(512)  # Typical embedding size
            
        # Simple feature extraction (replace with proper model)
        img_resized = cv2.resize(img, (224, 224))
        features = img_resized.flatten()
        
        # Normalize to embedding size
        feature_vector = np.zeros(512)
        feature_vector[:min(512, len(features))] = features[:512]
        
        return feature_vector / np.linalg.norm(feature_vector)
        
    def _identify_markers(self, images: List[Path]) -> Dict[str, Any]:
        """Identify distinguishing markers in character"""
        # Placeholder - would use object detection
        return {
            "has_glasses": False,
            "hair_color": "brown",
            "eye_color": "blue",
            "distinctive_features": []
        }
        
    def _angle_to_view_description(self, angle: float) -> str:
        """Convert angle to view description"""
        if angle < 22.5 or angle >= 337.5:
            return "front facing"
        elif angle < 67.5:
            return "three quarter right"
        elif angle < 112.5:
            return "right profile"
        elif angle < 157.5:
            return "three quarter back right"
        elif angle < 202.5:
            return "back facing"
        elif angle < 247.5:
            return "three quarter back left"
        elif angle < 292.5:
            return "left profile"
        else:
            return "three quarter left"
            
    def _format_identity_markers(self, markers: Dict) -> str:
        """Format identity markers for prompt"""
        parts = []
        
        if markers.get("hair_color"):
            parts.append(f"{markers['hair_color']} hair")
        if markers.get("eye_color"):
            parts.append(f"{markers['eye_color']} eyes")
        if markers.get("has_glasses"):
            parts.append("wearing glasses")
            
        for feature in markers.get("distinctive_features", []):
            parts.append(feature)
            
        return ", ".join(parts)
        
    def _preprocess_for_training(self, image_path: Path) -> np.ndarray:
        """Preprocess image for LoRA training"""
        img = cv2.imread(str(image_path))
        
        # Resize to training resolution
        img_resized = cv2.resize(img, (512, 512))
        
        # Normalize and enhance
        img_normalized = cv2.normalize(
            img_resized, None, 0, 255, 
            cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        return img_normalized
        
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors"""
        dot_product = np.dot(features1, features2)
        norm_product = np.linalg.norm(features1) * np.linalg.norm(features2)
        
        if norm_product == 0:
            return 0.0
            
        return float(dot_product / norm_product)
        
    def _save_profile(self, profile: CharacterProfile):
        """Save character profile to disk"""
        profile_path = self.profiles_dir / profile.character_id / "profile.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = {
            k: v.tolist() for k, v in profile.feature_embeddings.items()
        }
        
        profile_data = {
            "character_id": profile.character_id,
            "reference_images": [str(p) for p in profile.reference_images],
            "feature_embeddings": serializable_embeddings,
            "style_descriptors": profile.style_descriptors,
            "identity_markers": profile.identity_markers
        }
        
        with open(profile_path, "w") as f:
            json.dump(profile_data, f, indent=2)
            
    def load_profile(self, character_id: str) -> Optional[CharacterProfile]:
        """Load character profile from disk"""
        profile_path = self.profiles_dir / character_id / "profile.json"
        
        if not profile_path.exists():
            return None
            
        with open(profile_path) as f:
            data = json.load(f)
            
        # Convert lists back to numpy arrays
        feature_embeddings = {
            k: np.array(v) for k, v in data["feature_embeddings"].items()
        }
        
        return CharacterProfile(
            character_id=data["character_id"],
            reference_images=[Path(p) for p in data["reference_images"]],
            feature_embeddings=feature_embeddings,
            style_descriptors=data["style_descriptors"],
            identity_markers=data["identity_markers"]
        )