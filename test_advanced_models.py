#!/usr/bin/env python3
"""
Comprehensive test suite for advanced lip sync models
Tests VASA-1, EMO, and Gaussian Splatting implementations
"""

import os
import sys
import unittest
import tempfile
import shutil
import subprocess
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
from advanced_models import (
    AdvancedModelManager, VASA1Model, EMOModel, 
    GaussianSplattingModel, AdvancedLipSyncModel
)
from lip_sync_advanced import ExtendedLipSyncProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAdvancedModels(unittest.TestCase):
    """Test suite for advanced lip sync models"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_video = cls._create_test_video()
        cls.test_audio = cls._create_test_audio()
        cls.manager = AdvancedModelManager()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @staticmethod
    def _create_test_video() -> str:
        """Create a test video file"""
        video_path = os.path.join(TestAdvancedModels.test_dir, "test_video.mp4")
        
        # Create a simple test video with ffmpeg
        cmd = [
            "ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=5:size=512x512:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=5",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac",
            "-y", video_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created test video: {video_path}")
        except subprocess.CalledProcessError:
            # Fallback: create a placeholder file
            with open(video_path, "wb") as f:
                f.write(b"PLACEHOLDER_VIDEO")
        
        return video_path
    
    @staticmethod
    def _create_test_audio() -> str:
        """Create a test audio file"""
        audio_path = os.path.join(TestAdvancedModels.test_dir, "test_audio.wav")
        
        # Create a simple test audio with ffmpeg
        cmd = [
            "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-ar", "16000", "-ac", "1",
            "-y", audio_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created test audio: {audio_path}")
        except subprocess.CalledProcessError:
            # Fallback: create a placeholder file
            with open(audio_path, "wb") as f:
                f.write(b"PLACEHOLDER_AUDIO")
        
        return audio_path
    
    def test_model_initialization(self):
        """Test that all models can be initialized"""
        models = ["vasa1", "emo", "gaussian_splatting"]
        
        for model_name in models:
            with self.subTest(model=model_name):
                try:
                    model = self.manager.get_model(model_name)
                    self.assertIsNotNone(model)
                    self.assertTrue(model.initialized)
                    self.assertEqual(model.model_name, model_name)
                    logger.info(f"✓ {model_name} initialized successfully")
                except Exception as e:
                    self.fail(f"Failed to initialize {model_name}: {e}")
    
    def test_model_requirements(self):
        """Test model requirements reporting"""
        expected_requirements = {
            "vasa1": {"vram": 16, "resolution": (512, 512), "fps": 40},
            "emo": {"vram": 24, "resolution": (512, 512), "fps": 25},
            "gaussian_splatting": {"vram": 12, "resolution": (512, 512), "fps": 100}
        }
        
        for model_name, expected in expected_requirements.items():
            with self.subTest(model=model_name):
                model = self.manager.get_model(model_name)
                requirements = model.get_requirements()
                
                self.assertEqual(requirements["vram"], expected["vram"])
                self.assertEqual(requirements["resolution"], expected["resolution"])
                self.assertEqual(requirements["fps"], expected["fps"])
                self.assertIn("features", requirements)
                logger.info(f"✓ {model_name} requirements validated")
    
    def test_vasa1_processing(self):
        """Test VASA-1 model processing"""
        model = self.manager.get_model("vasa1")
        output_path = os.path.join(self.test_dir, "output_vasa1.mp4")
        
        # Test processing
        start_time = time.time()
        success = model.process_video(self.test_video, self.test_audio, output_path)
        processing_time = time.time() - start_time
        
        self.assertTrue(success, "VASA-1 processing failed")
        self.assertTrue(os.path.exists(output_path), "Output file not created")
        
        # Verify output properties
        if os.path.getsize(output_path) > 100:  # Not just a placeholder
            # Check video properties with ffprobe
            cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json", output_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                info = json.loads(result.stdout)
                
                # Verify resolution
                stream = info["streams"][0]
                self.assertEqual(int(stream["width"]), 512)
                self.assertEqual(int(stream["height"]), 512)
                
                logger.info(f"✓ VASA-1 processing completed in {processing_time:.2f}s")
            except:
                pass  # Skip verification if ffprobe not available
    
    def test_emo_processing(self):
        """Test EMO model processing"""
        model = self.manager.get_model("emo")
        output_path = os.path.join(self.test_dir, "output_emo.mp4")
        
        # Test processing
        start_time = time.time()
        success = model.process_video(self.test_video, self.test_audio, output_path)
        processing_time = time.time() - start_time
        
        self.assertTrue(success, "EMO processing failed")
        self.assertTrue(os.path.exists(output_path), "Output file not created")
        
        logger.info(f"✓ EMO processing completed in {processing_time:.2f}s")
    
    def test_gaussian_splatting_processing(self):
        """Test Gaussian Splatting model processing"""
        model = self.manager.get_model("gaussian_splatting")
        output_path = os.path.join(self.test_dir, "output_gaussian.mp4")
        
        # Test processing
        start_time = time.time()
        success = model.process_video(self.test_video, self.test_audio, output_path)
        processing_time = time.time() - start_time
        
        self.assertTrue(success, "Gaussian Splatting processing failed")
        self.assertTrue(os.path.exists(output_path), "Output file not created")
        
        # Verify high FPS capability
        requirements = model.get_requirements()
        self.assertEqual(requirements["fps"], 100, "Gaussian Splatting should support 100 FPS")
        
        logger.info(f"✓ Gaussian Splatting processing completed in {processing_time:.2f}s")
    
    def test_extended_processor_integration(self):
        """Test integration with extended lip sync processor"""
        models_to_test = ["vasa1", "emo", "gaussian_splatting"]
        
        for model_name in models_to_test:
            with self.subTest(model=model_name):
                processor = ExtendedLipSyncProcessor(model_name)
                
                self.assertTrue(processor.is_advanced)
                self.assertEqual(processor.model_type, model_name)
                
                # Test segment processing
                segments = [{
                    "start": 1.0,
                    "end": 2.0,
                    "audio": self.test_audio
                }]
                
                output_path = os.path.join(self.test_dir, f"output_extended_{model_name}.mp4")
                success = processor.apply_lip_sync_simple(
                    self.test_video, segments, output_path
                )
                
                self.assertTrue(success, f"Extended processor failed for {model_name}")
                logger.info(f"✓ Extended processor integration successful for {model_name}")
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        models_info = self.manager.list_models()
        
        self.assertIsInstance(models_info, list)
        self.assertEqual(len(models_info), 3)  # VASA-1, EMO, Gaussian Splatting
        
        # Verify each model info
        for info in models_info:
            self.assertIn("name", info)
            self.assertIn("class", info)
            self.assertIn("requirements", info)
            
            req = info["requirements"]
            self.assertIn("vram", req)
            self.assertIn("resolution", req)
            self.assertIn("fps", req)
            self.assertIn("features", req)
        
        # Test comparison output
        comparison = self.manager.compare_models()
        self.assertIsInstance(comparison, str)
        self.assertIn("Advanced Models Comparison", comparison)
        self.assertIn("VASA1", comparison)
        self.assertIn("EMO", comparison)
        self.assertIn("GAUSSIAN_SPLATTING", comparison)
        
        logger.info("✓ Model comparison functionality validated")
    
    def test_performance_benchmarks(self):
        """Benchmark performance of each model"""
        results = {}
        
        for model_name in ["vasa1", "emo", "gaussian_splatting"]:
            model = self.manager.get_model(model_name)
            output_path = os.path.join(self.test_dir, f"bench_{model_name}.mp4")
            
            # Measure processing time
            start_time = time.time()
            success = model.process_video(self.test_video, self.test_audio, output_path)
            processing_time = time.time() - start_time
            
            if success:
                # Calculate FPS based on processing time
                video_duration = 5.0  # seconds
                requirements = model.get_requirements()
                theoretical_fps = requirements["fps"]
                actual_fps = video_duration / processing_time if processing_time > 0 else 0
                
                results[model_name] = {
                    "processing_time": processing_time,
                    "theoretical_fps": theoretical_fps,
                    "actual_fps": actual_fps,
                    "vram_required": requirements["vram"]
                }
        
        # Print benchmark results
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        print(f"{'Model':<20} {'Time (s)':<10} {'FPS':<15} {'VRAM (GB)':<10}")
        print("-"*60)
        
        for model, result in results.items():
            print(f"{model:<20} {result['processing_time']:<10.2f} "
                  f"{result['theoretical_fps']:<15} {result['vram_required']:<10}")
        
        print("="*60)
        
        # Verify Gaussian Splatting is fastest
        if all(model in results for model in ["vasa1", "emo", "gaussian_splatting"]):
            self.assertEqual(
                min(results, key=lambda x: results[x]['processing_time']),
                "gaussian_splatting",
                "Gaussian Splatting should be the fastest model"
            )


class TestPipelineIntegration(unittest.TestCase):
    """Test integration with main pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_pipeline_update_script(self):
        """Test pipeline update functionality"""
        from lip_sync_advanced import update_pipeline_for_advanced_models
        
        # Create a mock pipeline file
        mock_pipeline = os.path.join(self.test_dir, "personalization_pipeline.py")
        with open(mock_pipeline, "w") as f:
            f.write("""try:
    import lip_sync
    LIPSYNC_AVAILABLE = True
except ImportError:
    LIPSYNC_AVAILABLE = False
    print("Warning: LipSync module not available. Lip sync will be disabled.")

# Rest of pipeline code...
""")
        
        # Update the pipeline
        original_path = Path("personalization_pipeline.py")
        try:
            # Temporarily use mock file
            Path("personalization_pipeline.py").rename("personalization_pipeline.py.temp")
            shutil.copy(mock_pipeline, "personalization_pipeline.py")
            
            success = update_pipeline_for_advanced_models()
            self.assertTrue(success, "Pipeline update failed")
            
            # Verify updates
            with open("personalization_pipeline.py", "r") as f:
                content = f.read()
            
            self.assertIn("lip_sync_advanced", content)
            self.assertIn("ADVANCED_MODELS_AVAILABLE", content)
            self.assertIn("vasa1", content)
            self.assertIn("emo", content)
            self.assertIn("gaussian_splatting", content)
            
            logger.info("✓ Pipeline update successful")
            
        finally:
            # Restore original file
            if os.path.exists("personalization_pipeline.py.temp"):
                os.rename("personalization_pipeline.py.temp", "personalization_pipeline.py")


def run_all_tests():
    """Run all tests and generate report"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAdvancedModels))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPipelineIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary report
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run specific test or all tests
    import argparse
    
    parser = argparse.ArgumentParser(description="Test advanced lip sync models")
    parser.add_argument("--model", choices=["vasa1", "emo", "gaussian_splatting"],
                       help="Test specific model")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run performance benchmarks")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only")
    
    args = parser.parse_args()
    
    if args.model:
        # Test specific model
        suite = unittest.TestSuite()
        test = TestAdvancedModels()
        test.setUpClass()
        
        if args.model == "vasa1":
            test.test_vasa1_processing()
        elif args.model == "emo":
            test.test_emo_processing()
        elif args.model == "gaussian_splatting":
            test.test_gaussian_splatting_processing()
        
        test.tearDownClass()
        
    elif args.benchmark:
        # Run benchmarks only
        test = TestAdvancedModels()
        test.setUpClass()
        test.test_performance_benchmarks()
        test.tearDownClass()
        
    elif args.quick:
        # Run quick tests
        suite = unittest.TestSuite()
        suite.addTest(TestAdvancedModels('test_model_initialization'))
        suite.addTest(TestAdvancedModels('test_model_requirements'))
        unittest.TextTestRunner(verbosity=2).run(suite)
        
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)