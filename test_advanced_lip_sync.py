#!/usr/bin/env python3
"""
Test script for all advanced lip sync models
Tests MuseTalk, LatentSync, VASA-1, EMO, and Gaussian Splatting
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.lip_sync.advanced_smart_selector import advanced_smart_selector, ProcessingOptions


def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_advanced_lip_sync.log')
        ]
    )


def test_individual_models():
    """Test each model individually"""
    
    # Test video and audio paths
    video_path = "test_assets/input/test_video.mp4"
    audio_path = "test_assets/input/test_audio.wav"
    
    models_to_test = [
        ("musetalk", "MuseTalk - Real-time high quality"),
        ("latentsync", "LatentSync - Stable Diffusion based"),
        ("vasa1", "VASA-1 - Microsoft expressive"),
        ("emo", "EMO - Emotional expressions"),
        ("gaussian_splatting", "Gaussian Splatting - Ultra-fast 3D")
    ]
    
    results = {}
    
    for model_name, description in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {description}")
        print(f"{'='*60}")
        
        try:
            # Check if model is available
            method_info = advanced_smart_selector.get_method_info(model_name)
            
            if method_info is None:
                print(f"❌ {model_name} not available")
                results[model_name] = {"available": False, "error": "Not found"}
                continue
            
            print(f"✅ {model_name} is available")
            print(f"   Quality Score: {method_info['quality_score']}")
            print(f"   VRAM Required: {method_info['min_vram_gb']}GB")
            print(f"   Resolution: {method_info['resolution']}")
            print(f"   Real-time: {method_info.get('supports_real_time', False)}")
            print(f"   Emotions: {method_info.get('supports_emotions', False)}")
            print(f"   3D: {method_info.get('supports_3d', False)}")
            
            # Check system compatibility
            manager = method_info.get("manager")
            if manager and hasattr(manager, 'is_available'):
                system_compatible = manager.is_available()
                print(f"   System Compatible: {system_compatible}")
                
                if system_compatible:
                    # Test processing
                    output_path = f"test_assets/output/{model_name}_output.mp4"
                    
                    print(f"   Testing processing...")
                    success = manager.process_video(video_path, audio_path, output_path)
                    
                    if success:
                        print(f"   ✅ Processing successful: {output_path}")
                        results[model_name] = {
                            "available": True,
                            "compatible": True,
                            "processed": True,
                            "output": output_path
                        }
                    else:
                        print(f"   ❌ Processing failed")
                        results[model_name] = {
                            "available": True,
                            "compatible": True,
                            "processed": False,
                            "error": "Processing failed"
                        }
                else:
                    print(f"   ⚠️  System not compatible")
                    results[model_name] = {
                        "available": True,
                        "compatible": False,
                        "error": "System incompatible"
                    }
            else:
                print(f"   ⚠️  Cannot test system compatibility")
                results[model_name] = {
                    "available": True,
                    "compatible": "unknown",
                    "error": "No compatibility check"
                }
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            results[model_name] = {"available": False, "error": str(e)}
    
    return results


def test_smart_selector():
    """Test the smart selector with different scenarios"""
    
    print(f"\n{'='*60}")
    print("Testing Smart Selector")
    print(f"{'='*60}")
    
    video_path = "test_assets/input/test_video.mp4"
    audio_path = "test_assets/input/test_audio.wav"
    
    test_scenarios = [
        {
            "name": "Quality Priority",
            "options": ProcessingOptions(
                quality_priority=True,
                max_cost_usd=10.0,
                max_processing_time_seconds=1800
            )
        },
        {
            "name": "Speed Priority", 
            "options": ProcessingOptions(
                quality_priority=False,
                require_real_time=True,
                max_processing_time_seconds=60
            )
        },
        {
            "name": "Emotional Expressions",
            "options": ProcessingOptions(
                quality_priority=True,
                enable_emotions=True,
                max_cost_usd=5.0
            )
        },
        {
            "name": "3D Processing",
            "options": ProcessingOptions(
                quality_priority=False,
                enable_3d=True,
                require_real_time=True
            )
        },
        {
            "name": "Budget Constrained",
            "options": ProcessingOptions(
                quality_priority=True,
                max_cost_usd=0.0,  # Free only
                prefer_local=True
            )
        }
    ]
    
    selector_results = {}
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        try:
            selected_method = advanced_smart_selector.select_best_method(
                video_path, scenario['options']
            )
            
            method_info = advanced_smart_selector.get_method_info(selected_method)
            
            print(f"Selected Method: {selected_method}")
            if method_info:
                print(f"Quality Score: {method_info['quality_score']}")
                print(f"Speed Multiplier: {method_info['speed_multiplier']}")
                print(f"Cost per Second: ${method_info['cost_per_second']}")
            
            # Test actual processing
            output_path = f"test_assets/output/selector_{scenario['name'].lower().replace(' ', '_')}.mp4"
            
            success, used_method = advanced_smart_selector.process_video(
                video_path, audio_path, output_path, scenario['options']
            )
            
            if success:
                print(f"✅ Processing successful with {used_method}")
                selector_results[scenario['name']] = {
                    "selected": selected_method,
                    "used": used_method,
                    "success": True,
                    "output": output_path
                }
            else:
                print(f"❌ Processing failed")
                selector_results[scenario['name']] = {
                    "selected": selected_method,
                    "used": used_method,
                    "success": False
                }
                
        except Exception as e:
            print(f"❌ Error in scenario {scenario['name']}: {e}")
            selector_results[scenario['name']] = {"error": str(e)}
    
    return selector_results


def test_system_capabilities():
    """Test system capability detection"""
    
    print(f"\n{'='*60}")
    print("System Capabilities")
    print(f"{'='*60}")
    
    caps = advanced_smart_selector.system_caps
    
    print(f"CUDA Available: {caps.has_cuda}")
    print(f"MPS Available: {caps.has_mps}")
    print(f"Total VRAM: {caps.total_vram_gb:.1f} GB")
    print(f"Available VRAM: {caps.available_vram_gb:.1f} GB")
    print(f"CPU Cores: {caps.cpu_cores}")
    print(f"CUDA Compute Capability: {caps.cuda_compute_capability}")
    
    print(f"\nAvailable Methods:")
    for method_name in advanced_smart_selector.list_available_methods():
        method_info = advanced_smart_selector.get_method_info(method_name)
        if method_info:
            print(f"  {method_name}: Quality {method_info['quality_score']}, "
                  f"VRAM {method_info['min_vram_gb']}GB, "
                  f"Type {method_info['type']}")


def generate_test_report(individual_results, selector_results):
    """Generate comprehensive test report"""
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST REPORT")
    print(f"{'='*60}")
    
    # Individual model results
    print("\n📊 Individual Model Test Results:")
    print("-" * 40)
    
    for model_name, result in individual_results.items():
        status = "✅" if result.get("processed", False) else "❌"
        print(f"{status} {model_name.upper()}")
        
        if result.get("available", False):
            if result.get("compatible", False):
                if result.get("processed", False):
                    print(f"    Status: Fully functional")
                    print(f"    Output: {result.get('output', 'N/A')}")
                else:
                    print(f"    Status: Compatible but processing failed")
                    print(f"    Error: {result.get('error', 'Unknown')}")
            else:
                print(f"    Status: Available but not compatible with system")
                print(f"    Reason: {result.get('error', 'System requirements not met')}")
        else:
            print(f"    Status: Not available")
            print(f"    Reason: {result.get('error', 'Model not found')}")
        print()
    
    # Smart selector results
    print("🧠 Smart Selector Test Results:")
    print("-" * 40)
    
    for scenario_name, result in selector_results.items():
        status = "✅" if result.get("success", False) else "❌"
        print(f"{status} {scenario_name}")
        
        if "error" not in result:
            print(f"    Selected: {result.get('selected', 'N/A')}")
            print(f"    Used: {result.get('used', 'N/A')}")
            
            if result.get("success", False):
                print(f"    Output: {result.get('output', 'N/A')}")
            else:
                print(f"    Status: Selection successful but processing failed")
        else:
            print(f"    Error: {result['error']}")
        print()
    
    # Summary
    working_models = sum(1 for r in individual_results.values() if r.get("processed", False))
    total_models = len(individual_results)
    working_scenarios = sum(1 for r in selector_results.values() if r.get("success", False))
    total_scenarios = len(selector_results)
    
    print("📈 Summary:")
    print("-" * 40)
    print(f"Working Models: {working_models}/{total_models}")
    print(f"Working Scenarios: {working_scenarios}/{total_scenarios}")
    print(f"Success Rate: {((working_models + working_scenarios) / (total_models + total_scenarios) * 100):.1f}%")


def main():
    """Main test function"""
    
    setup_logging()
    
    print("🚀 Advanced Lip Sync Models Test Suite")
    print("=" * 60)
    
    # Ensure test directories exist
    os.makedirs("test_assets/input", exist_ok=True)
    os.makedirs("test_assets/output", exist_ok=True)
    
    # Create dummy test files if they don't exist
    video_path = Path("test_assets/input/test_video.mp4")
    audio_path = Path("test_assets/input/test_audio.wav")
    
    if not video_path.exists():
        print("⚠️  Test video not found. Creating dummy file.")
        with open(video_path, 'w') as f:
            f.write("dummy_video_content")
    
    if not audio_path.exists():
        print("⚠️  Test audio not found. Creating dummy file.")
        with open(audio_path, 'w') as f:
            f.write("dummy_audio_content")
    
    # Run tests
    print("\n🔍 Testing system capabilities...")
    test_system_capabilities()
    
    print("\n🧪 Testing individual models...")
    individual_results = test_individual_models()
    
    print("\n🤖 Testing smart selector...")
    selector_results = test_smart_selector()
    
    # Generate final report
    generate_test_report(individual_results, selector_results)
    
    print(f"\n✅ Test completed! Check test_advanced_lip_sync.log for detailed logs.")


if __name__ == "__main__":
    main()