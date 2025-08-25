#!/usr/bin/env python3
"""
Test script to verify all dependencies and functionality are working correctly.
Run this script to ensure your development environment is properly set up.
"""

import sys
import importlib
import traceback

def test_import(module_name, description=""):
    """Test if a module can be imported successfully."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown version')
        print(f"✅ {module_name} ({description}): {version}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} ({description}): Import failed - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name} ({description}): Unexpected error - {e}")
        return False

def test_pytorch():
    """Test PyTorch functionality."""
    try:
        import torch
        import torchvision
        
        # Test basic tensor operations
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.mm(x, y)
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        
        print(f"✅ PyTorch: Basic operations working")
        print(f"   - CUDA available: {cuda_available}")
        if cuda_available:
            print(f"   - GPU count: {cuda_count}")
        else:
            print(f"   - Using CPU (this is fine for development)")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_opencv():
    """Test OpenCV functionality."""
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, (255, 255, 255), -1)
        
        # Test basic operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (50, 50))
        
        print(f"✅ OpenCV: Image operations working")
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        traceback.print_exc()
        return False

def test_project_modules():
    """Test project-specific modules."""
    try:
        # Add project root to path
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(project_root, 'src')
        sys.path.insert(0, src_path)
        
        from drivable_corridor.models.cnn import CNN
        import torch
        
        # Test model creation
        model = CNN()
        param_count = sum(p.numel() for p in model.parameters())
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 360, 640)
        output = model(dummy_input)
        
        print(f"✅ Project CNN model: Working correctly")
        print(f"   - Parameters: {param_count:,}")
        print(f"   - Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Project module test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔍 Testing Development Environment Setup")
    print("=" * 50)
    
    # Core Python libraries
    test_import("torch", "PyTorch")
    test_import("torchvision", "TorchVision")
    test_import("cv2", "OpenCV")
    test_import("matplotlib", "Matplotlib")
    test_import("numpy", "NumPy")
    test_import("tqdm", "Progress bars")
    test_import("wandb", "Weights & Biases")
    
    print("\n🧪 Testing Functionality")
    print("=" * 50)
    
    # Functional tests
    pytorch_ok = test_pytorch()
    opencv_ok = test_opencv()
    project_ok = test_project_modules()
    
    print("\n📊 Summary")
    print("=" * 50)
    
    if pytorch_ok and opencv_ok and project_ok:
        print("🎉 All tests passed! Your development environment is ready.")
        print("\nYou can now:")
        print("   - Run training: cd scripts && python train.py")
        print("   - Run inference: cd scripts && python inference.py")
        print("   - Develop new features in src/drivable_corridor/")
        print("   - Run tests: pytest tests/")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
