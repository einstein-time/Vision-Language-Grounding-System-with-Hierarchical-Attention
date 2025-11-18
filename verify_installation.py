"""
Installation verification script
Checks that all components are properly installed and functional
"""

import sys
import os


def check_imports():
    """Check that all required packages can be imported"""
    print("Checking package imports...")
    packages = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'matplotlib',
        'tqdm',
        'cv2'
    ]

    failed = []
    for package in packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            failed.append(package)

    return len(failed) == 0


def check_project_structure():
    """Check that all project files exist"""
    print("\nChecking project structure...")

    required_files = [
        'config.py',
        'train.py',
        'inference.py',
        'evaluate.py',
        'requirements.txt',
        'README.md',
        'models/__init__.py',
        'models/vision_transformer.py',
        'models/text_encoder.py',
        'models/fusion_module.py',
        'models/vlg_model.py',
        'data/__init__.py',
        'data/dataset.py',
        'utils/__init__.py',
        'utils/visualization.py',
        'tests/__init__.py',
        'tests/test_models.py'
    ]

    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing.append(file)

    return len(missing) == 0


def test_model_creation():
    """Test that model can be created"""
    print("\nTesting model creation...")

    try:
        from config import get_config
        from models import create_vlg_model

        config = get_config()
        model = create_vlg_model(config)

        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Total parameters: {num_params:.2f}M")

        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass"""
    print("\nTesting forward pass...")

    try:
        import torch
        from config import get_config
        from models import create_vlg_model

        config = get_config()
        model = create_vlg_model(config)
        model.eval()

        # Create dummy inputs
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 49408, (2, 77))

        # Forward pass
        with torch.no_grad():
            outputs = model(images, input_ids)

        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Output keys: {list(outputs.keys())}")

        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation"""
    print("\nTesting loss computation...")

    try:
        import torch
        from config import get_config
        from models import create_vlg_model

        config = get_config()
        model = create_vlg_model(config)

        # Create dummy inputs
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 49408, (2, 77))

        # Compute loss
        losses = model.compute_loss(images, input_ids)

        print(f"  ✓ Loss computation successful")
        print(f"  ✓ Total loss: {losses['total_loss'].item():.4f}")
        print(f"  ✓ Contrastive loss: {losses['contrastive_loss'].item():.4f}")

        return True
    except Exception as e:
        print(f"  ✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow properly"""
    print("\nTesting gradient flow...")

    try:
        import torch
        from config import get_config
        from models import create_vlg_model

        config = get_config()
        model = create_vlg_model(config)

        # Create dummy inputs
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 49408, (2, 77))

        # Compute loss and backprop
        losses = model.compute_loss(images, input_ids)
        losses['total_loss'].backward()

        # Check gradients
        has_grad = any(p.grad is not None for p in model.parameters())

        if has_grad:
            print(f"  ✓ Gradients flow correctly")
            return True
        else:
            print(f"  ✗ No gradients found")
            return False

    except Exception as e:
        print(f"  ✗ Gradient test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("VISION-LANGUAGE GROUNDING SYSTEM - INSTALLATION VERIFICATION")
    print("=" * 60)

    results = []

    # Run all checks
    results.append(("Package Imports", check_imports()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Loss Computation", test_loss_computation()))
    results.append(("Gradient Flow", test_gradient_flow()))

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All checks passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Run training: python train.py")
        print("  2. Run inference: python inference.py --checkpoint <path> --image <path>")
        print("  3. Run tests: pytest tests/ -v")
        print("  4. Check demo: jupyter notebook demo.ipynb")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check Python version (>=3.8 required)")
        print("  3. Verify CUDA installation if using GPU")
        return 1


if __name__ == "__main__":
    sys.exit(main())
