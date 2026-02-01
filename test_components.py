import torch
import sys

print("="*60)
print("Testing Lab 2 Components")
print("="*60)

print("\n[Test 1] Testing model.py...")
try:
    from model import get_model, count_parameters
    model = get_model(num_classes=10)
    total_params, trainable_params = count_parameters(model)
    print(f"✓ Model created successfully")
    print(f"  Parameters: {total_params:,}")
except Exception as e:
    print(f"✗ Model test failed: {e}")
    sys.exit(1)

# Test 2: DataLoader
print("\n[Test 2] Testing dataloader.py...")
try:
    from dataloader import get_cifar10_dataloaders
    # Small test to avoid downloading full dataset if not needed
    print("✓ DataLoader imported successfully")
    print("  Note: Full data loading will occur during training")
except Exception as e:
    print(f"✗ DataLoader test failed: {e}")
    sys.exit(1)

# Test 3: FLOPs Counter
print("\n[Test 3] Testing flops_counter.py...")
try:
    from flops_counter import count_model_flops, print_flops_report
    device = 'cpu'
    total_flops, layer_flops = count_model_flops(
        model,
        input_shape=(1, 3, 32, 32),
        device=device
    )
    flops_info = print_flops_report(total_flops, layer_flops)
    print(f"✓ FLOPs counter working correctly")
except Exception as e:
    print(f"✗ FLOPs counter test failed: {e}")
    sys.exit(1)

# Test 4: Visualization
print("\n[Test 4] Testing visualization.py...")
try:
    from visualization import GradientFlowTracker, WeightUpdateTracker
    grad_tracker = GradientFlowTracker()
    weight_tracker = WeightUpdateTracker(model)
    
    # Simulate a forward-backward pass
    dummy_input = torch.randn(2, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (2,))
    
    model.train()
    output = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_target)
    loss.backward()
    
    # Track gradients
    grad_stats = grad_tracker.track_gradients(model, step=0)
    weight_tracker.track_weights(model, step=0)
    
    print(f"✓ Visualization trackers working correctly")
    print(f"  Tracked {len(grad_stats['layers'])} layers")
except Exception as e:
    print(f"✗ Visualization test failed: {e}")
    sys.exit(1)

# Test 5: Training Script Import
print("\n[Test 5] Testing train.py imports...")
try:
    # Check if train.py has correct imports
    import train
    print(f"✓ Training script imports successfully")
except Exception as e:
    print(f"✗ Training script test failed: {e}")
    sys.exit(1)

# Test 6: Device Check
print("\n[Test 6] Checking available devices...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"⚠ CUDA not available - training will use CPU")
    print(f"  Note: CPU training will be significantly slower")

# Summary
print("\n" + "="*60)
print("Component Test Summary")
print("="*60)
print("✓ All components tested successfully!")
print("\nYou can now run training with:")
print("  python train.py")
print("\nOr use the quick start script:")
print("  ./run_training.sh")
print("="*60 + "\n")
