"""
GPU Training Startup Script for 8GB VRAM
- Forces CUDA usage (no CPU fallback)
- Memory optimized for RTX 4060
- Automatic GPU verification
- Real-time monitoring
"""

import torch
import os
import sys
from pathlib import Path

print("\n" + "="*70)
print("GPU TRAINING STARTUP - 8GB VRAM OPTIMIZED")
print("="*70 + "\n")

# ============================================================================
# STEP 1: VERIFY GPU AVAILABILITY (CRITICAL)
# ============================================================================
print("[STEP 1] VERIFYING GPU SETUP...")
print("-" * 70)

if not torch.cuda.is_available():
    print("[FATAL ERROR] CUDA not available!")
    print("[ACTION REQUIRED] Please:")
    print("  1. Install NVIDIA drivers from: nvidia.com/drivers")
    print("  2. Reinstall PyTorch with CUDA:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("  3. Verify with: python -c \"import torch; print(torch.cuda.is_available())\"")
    sys.exit(1)

print("[OK] CUDA is available")

# Get GPU info
device_name = torch.cuda.get_device_name(0)
total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"[OK] GPU Device: {device_name}")
print(f"[OK] Total Memory: {total_memory:.2f} GB")

# Check if sufficient memory
if total_memory < 6:
    print(f"[WARNING] Only {total_memory:.2f}GB available. Model may not fit without optimizations.")
else:
    print(f"[OK] Sufficient memory for model training")

print()

# ============================================================================
# STEP 2: FORCE GPU DEVICE
# ============================================================================
print("[STEP 2] SETTING GPU DEVICE...")
print("-" * 70)

# Set environment variable to use only GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Force CUDA as default device
device = torch.device('cuda:0')
torch.cuda.set_device(0)

print(f"[OK] Primary device: {device}")
print(f"[OK] GPU count: {torch.cuda.device_count()}")

print()

# ============================================================================
# STEP 3: OPTIMIZE FOR 8GB
# ============================================================================
print("[STEP 3] MEMORY OPTIMIZATION...")
print("-" * 70)

# Clear any cached memory
torch.cuda.empty_cache()
print("[OK] GPU cache cleared")

# Enable memory efficient operations
torch.set_float32_matmul_precision('high')
print("[OK] Float32 matmul precision set to HIGH")

# Get current memory status
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"[OK] Memory allocated: {allocated:.2f} GB")
print(f"[OK] Memory reserved: {reserved:.2f} GB")

print()

# ============================================================================
# STEP 4: VERIFY GPU COMPUTATION
# ============================================================================
print("[STEP 4] TESTING GPU COMPUTATION...")
print("-" * 70)

try:
    # Test tensor operation
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = torch.mm(x, y)
    
    print(f"[OK] GPU computation test passed")
    print(f"[OK] Test tensor device: {z.device}")
    
except Exception as e:
    print(f"[ERROR] GPU computation failed: {e}")
    sys.exit(1)

print()

# ============================================================================
# STEP 5: LOAD TRAINING MODULE
# ============================================================================
print("[STEP 5] LOADING TRAINING MODULE...")
print("-" * 70)

try:
    from train_ht_hgnn_safe_v2 import SafeHT_HGNN_Trainer
    print("[OK] Training module loaded")
except ImportError as e:
    print(f"[ERROR] Could not import training module: {e}")
    print("[ACTION] Make sure train_ht_hgnn_safe_v2.py is in the current directory")
    sys.exit(1)

print()

# ============================================================================
# STEP 6: INITIALIZE TRAINER
# ============================================================================
print("[STEP 6] INITIALIZING MODEL...")
print("-" * 70)

try:
    trainer = SafeHT_HGNN_Trainer(
        in_channels=18,
        hidden_channels=64,           # Optimized for 8GB
        out_channels=32,
        num_nodes=1206,
        num_hyperedges=36,
        learning_rate=0.001,
        weight_decay=1e-5,
        use_amp=True,                 # CRITICAL: Mixed precision for 8GB
        checkpoint_dir='outputs/checkpoints'
    )
    print("[OK] Trainer initialized successfully")
    
    # Verify model is on GPU
    model_device = next(trainer.model.parameters()).device
    print(f"[OK] Model device: {model_device}")
    
    if model_device.type != 'cuda':
        print(f"[ERROR] Model is on {model_device}, should be CUDA!")
        sys.exit(1)
    
except Exception as e:
    print(f"[ERROR] Trainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# STEP 7: PREPARE DATA
# ============================================================================
print("[STEP 7] PREPARING DATA...")
print("-" * 70)

try:
    data = trainer.prepare_data()
    print("[OK] Data prepared successfully")
    
    # Verify data is on GPU
    print(f"[OK] Node features device: {data['X'].device}")
    print(f"[OK] Incidence matrix device: {data['incidence_matrix'].device}")
    
    if data['X'].device.type != 'cuda':
        print("[ERROR] Data is on CPU, should be CUDA!")
        sys.exit(1)
    
except Exception as e:
    print(f"[ERROR] Data preparation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# STEP 8: PRE-TRAINING CHECKS
# ============================================================================
print("[STEP 8] PRE-TRAINING CHECKS...")
print("-" * 70)

# Check checkpoint directory
checkpoint_dir = Path('outputs/checkpoints')
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"[OK] Checkpoint directory: {checkpoint_dir}")

# Verify training.log is writable
try:
    with open('training.log', 'a') as f:
        f.write("\n" + "="*70 + "\n")
        f.write("GPU TRAINING SESSION STARTED\n")
        f.write(f"Device: {device_name}\n")
        f.write(f"Memory: {total_memory:.2f} GB\n")
        f.write("="*70 + "\n")
    print("[OK] Training log is writable")
except Exception as e:
    print(f"[WARNING] Could not write to training.log: {e}")

# Final GPU status
torch.cuda.synchronize()  # Ensure all GPU operations complete
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
free = (total_memory - reserved)
print(f"[OK] GPU Memory Status:")
print(f"     Allocated: {allocated:.2f} GB")
print(f"     Reserved:  {reserved:.2f} GB")
print(f"     Free:      {free:.2f} GB")
print(f"     Utilization: {(reserved/total_memory)*100:.1f}%")

if reserved > total_memory * 0.8:
    print(f"[WARNING] GPU usage already high before training!")

print()

# ============================================================================
# STEP 9: START TRAINING
# ============================================================================
print("[STEP 9] STARTING TRAINING...")
print("-" * 70)
print()

try:
    history = trainer.train(
        data,
        epochs=50,
        verbose=True,
        save_interval=10
    )
    print("\n[SUCCESS] Training completed!")
    
except KeyboardInterrupt:
    print("\n[INFO] Training interrupted by user")
    print("[INFO] Checkpoint has been saved")
    sys.exit(0)
    
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================================================
# STEP 10: POST-TRAINING SUMMARY
# ============================================================================
print("="*70)
print("TRAINING SUMMARY")
print("="*70)

print(f"\nInitial Loss: {history['loss'][0]:.2f}")
print(f"Final Loss:   {history['loss'][-1]:.2f}")
improvement = (history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100
print(f"Improvement:  {improvement:.1f}%")

print(f"\nCheckpoints saved to: outputs/checkpoints/")
print(f"  - Best model: best.pt")
print(f"  - Latest: latest.pt")

print(f"\nTraining logs: training.log")
print(f"History data: outputs/training_history.json")

print()
print("="*70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*70)
print()

# Final GPU status
torch.cuda.synchronize()
final_allocated = torch.cuda.memory_allocated() / 1e9
final_reserved = torch.cuda.memory_reserved() / 1e9
print(f"Final GPU Memory:")
print(f"  Allocated: {final_allocated:.2f} GB")
print(f"  Reserved:  {final_reserved:.2f} GB")
print()
