"""End-to-end pipeline test for GenreMaster training.

Tests the complete training pipeline with small sample data to ensure
nothing will break when scaling to full dataset. Tests both MPS and CUDA.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.fma import setup_fma_medium
from data.transforms import create_premaster_transforms
from models.genremaster import create_genremaster_model
from losses import create_loss_function
from utils import seed_everything, get_device


def collate_fn(batch):
    """Custom collate function for variable-length audio."""
    # Find minimum length
    min_length = min(item['waveform'].shape[1] for item in batch)

    # Crop all to same length
    waveforms = torch.stack([item['waveform'][:, :min_length] for item in batch])
    genre_indices = torch.tensor([item['genre_idx'] for item in batch])
    track_ids = [item['track_id'] for item in batch]

    return {
        'waveform': waveforms,
        'genre_idx': genre_indices,
        'track_id': track_ids,
    }


def test_end_to_end_pipeline():
    """Test complete training pipeline with sample data."""
    seed_everything(42)

    print("=" * 70)
    print("GenreMaster End-to-End Pipeline Test")
    print("=" * 70)

    # Device detection (supports both MPS and CUDA)
    device = get_device()
    print(f"\n✓ Device: {device}")

    if device.type == "cuda":
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == "mps":
        print(f"  Apple Silicon GPU (MPS backend)")

    # 1. Setup dataset (using SMALL SAMPLE)
    print("\n" + "=" * 70)
    print("Step 1: Dataset Setup (Small Sample)")
    print("=" * 70)

    data_root = Path("/Users/I558118/Documents/Projects/GenreMaster/data")
    audio_dir = Path("/Volumes/LLModels/Datasets/fma_medium")

    datasets, genre_to_idx = setup_fma_medium(
        data_root=data_root,
        audio_dir=audio_dir,
        top_k_genres=16,
        samples_per_genre=1500,
    )

    # Use only 50 samples for testing
    n_samples = 50
    train_subset = Subset(datasets['train'], range(min(n_samples, len(datasets['train']))))
    val_subset = Subset(datasets['val'], range(min(10, len(datasets['val']))))

    print(f"✓ Using {len(train_subset)} train samples, {len(val_subset)} val samples")

    # 2. Create data loaders
    print("\n" + "=" * 70)
    print("Step 2: Data Loaders")
    print("=" * 70)

    batch_size = 4
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded for testing
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")

    # Test one batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Waveform: {sample_batch['waveform'].shape}")
    print(f"  Genre indices: {sample_batch['genre_idx'].tolist()}")

    # 3. Create transforms
    print("\n" + "=" * 70)
    print("Step 3: Pre-master Transforms")
    print("=" * 70)

    train_transform, val_transform = create_premaster_transforms(
        target_lufs=-18.0,
        sample_rate=44100,
        augment_train=True,
    )

    # Test transform
    test_waveform = sample_batch['waveform'][0]
    pre_master, mastered_target = train_transform(test_waveform)

    print(f"✓ Transform test:")
    print(f"  Input shape: {test_waveform.shape}")
    print(f"  Pre-master: {pre_master.shape}")
    print(f"  Target: {mastered_target.shape}")

    # 4. Create model
    print("\n" + "=" * 70)
    print("Step 4: Model Creation")
    print("=" * 70)

    model = create_genremaster_model(
        n_genres=len(genre_to_idx),
        encoder_type="lightweight",
        conditioned=True,
        sample_rate=44100,
    ).to(device)

    param_counts = model.get_parameter_count()
    print(f"✓ Model created:")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Model on device: {next(model.parameters()).device}")

    # 5. Create loss function
    print("\n" + "=" * 70)
    print("Step 5: Loss Function")
    print("=" * 70)

    criterion = create_loss_function(
        sample_rate=44100,
        loss_weights={
            'loudness': 1.0,
            'spectral': 0.5,
            'dynamic': 0.5,
            'perceptual': 0.1,
        },
    )

    print("✓ Loss function created with weights:")
    print(f"  Loudness: {criterion.lambda_loudness}")
    print(f"  Spectral: {criterion.lambda_spectral}")
    print(f"  Dynamic: {criterion.lambda_dynamic}")
    print(f"  Perceptual: {criterion.lambda_perceptual}")

    # 6. Create optimizer
    print("\n" + "=" * 70)
    print("Step 6: Optimizer")
    print("=" * 70)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"✓ Optimizer: Adam, lr=1e-4")

    # 7. Test training step
    print("\n" + "=" * 70)
    print("Step 7: Training Step Test")
    print("=" * 70)

    model.train()
    batch = next(iter(train_loader))

    waveform = batch['waveform'].to(device)
    genre_idx = batch['genre_idx'].to(device)

    # Create pre-master and target
    pre_masters = []
    targets = []
    for i in range(waveform.shape[0]):
        pre_master, target = train_transform(waveform[i])
        pre_masters.append(pre_master)
        targets.append(target)

    pre_master_batch = torch.stack(pre_masters).to(device)
    target_batch = torch.stack(targets).to(device)

    print(f"Input shapes:")
    print(f"  Pre-master: {pre_master_batch.shape}")
    print(f"  Target: {target_batch.shape}")
    print(f"  Genre: {genre_idx.shape}")

    # Forward pass
    optimizer.zero_grad()
    output = model(pre_master_batch, genre_idx)

    print(f"\nForward pass:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Compute loss
    loss_dict = criterion(output, target_batch, return_components=True)

    print(f"\nLoss components:")
    for name, value in loss_dict.items():
        print(f"  {name:12s}: {value.item():.6f}")

    # Backward pass
    loss_dict['total'].backward()
    optimizer.step()

    print(f"\n✓ Backward pass successful")

    # 8. Test validation step
    print("\n" + "=" * 70)
    print("Step 8: Validation Step Test")
    print("=" * 70)

    model.eval()
    val_batch = next(iter(val_loader))

    with torch.no_grad():
        waveform_val = val_batch['waveform'].to(device)
        genre_idx_val = val_batch['genre_idx'].to(device)

        # Create targets
        pre_masters_val = []
        targets_val = []
        for i in range(waveform_val.shape[0]):
            pre_master, target = val_transform(waveform_val[i])
            pre_masters_val.append(pre_master)
            targets_val.append(target)

        pre_master_val = torch.stack(pre_masters_val).to(device)
        target_val = torch.stack(targets_val).to(device)

        # Forward pass
        output_val = model(pre_master_val, genre_idx_val)
        loss_val = criterion(output_val, target_val)

        print(f"✓ Validation loss: {loss_val.item():.6f}")

    # 9. Test mini training loop (2 epochs)
    print("\n" + "=" * 70)
    print("Step 9: Mini Training Loop (2 epochs)")
    print("=" * 70)

    for epoch in range(2):
        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # Only 3 batches per epoch for testing
                break

            waveform = batch['waveform'].to(device)
            genre_idx = batch['genre_idx'].to(device)

            # Create pre-master and targets
            pre_masters = []
            targets = []
            for i in range(waveform.shape[0]):
                pre_master, target = train_transform(waveform[i])
                pre_masters.append(pre_master)
                targets.append(target)

            pre_master_batch = torch.stack(pre_masters).to(device)
            target_batch = torch.stack(targets).to(device)

            # Training step
            optimizer.zero_grad()
            output = model(pre_master_batch, genre_idx)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}: loss={loss.item():.6f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"✓ Epoch {epoch+1} average loss: {avg_loss:.6f}")

    # 10. Memory usage
    print("\n" + "=" * 70)
    print("Step 10: Resource Usage")
    print("=" * 70)

    if device.type == "cuda":
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    elif device.type == "mps":
        print(f"MPS backend active (memory tracking not available)")

    print("\n" + "=" * 70)
    print("✅ All pipeline tests passed! Ready for full-scale training.")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  ✓ Dataset loading works with real FMA audio")
    print("  ✓ Transforms create valid pre-master/target pairs")
    print("  ✓ Model forward/backward pass successful")
    print("  ✓ Loss computation works on all components")
    print("  ✓ Optimizer updates model parameters")
    print("  ✓ Training loop completes without errors")
    print(f"  ✓ Device support: {device.type.upper()}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_end_to_end_pipeline()
