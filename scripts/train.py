"""
train.py - SONAR 2.0 Model Training Pipeline
=============================================
Train all models in the SONAR 2.0 archaeological detection system:
1. ResUNet Autoencoder (anomaly detection via reconstruction)
2. Isolation Forest (outlier detection in latent space)
3. K-Means Clustering (signature matching)
4. GATE Meta-learner (ensemble predictions)

Usage Examples:
    # Train all models sequentially
    python train.py --all
    
    # Train specific model
    python train.py --model autoencoder
    python train.py --model iforest
    python train.py --model kmeans
    python train.py --model gate
    
    # Train with custom parameters
    python train.py --model autoencoder --epochs 100 --batch-size 64
    python train.py --model iforest --contamination 0.05
    
    # Resume training
    python train.py --model autoencoder --resume

Author: SONAR 2.0 Team
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt

# Import configuration and utilities
from config import config
from utils import (
    ResUNetAutoencoder,
    ResUNetEncoder,
    load_patches,
    extract_embeddings,
    train_isolation_forest,
    save_model
)


# ==============================================================================
# DATASET CLASSES
# ==============================================================================

class PatchDataset(Dataset):
    """PyTorch Dataset for patch data"""
    
    def __init__(self, patches: np.ndarray, metadata: List[dict]):
        self.patches = torch.from_numpy(patches).float()
        self.metadata = metadata
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx], self.metadata[idx]


def load_training_data(train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, List, List]:
    """
    Load and split training data from all AOIs
    
    Returns:
        train_patches, val_patches, train_metadata, val_metadata
    """
    print(f"\n{'='*80}")
    print("LOADING TRAINING DATA")
    print(f"{'='*80}")
    
    # Find all patch files
    patch_files = sorted(config.PATCHES_DIR.glob("AOI_*_all_patches.npz"))
    
    if len(patch_files) == 0:
        raise FileNotFoundError(f"No patch files found in {config.PATCHES_DIR}")
    
    print(f"Found {len(patch_files)} AOI patch files")
    
    # Load all patches
    all_patches = []
    all_metadata = []
    
    for patch_file in tqdm(patch_files, desc="Loading patches"):
        patches, metadata = load_patches(patch_file)
        all_patches.append(patches)
        all_metadata.extend(metadata)
    
    all_patches = np.concatenate(all_patches, axis=0)
    
    print(f"\nDataset statistics:")
    print(f"  Total patches: {len(all_patches)}")
    print(f"  Patch shape: {all_patches.shape}")
    print(f"  Memory usage: {all_patches.nbytes / 1024 / 1024:.2f} MB")
    
    # Train/val split
    n_train = int(len(all_patches) * train_ratio)
    indices = np.random.permutation(len(all_patches))
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_patches = all_patches[train_indices]
    val_patches = all_patches[val_indices]
    train_metadata = [all_metadata[i] for i in train_indices]
    val_metadata = [all_metadata[i] for i in val_indices]
    
    print(f"\nSplit:")
    print(f"  Training: {len(train_patches)} patches ({len(train_patches)/len(all_patches)*100:.1f}%)")
    print(f"  Validation: {len(val_patches)} patches ({len(val_patches)/len(all_patches)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    return train_patches, val_patches, train_metadata, val_metadata


# ==============================================================================
# MODEL 1: AUTOENCODER TRAINING
# ==============================================================================

def train_autoencoder(args):
    """Train ResUNet Autoencoder"""
    
    print(f"\n{'='*80}")
    print("TRAINING MODEL 1: ResUNet AUTOENCODER")
    print(f"{'='*80}")
    
    # Load data
    train_patches, val_patches, train_meta, val_meta = load_training_data(train_ratio=0.8)
    
    # Create datasets
    train_dataset = PatchDataset(train_patches, train_meta)
    val_dataset = PatchDataset(val_patches, val_meta)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize model
    model = ResUNetAutoencoder(in_channels=7).to(config.DEVICE)
    
    # Resume if requested
    start_epoch = 0
    if args.resume and config.AUTOENCODER_PATH.exists():
        print(f"Resuming from {config.AUTOENCODER_PATH}")
        model.load_state_dict(torch.load(config.AUTOENCODER_PATH, map_location=config.DEVICE))
        start_epoch = args.epochs // 2  # Estimate
    
    print(f"\nModel: ResUNet Autoencoder")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training...\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for patches, _ in pbar:
            patches = patches.to(config.DEVICE)
            
            # Forward pass
            reconstruction, _ = model(patches)
            loss = criterion(reconstruction, patches)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for patches, _ in pbar:
                patches = patches.to(config.DEVICE)
                reconstruction, _ = model(patches)
                loss = criterion(reconstruction, patches)
                val_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.AUTOENCODER_PATH)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = config.MODELS_DIR / f'autoencoder_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path.name}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(config.MODELS_DIR / 'autoencoder_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print("AUTOENCODER TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Best model saved: {config.AUTOENCODER_PATH}")
    print(f"✓ Best validation loss: {best_val_loss:.6f}")
    print(f"✓ Training curves: {config.MODELS_DIR / 'autoencoder_training_curves.png'}")
    print(f"{'='*80}\n")


# ==============================================================================
# MODEL 2: ISOLATION FOREST TRAINING
# ==============================================================================

def train_iforest_model(args):
    """Train Isolation Forest on embeddings"""
    
    print(f"\n{'='*80}")
    print("TRAINING MODEL 2: ISOLATION FOREST")
    print(f"{'='*80}")
    
    # Check if autoencoder exists
    if not config.AUTOENCODER_PATH.exists():
        print(f"❌ Autoencoder not found: {config.AUTOENCODER_PATH}")
        print("   Train autoencoder first: python train.py --model autoencoder")
        return False
    
    # Load data
    train_patches, _, train_meta, _ = load_training_data(train_ratio=1.0)  # Use all data
    
    # Load encoder
    print("Loading encoder...")
    encoder = ResUNetEncoder(in_channels=7, embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)
    encoder.load_from_autoencoder(str(config.AUTOENCODER_PATH))
    encoder.eval()
    print("✓ Encoder loaded\n")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(encoder, train_patches, config.DEVICE, batch_size=args.batch_size)
    print(f"✓ Extracted embeddings: {embeddings.shape}\n")
    
    # Train Isolation Forest
    iforest, scaler = train_isolation_forest(
        embeddings=embeddings,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples='auto',
        random_state=42
    )
    
    # Save model
    save_model(iforest, scaler, str(config.IFOREST_PATH))
    
    print(f"\n{'='*80}")
    print("ISOLATION FOREST TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Model saved: {config.IFOREST_PATH}")
    print(f"✓ Contamination: {args.contamination}")
    print(f"✓ Trees: {args.n_estimators}")
    print(f"{'='*80}\n")
    
    return True


# ==============================================================================
# MODEL 3: K-MEANS TRAINING
# ==============================================================================

def train_kmeans_model(args):
    """Train K-Means clustering on embeddings"""
    
    print(f"\n{'='*80}")
    print("TRAINING MODEL 3: K-MEANS CLUSTERING")
    print(f"{'='*80}")
    
    # Check if autoencoder exists
    if not config.AUTOENCODER_PATH.exists():
        print(f"❌ Autoencoder not found: {config.AUTOENCODER_PATH}")
        print("   Train autoencoder first: python train.py --model autoencoder")
        return False
    
    # Load data (use only important/archaeological patches if available)
    train_patches, _, train_meta, _ = load_training_data(train_ratio=1.0)
    
    # Load encoder
    print("Loading encoder...")
    encoder = ResUNetEncoder(in_channels=7, embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)
    encoder.load_from_autoencoder(str(config.AUTOENCODER_PATH))
    encoder.eval()
    print("✓ Encoder loaded\n")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(encoder, train_patches, config.DEVICE, batch_size=args.batch_size)
    print(f"✓ Extracted embeddings: {embeddings.shape}\n")
    
    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Train K-Means
    print(f"Training K-Means with {args.n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42,
        verbose=1
    )
    
    kmeans.fit(embeddings_scaled)
    
    # Analyze clusters
    labels = kmeans.labels_
    cluster_sizes = np.bincount(labels)
    
    print(f"\n✓ K-Means training complete")
    print(f"\nCluster distribution:")
    for i, size in enumerate(cluster_sizes):
        print(f"  Cluster {i}: {size} patches ({size/len(labels)*100:.1f}%)")
    
    # Save model
    model_data = {
        'kmeans': kmeans,
        'scaler': scaler,
        'n_clusters': args.n_clusters,
        'cluster_sizes': cluster_sizes.tolist()
    }
    
    with open(config.KMEANS_PATH, 'wb') as f:
        joblib.dump(model_data, f)
    
    print(f"\n{'='*80}")
    print("K-MEANS TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Model saved: {config.KMEANS_PATH}")
    print(f"✓ Clusters: {args.n_clusters}")
    print(f"{'='*80}\n")
    
    return True


# ==============================================================================
# MODEL 4: GATE META-LEARNER TRAINING
# ==============================================================================

def train_gate_model(args):
    """Train GATE meta-learner (requires all other models)"""
    
    print(f"\n{'='*80}")
    print("TRAINING MODEL 4: GATE META-LEARNER")
    print(f"{'='*80}")
    
    # Check dependencies
    required_models = [
        config.AUTOENCODER_PATH,
        config.IFOREST_PATH,
        config.KMEANS_PATH,
        config.ARCH_EMBEDDINGS_CSV
    ]
    
    for model_path in required_models:
        if not model_path.exists():
            print(f"❌ Required model/data not found: {model_path}")
            print("   Train prerequisite models first")
            return False
    
    print("✓ All prerequisite models found\n")
    
    # TODO: This requires labeled data (archaeological vs non-archaeological)
    # For now, this is a placeholder showing the structure
    
    print("⚠️  GATE training requires labeled archaeological data")
    print("    Please prepare a labeled dataset with columns:")
    print("    - prob_autoencoder")
    print("    - prob_iforest") 
    print("    - prob_kmeans")
    print("    - arch_similarity")
    print("    - label (0=normal, 1=archaeological)")
    print()
    print("    Then uncomment and adapt the training code below.")
    
    # PLACEHOLDER TRAINING CODE (uncomment and adapt when you have labeled data)
    """
    # Load labeled training data
    X_train = np.load('path/to/labeled_features.npy')  # Shape: (N, 4)
    y_train = np.load('path/to/labels.npy')  # Shape: (N,)
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train GATE MLP
    gate_model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        verbose=True
    )
    
    gate_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = gate_model.predict(X_val_scaled)
    print("\nValidation Results:")
    print(classification_report(y_val, y_pred, target_names=['Normal', 'Archaeological']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    # Save model
    joblib.dump(gate_model, config.GATE_MODEL_PKL)
    joblib.dump(scaler, config.GATE_SCALER_PATH)
    
    print(f"\n✓ GATE model saved: {config.GATE_MODEL_PKL}")
    print(f"✓ Scaler saved: {config.GATE_SCALER_PATH}")
    """
    
    print(f"\n{'='*80}")
    print("GATE TRAINING PLACEHOLDER")
    print(f"{'='*80}\n")
    
    return True


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SONAR 2.0 Model Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        choices=['autoencoder', 'iforest', 'kmeans', 'gate', 'all'],
        help='Which model to train (use "all" to train sequentially)'
    )
    
    parser.add_argument('--all', action='store_true', help='Train all models sequentially')
    
    # Autoencoder args
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    
    # IForest args
    parser.add_argument('--contamination', type=float, default=0.05, help='IForest contamination')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees')
    
    # K-Means args
    parser.add_argument('--n-clusters', type=int, default=10, help='Number of clusters')
    
    args = parser.parse_args()
    
    # Determine what to train
    if args.all or args.model == 'all':
        models_to_train = ['autoencoder', 'iforest', 'kmeans', 'gate']
    elif args.model:
        models_to_train = [args.model]
    else:
        parser.print_help()
        sys.exit(1)
    
    # Print header
    print(f"\n{'='*80}")
    print("SONAR 2.0 - MODEL TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Device: {config.DEVICE}")
    print(f"Models to train: {', '.join(models_to_train)}")
    print(f"Output directory: {config.MODELS_DIR}")
    print(f"{'='*80}\n")
    
    # Create output directory
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    config.GATE_DIR.mkdir(exist_ok=True, parents=True)
    
    # Train models
    results = {}
    
    for model_name in models_to_train:
        try:
            if model_name == 'autoencoder':
                train_autoencoder(args)
                results['autoencoder'] = 'success'
            
            elif model_name == 'iforest':
                success = train_iforest_model(args)
                results['iforest'] = 'success' if success else 'failed'
            
            elif model_name == 'kmeans':
                success = train_kmeans_model(args)
                results['kmeans'] = 'success' if success else 'failed'
            
            elif model_name == 'gate':
                success = train_gate_model(args)
                results['gate'] = 'success' if success else 'failed'
        
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = 'error'
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'='*80}")
    
    for model_name, status in results.items():
        icon = '✅' if status == 'success' else '❌'
        print(f"{icon} {model_name.capitalize()}: {status}")
    
    print(f"\n📁 Models saved to: {config.MODELS_DIR}")
    print(f"{'='*80}\n")
    
    # Next steps
    if all(status == 'success' for status in results.values()):
        print("✅ All models trained successfully!")
        print("\n💡 Next steps:")
        print("   1. Generate predictions: python main.py --mode predict")
        print("   2. Visualize results: python gradio_app.py")
    else:
        print("⚠️  Some models failed to train. Check logs above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)