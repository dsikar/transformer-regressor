"""
Training script for the Vision Transformer steering angle regressor.
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Local imports
from model_implementation import SteeringAngleTransformer
from dataset_utils import create_data_loaders, visualize_sample_images


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Vision Transformer for steering angle prediction')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--img_size', type=int, nargs=2, default=[480, 640],
                        help='Input image size (height, width)')
    
    # Model parameters
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Patch size for Vision Transformer')
    parser.add_argument('--dim', type=int, default=384,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=6,
                        help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=1536,
                        help='MLP hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of data to use for testing')
    
    # Optimization parameters
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--grad_accumulation', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (small subset of data)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion, device, scaler=None, 
                grad_accumulation=1, max_grad_norm=1.0, mixed_precision=False):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()
    
    for i, (images, targets) in enumerate(tqdm(loader, desc="Training")):
        images = images.to(device)
        targets = targets.to(device)
        
        # Mixed precision training
        if mixed_precision and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss = loss / grad_accumulation  # Normalize loss for gradient accumulation
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % grad_accumulation == 0 or (i + 1) == len(loader):
                # Gradient clipping
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard training
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / grad_accumulation  # Normalize loss for gradient accumulation
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (i + 1) % grad_accumulation == 0 or (i + 1) == len(loader):
                # Gradient clipping
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
        
        # Collect statistics
        epoch_loss += loss.item() * grad_accumulation
        all_preds.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    # Calculate metrics
    epoch_loss /= len(loader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    epoch_mse = np.mean((all_preds - all_targets) ** 2)
    epoch_mae = np.mean(np.abs(all_preds - all_targets))
    
    return {
        'loss': epoch_loss,
        'mse': epoch_mse,
        'mae': epoch_mae,
        'predictions': all_preds,
        'targets': all_targets
    }


def validate(model, loader, criterion, device, mixed_precision=False):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            if mixed_precision:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate metrics
    val_loss /= len(loader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    val_mse = np.mean((all_preds - all_targets) ** 2)
    val_mae = np.mean(np.abs(all_preds - all_targets))
    
    return {
        'loss': val_loss,
        'mse': val_mse,
        'mae': val_mae,
        'predictions': all_preds,
        'targets': all_targets
    }


def plot_training_curves(train_metrics, val_metrics, save_path):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(train_metrics['loss'], label='Train')
    plt.plot(val_metrics['loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot MSE
    plt.subplot(1, 3, 2)
    plt.plot(train_metrics['mse'], label='Train')
    plt.plot(val_metrics['mse'], label='Validation')
    plt.title('MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot MAE
    plt.subplot(1, 3, 3)
    plt.plot(train_metrics['mae'], label='Train')
    plt.plot(val_metrics['mae'], label='Validation')
    plt.title('MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictions(predictions, targets, epoch, save_dir):
    """Plot predicted vs actual steering angles."""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of predictions vs targets
    plt.scatter(targets, predictions, alpha=0.5, s=10)
    
    # Plot y=x line
    min_val = min(np.min(predictions), np.min(targets))
    max_val = max(np.max(predictions), np.max(targets))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Predicted vs Actual Steering Angles (Epoch {epoch})')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    plt.savefig(os.path.join(save_dir, f'predictions_epoch_{epoch}.png'))
    plt.close()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Print GPU info if available
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        img_size=tuple(args.img_size),
        seed=args.seed
    )
    
    # Visualize sample images
    visualize_sample_images(
        train_loader,
        num_samples=5,
        save_path=os.path.join(args.output_dir, 'sample_images.png')
    )
    
    # Create model
    model = SteeringAngleTransformer(
        image_size=tuple(args.img_size),
        patch_size=args.patch_size,
        num_classes=1,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout
    )
    
    # Print model summary
    print(model)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if args.mixed_precision else None
    
    # Initialize metrics tracking
    train_metrics = {'loss': [], 'mse': [], 'mae': []}
    val_metrics = {'loss': [], 'mse': [], 'mae': []}
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        start_time = time.time()
        train_results = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            grad_accumulation=args.grad_accumulation,
            max_grad_norm=args.max_grad_norm,
            mixed_precision=args.mixed_precision
        )
        train_time = time.time() - start_time
        
        # Validate
        val_results = validate(
            model, val_loader, criterion, device,
            mixed_precision=args.mixed_precision
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        for metric in ['loss', 'mse', 'mae']:
            train_metrics[metric].append(train_results[metric])
            val_metrics[metric].append(val_results[metric])
        
        # Print epoch summary
        print(f"Time: {train_time:.2f}s | "
              f"Train Loss: {train_results['loss']:.6f} | "
              f"Train MSE: {train_results['mse']:.6f} | "
              f"Train MAE: {train_results['mae']:.6f} | "
              f"Val Loss: {val_results['loss']:.6f} | "
              f"Val MSE: {val_results['mse']:.6f} | "
              f"Val MAE: {val_results['mae']:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save model if it's the best so far
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'val_results': val_results
            }, os.path.join(args.output_dir, 'best_model_640x480_segmented.pth'))
            
            # Plot predictions for best model
            plot_predictions(
                val_results['predictions'].flatten(),
                val_results['targets'].flatten(),
                epoch + 1,
                args.output_dir
            )
        else:
            patience_counter += 1
            
        # Plot training curves for each epoch
        plot_training_curves(
            train_metrics,
            val_metrics,
            os.path.join(args.output_dir, 'training_curves.png')
        )
        
        # Check for early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
            
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Load best model for final evaluation
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_640x480_segmented.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print(f"\nBest model from epoch {best_epoch+1} with validation loss {best_val_loss:.6f}")
    
    # Final evaluation on test set
    test_results = validate(
        model, test_loader, criterion, device,
        mixed_precision=args.mixed_precision
    )
    
    print(f"Test MSE: {test_results['mse']:.6f} | Test MAE: {test_results['mae']:.6f}")
    
    # Save test predictions
    plot_predictions(
        test_results['predictions'].flatten(),
        test_results['targets'].flatten(),
        'final_test',
        args.output_dir
    )
    
    # Save final test results
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'mse': float(test_results['mse']),
            'mae': float(test_results['mae']),
            'loss': float(test_results['loss'])
        }, f, indent=2)
    
    # Export ONNX model for inference
    dummy_input = torch.randn(1, 3, args.img_size[0], args.img_size[1], device=device)
    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(args.output_dir, 'model.onnx'),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model to {os.path.join(args.output_dir, 'model.onnx')}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
