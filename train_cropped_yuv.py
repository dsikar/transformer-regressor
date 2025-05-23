"""
Training script for the Vision Transformer steering angle regressor with image preprocessing and YUV conversion.
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
from torchvision import transforms
import cv2

# Local imports
from model_implementation import SteeringAngleTransformer
from dataset_utils import create_data_loaders, visualize_sample_images


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Vision Transformer for steering angle prediction')
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default="/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_01",
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='/home/daniel/git/neurips-2025/scripts/vit-models',
                        help='Directory to save outputs')
    parser.add_argument('--img_size', type=int, nargs=2, default=[480, 640],
                        help='Output image size after preprocessing (height, width)')
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
    parser.add_argument('--epochs', type=int, default=100,
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
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode (small subset of data and save processed images)')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define preprocessing transform with YUV conversion
def preprocess_image(img, debug=False, debug_dir=None, idx=0):

    # Initial crop
    img = img[:, 210:480, :]  # Crop to 270x640 (H×W)
    # debug img[:, 259, 94] tensor([0.9961, 0.0000, 0.0000]) red fiducial, img is RGB   
    # Convert tensor to numpy for OpenCV processing (HWC format)
    img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]

    # Scale to [0, 255] and convert to uint8 for OpenCV
    img_np = (img_np * 255).astype(np.uint8)
    # print(img_np[269, 95,:]) [254   0   0] # fiducial is red
    # Convert RGB to YUV
    # invert R and B channels to get fiducial in blue
    # STOPPED HERE, align CNN with ViT preprocessing - YUV fiducials need to be RED
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    # resize
    img_resized = cv2.resize(img_yuv, (640, 480))  # Resize to 480x640
    img_resized_normalized = img_resized / 255.0  # Normalize to [0, 1]

    # print(img_yuv[269, 95,:]) [ 76  91 255] # fiducial is blue
    if debug:
        print(f"After RGB to YUV (H, W, C): {img_yuv.shape}")
        
        # Save YUV conversion (already in [0, 255])
        cv2.imwrite(os.path.join(debug_dir, f'debug_yuv_{idx}.png'), img_yuv)
        # red on disk
    #img_pil = transforms.ToPILImage()(img_yuv)
    
    #if debug:
        # Save PIL image
    #    img_pil.save(os.path.join(debug_dir, f'debug_pil_{idx}.png'))

    # Resize to original dimensions (480x640) - corrected order
    # img_resized = transforms.Resize((480, 640))(img_pil)
    # img_resized = transforms.Resize((480, 640))(img_yuv)

    if debug:
        # Save resized image
        img_resized.save(os.path.join(debug_dir, f'debug_resized_{idx}.png'))
    
    # Convert back to tensor - print(img_resized_normalized[479, 95,:]) [0.29803922 0.35686275 1.        ] - Fiducial is blue in YUV space
    img_tensor = transforms.ToTensor()(img_resized_normalized)
    
    if debug:
        print(f"After resize and to tensor (C, H, W): {img_tensor.shape}")
        
        # Save final tensor
        final_img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f'debug_final_{idx}.png'), 
                   cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    # print(img_tensor[:,479, 95]) tensor([0.2980, 0.3569, 1.0000]), fiducial is blue in YUV space
    return img_tensor

# Image is already in YUV space at this point ~ loader ?
def train_epoch(model, loader, optimizer, criterion, device, scaler=None, 
                grad_accumulation=1, max_grad_norm=1.0, mixed_precision=False, 
                debug=False, debug_dir=None):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()
    
    for i, (images, targets) in enumerate(tqdm(loader, desc="Training")):

        if i == 0:
            raw_img = images[0].clone()  # Clone to avoid modifying original
            
            # # Save the raw image before any processing
            if debug and debug_dir:
                raw_img_np = (raw_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, 'raw_image_sanity_check.png'), 
                           cv2.cvtColor(raw_img_np, cv2.COLOR_RGB2BGR))
                
                # Also save a cropped version for comparison
                cropped_raw = raw_img[:, 210:480, :]
                cropped_np = (cropped_raw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, 'cropped_raw_sanity_check.png'), 
                           cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR))
                
                # Print stats to help detect issues
                print(f"Raw image shape: {raw_img.shape}")
                print(f"Raw image min, max, mean: {raw_img.min():.3f}, {raw_img.max():.3f}, {raw_img.mean():.3f}")
                # print(f"Raw image has yellow borders: {is_yellow_present(raw_img)}")
                # import sys
                # sys.exit(0)  # Exit after first batch for debugging
               


        # Apply preprocessing with YUV conversion
        images = torch.stack([preprocess_image(img, debug=debug, debug_dir=debug_dir, idx=f"train_{i}_{j}") 
                            for j, img in enumerate(images)])
        images = images.to(device)
        targets = targets.to(device)
        
        if mixed_precision and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss = loss / grad_accumulation
                
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accumulation == 0 or (i + 1) == len(loader):
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / grad_accumulation
            
            loss.backward()
            
            if (i + 1) % grad_accumulation == 0 or (i + 1) == len(loader):
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
        
        epoch_loss += loss.item() * grad_accumulation
        all_preds.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
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


def validate(model, loader, criterion, device, mixed_precision=False, 
             debug=False, debug_dir=None):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(loader, desc="Validation")):
            images = torch.stack([preprocess_image(img, debug=debug, debug_dir=debug_dir, idx=f"val_{i}_{j}") 
                                for j, img in enumerate(images)])
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
    
    plt.subplot(1, 3, 1)
    plt.plot(train_metrics['loss'], label='Train')
    plt.plot(val_metrics['loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_metrics['mse'], label='Train')
    plt.plot(val_metrics['mse'], label='Validation')
    plt.title('MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)
    
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
    
    plt.scatter(targets, predictions, alpha=0.5, s=10)
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = os.path.join(args.output_dir, 'debug_images') if args.debug else None
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Processed images will be saved to {debug_dir}")
    
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        img_size=args.img_size, 
        seed=args.seed
    )
    
    visualize_sample_images(
        train_loader,
        num_samples=5,
        save_path=os.path.join(args.output_dir, 'sample_images.png')
    )
    
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
    
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    scaler = GradScaler() if args.mixed_precision else None
    
    train_metrics = {'loss': [], 'mse': [], 'mae': []}
    val_metrics = {'loss': [], 'mse': [], 'mae': []}
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        start_time = time.time()
        train_results = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            grad_accumulation=args.grad_accumulation,
            max_grad_norm=args.max_grad_norm,
            mixed_precision=args.mixed_precision,
            debug=args.debug,
            debug_dir=debug_dir
        )
        train_time = time.time() - start_time
        
        val_results = validate(
            model, val_loader, criterion, device,
            mixed_precision=args.mixed_precision,
            debug=args.debug,
            debug_dir=debug_dir
        )
        
        scheduler.step()
        
        for metric in ['loss', 'mse', 'mae']:
            train_metrics[metric].append(train_results[metric])
            val_metrics[metric].append(val_results[metric])
        
        print(f"Time: {train_time:.2f}s | "
              f"Train Loss: {train_results['loss']:.6f} | "
              f"Train MSE: {train_results['mse']:.6f} | "
              f"Train MAE: {train_results['mae']:.6f} | "
              f"Val Loss: {val_results['loss']:.6f} | "
              f"Val MSE: {val_results['mse']:.6f} | "
              f"Val MAE: {val_results['mae']:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'val_results': val_results
            }, os.path.join(args.output_dir, 'best_model_640x480_segmented_yuv_01_100_epochs.pth'))
            
            plot_predictions(
                val_results['predictions'].flatten(),
                val_results['targets'].flatten(),
                epoch + 1,
                args.output_dir
            )
        else:
            patience_counter += 1
            
        plot_training_curves(
            train_metrics,
            val_metrics,
            os.path.join(args.output_dir, 'training_curves.png')
        )
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
            
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_640x480_segmented_yuv_01.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print(f"\nBest model from epoch {best_epoch+1} with validation loss {best_val_loss:.6f}")
    
    test_results = validate(
        model, test_loader, criterion, device,
        mixed_precision=args.mixed_precision,
        debug=args.debug,
        debug_dir=debug_dir
    )
    
    print(f"Test MSE: {test_results['mse']:.6f} | Test MAE: {test_results['mae']:.6f}")
    
    plot_predictions(
        test_results['predictions'].flatten(),
        test_results['targets'].flatten(),
        'final_test',
        args.output_dir
    )
    
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'mse': float(test_results['mse']),
            'mae': float(test_results['mae']),
            'loss': float(test_results['loss'])
        }, f, indent=2)
    
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