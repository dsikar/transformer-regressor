"""
Inference script for Vision Transformer steering angle regressor.
Generates predictions on training dataset and saves them with labels in a .npy file.
"""

import os
import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from torchvision import transforms
import cv2

# Local imports (assuming these are in the same directory)
from model_implementation import SteeringAngleTransformer
from dataset_utils import create_data_loaders

def parse_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description='Run inference with Vision Transformer for steering angle prediction')
    parser.add_argument('--data_dir', type=str, default="/home/daniel/git/carla-driver-data/scripts/wip/config_640x480_laneid_1/",
                        help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='/home/daniel/git/carla-driver-data/models/best_model_640x480_segmented.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='/home/daniel/git/carla-driver-data/models/inference/',
                        help='Directory to save inference outputs')
    parser.add_argument('--output_file', type=str, default='inference_results.npy',
                        help='Name of the output .npy file for inference results')    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--img_size', type=int, nargs=2, default=[480, 640],
                        help='Image size (height, width)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA inference')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision inference')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode (save processed images)')
    return parser.parse_args()

def preprocess_image(img, debug=False, debug_dir=None, idx=0):
    """
    Preprocess the image by cropping and resizing, preserving RGB format with segmentation overlays.
    Same preprocessing as training script.
    """
    # Initial crop (height 210:480, keep full width)
    img = img[:, 210:480, :]  # Crop to 270x640 (H×W)
    
    # Convert tensor to numpy for OpenCV processing (HWC format)
    img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]

    # Scale to [0, 255] and convert to uint8 for OpenCV
    img_np = (img_np * 255).astype(np.uint8)

    if debug:
        print(f"After cropping (H, W, C): {img_np.shape}")
        cv2.imwrite(os.path.join(debug_dir, f'debug_cropped_{idx}.png'), 
                    cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Convert to PIL image for torchvision transforms
    img_pil = transforms.ToPILImage()(img_np)
    
    if debug:
        img_pil.save(os.path.join(debug_dir, f'debug_pil_{idx}.png'))

    # Resize to original dimensions (480x640)
    img_resized = transforms.Resize((480, 640))(img_pil)
    
    if debug:
        img_resized.save(os.path.join(debug_dir, f'debug_resized_{idx}.png'))
    
    # Convert back to tensor
    img_tensor = transforms.ToTensor()(img_resized)
    
    if debug:
        print(f"After resize and to tensor (C, H, W): {img_tensor.shape}")
        final_img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f'debug_final_{idx}.png'), 
                    cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    
    return img_tensor

def run_inference(model, loader, device, mixed_precision=False, debug=False, debug_dir=None):
    """Run inference on the dataset and collect predictions and labels."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(loader, desc="Inference")):
            # Preprocess images (same as training)
            images = torch.stack([preprocess_image(img, debug=debug, debug_dir=debug_dir, idx=f"inf_{i}_{j}") 
                                for j, img in enumerate(images)])
            images = images.to(device)
            targets = targets.to(device)
            
            if mixed_precision:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Stack all predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    return all_preds, all_targets

def main():
    """Main inference function."""
    args = parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = os.path.join(args.output_dir, 'debug_images') if args.debug else None
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Processed images will be saved to {debug_dir}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Load the full dataset as training data (no split for inference)
    train_loader, _, _ = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=0.0,  # No validation split
        test_split=0.0,  # No test split
        img_size=args.img_size,
        seed=42  # Fixed seed for reproducibility
    )
    
    # Load the model
    checkpoint = torch.load(args.model_path, weights_only=False)
    
    model = SteeringAngleTransformer(
        image_size=tuple(args.img_size),
        patch_size=32,  # Match training default
        num_classes=1,
        dim=384,        # Match training default
        depth=4,        # Match training default
        heads=6,        # Match training default
        mlp_dim=1536,   # Match training default
        dropout=0.1     # Match training default
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded model from {args.model_path}")
    print(f"Best validation loss from training: {checkpoint['val_results']['loss']:.6f}")
    
    # Run inference
    predictions, labels = run_inference(
        model,
        train_loader,
        device,
        mixed_precision=args.mixed_precision,
        debug=args.debug,
        debug_dir=debug_dir
    )
    
    # Calculate metrics
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    print(f"Inference MSE: {mse:.6f} | Inference MAE: {mae:.6f}")
    
    # Save results to .npy file
    results = np.hstack((labels, predictions))
    output_path = os.path.join(args.output_dir, args.output_file)
    np.save(output_path, results)
    print(f"Saved inference results to {output_path}")
    print(f"Shape of saved array: {results.shape} (columns: labels, predictions)")

if __name__ == "__main__":
    main()