"""
Real-time inference script for Vision Transformer steering angle prediction.
Combines preprocessing from infer_steering_vit.py with file handling from inference_loop_yuv.py.
"""

import os
import time
import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Local imports
from model_implementation import SteeringAngleTransformer

def parse_args():
    """Parse command line arguments combining both scripts' parameters."""
    parser = argparse.ArgumentParser(description='Real-time ViT steering angle prediction')
    parser.add_argument('--model_path', type=str, 
                        default='/home/daniel/git/carla-driver-data/models/best_model_640x480_segmented_20250329_205433.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_dir', type=str, 
                        default='/home/daniel/dev/claude-dr/transformer-regressor/input_dir/',
                        help='Directory to watch for input images')
    parser.add_argument('--prediction_file', type=str, 
                        default='/home/daniel/dev/claude-dr/transformer-regressor/output_dir/prediction.txt',
                        help='File to write steering predictions')
    parser.add_argument('--img_size', type=int, nargs=2, default=[480, 640],
                        help='Image size (height, width)')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Patch size for Vision Transformer')
    parser.add_argument('--dim', type=int, default=384,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=6,
                        help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=1536,
                        help='MLP dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA inference')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision inference')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (save processed images)')
    parser.add_argument('--debug_dir', type=str, default='debug_images',
                        help='Directory to save debug images')
    return parser.parse_args()

def load_model(model_path, args, device):
    """Load the ViT model from checkpoint."""
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, img_size, debug=False, debug_dir=None):
    """
    Preprocess the image by cropping and resizing, preserving RGB format.
    Based on infer_steering_vit.py's preprocessing.
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Convert to numpy array for OpenCV processing
    img_np = np.array(img)
    
    if debug:
        print(f"Original image shape (H, W, C): {img_np.shape}")
        os.makedirs(debug_dir, exist_ok=True)
        plt.imsave(os.path.join(debug_dir, f"{os.path.basename(image_path)}_original.png"), img_np)
    
    # Initial crop (height 210:480, keep full width) - same as infer_steering_vit.py
    img_np = img_np[210:480, :, :]  # Crop to 270x640 (H×W)
    
    if debug:
        print(f"After cropping (H, W, C): {img_np.shape}")
        plt.imsave(os.path.join(debug_dir, f"{os.path.basename(image_path)}_cropped.png"), img_np)
    
    # Convert to PIL image for torchvision transforms
    img_pil = Image.fromarray(img_np)
    
    # Resize to original dimensions (480x640)
    img_resized = transforms.Resize((img_size[0], img_size[1]))(img_pil)
    
    if debug:
        img_resized.save(os.path.join(debug_dir, f"{os.path.basename(image_path)}_resized.png"))
    
    # Convert to tensor and normalize
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img_resized)
    
    if debug:
        print(f"Preprocessed tensor shape (C, H, W): {img_tensor.shape}")
        # Save normalized image for debugging
        denormalized = img_tensor.clone()
        for t, m, s in zip(denormalized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        denormalized = denormalized.permute(1, 2, 0).numpy()
        plt.imsave(os.path.join(debug_dir, f"{os.path.basename(image_path)}_preprocessed.png"), 
                  np.clip(denormalized, 0, 1))
    
    return img_tensor

def predict_steering(model, img_tensor, device, mixed_precision=False, debug=False):
    """Run inference on a single image."""
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
    if debug:
        print(f"Input tensor shape (B, C, H, W): {img_tensor.shape}")
    
    with torch.no_grad():
        if mixed_precision:
            with torch.cuda.amp.autocast():
                output = model(img_tensor)
        else:
            output = model(img_tensor)
    
    return float(output.cpu().numpy()[0, 0])

def wait_for_file_stability(file_path, timeout=1.0, check_interval=0.05):
    """Wait until the file size stops changing or timeout is reached."""
    start_time = time.time()
    last_size = -1
    
    while time.time() - start_time < timeout:
        if not os.path.exists(file_path):
            return False
        current_size = os.path.getsize(file_path)
        if current_size == last_size and current_size > 0:
            return True
        last_size = current_size
        time.sleep(check_interval)
    
    print(f"Warning: File {file_path} did not stabilize within {timeout}s")
    return False

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, args, device)
    print(f"Loaded model from {args.model_path}")
    
    # Ensure directories exist
    os.makedirs(args.image_dir, exist_ok=True)
    if args.debug:
        os.makedirs(args.debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Images will be saved to {args.debug_dir}")
    
    print(f"Monitoring {args.image_dir} for new images...")
    
    while True:
        # Check for new images
        image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if image_files:
            filename = image_files[0]  # Process the first available image
            img_path = os.path.join(args.image_dir, filename)
            
            # Wait for file to stabilize
            if wait_for_file_stability(img_path):
                try:
                    # Preprocess and predict
                    img_tensor = preprocess_image(
                        img_path, 
                        args.img_size, 
                        debug=args.debug, 
                        debug_dir=args.debug_dir
                    )
                    
                    steering = predict_steering(
                        model, 
                        img_tensor, 
                        device, 
                        mixed_precision=args.mixed_precision, 
                        debug=args.debug
                    )
                    
                    # Write prediction
                    with open(args.prediction_file, 'w') as f:
                        f.write(f"{steering}\n")
                    
                    # Clean up processed image
                    os.remove(img_path)
                    print(f"Processed {filename} - Steering: {steering:.4f}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    if os.path.exists(img_path):
                        os.remove(img_path)  # Clean up on error
            else:
                print(f"Skipping {filename} - file not stable")
                if os.path.exists(img_path):
                    os.remove(img_path)  # Clean up unstable file
        
        time.sleep(0.01)  # Small delay to prevent CPU overload

if __name__ == "__main__":
    main()
