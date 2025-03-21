"""
Inference script for the Vision Transformer steering angle regressor.
Supports both PyTorch and ONNX Runtime inference.
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# ONNX Runtime for optimized inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available. Install with 'pip install onnxruntime'")

# Local imports
from model_implementation import SteeringAngleTransformer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inference for Vision Transformer steering angle prediction')
    
    # Input/output parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint (.pth) or ONNX model (.onnx)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save output visualizations')
    
    # Model parameters (only needed for PyTorch model)
    parser.add_argument('--img_size', type=int, nargs=2, default=[480, 640],
                        help='Input image size (height, width)')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Patch size for Vision Transformer')
    parser.add_argument('--dim', type=int, default=384,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=6,
                        help='Number of attention heads')
    
    # Inference parameters
    parser.add_argument('--use_onnx', action='store_true',
                        help='Use ONNX Runtime for inference')
    parser.add_argument('--quantized', action='store_true',
                        help='Use quantized ONNX model (only with --use_onnx)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference on directories')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA inference')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of steering predictions')
    
    return parser.parse_args()


def load_pytorch_model(model_path, args, device):
    """Load PyTorch model from checkpoint."""
    # Create model with same architecture
    model = SteeringAngleTransformer(
        image_size=tuple(args.img_size),
        patch_size=args.patch_size,
        num_classes=1,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def load_onnx_model(model_path):
    """Load ONNX model for inference with ONNX Runtime."""
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime is not available. Install with 'pip install onnxruntime'")
    
    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Set execution providers
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    
    # Load model
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    
    return session


def preprocess_image(image_path, img_size):
    """Preprocess image for model input."""
    # Load and convert to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Apply preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return preprocess(image)


def get_image_paths(input_path):
    """Get list of image paths from directory or single image path."""
    if os.path.isdir(input_path):
        # Find all image files in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(input_path, filename))
        
        return sorted(image_paths)
    else:
        # Single image file
        return [input_path]


def extract_steering_angle_from_filename(filename):
    """Extract steering angle from filename if available."""
    pattern = re.compile(r'.*_steering_([-+]?\d*\.\d+)\..*$')
    match = pattern.match(filename)
    
    if match:
        return float(match.group(1))
    return None


def batch_predict_pytorch(model, image_paths, img_size, batch_size, device):
    """Run batch prediction with PyTorch model."""
    predictions = []
    true_angles = []
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_inputs = []
        batch_angles = []
        
        for path in batch_paths:
            # Preprocess image
            img_tensor = preprocess_image(path, img_size)
            batch_inputs.append(img_tensor)
            
            # Extract ground truth angle if available
            angle = extract_steering_angle_from_filename(os.path.basename(path))
            batch_angles.append(angle)
        
        # Stack inputs into batch
        input_batch = torch.stack(batch_inputs).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_batch)
        
        # Add to results
        predictions.extend(outputs.cpu().numpy().flatten())
        true_angles.extend(batch_angles)
    
    return predictions, true_angles


def batch_predict_onnx(session, image_paths, img_size, batch_size):
    """Run batch prediction with ONNX Runtime."""
    predictions = []
    true_angles = []
    input_name = session.get_inputs()[0].name
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_inputs = []
        batch_angles = []
        
        for path in batch_paths:
            # Preprocess image
            img_tensor = preprocess_image(path, img_size)
            batch_inputs.append(img_tensor.numpy())
            
            # Extract ground truth angle if available
            angle = extract_steering_angle_from_filename(os.path.basename(path))
            batch_angles.append(angle)
        
        # Stack inputs into batch
        input_batch = np.stack(batch_inputs)
        
        # Run inference
        outputs = session.run(None, {input_name: input_batch})[0]
        
        # Add to results
        predictions.extend(outputs.flatten())
        true_angles.extend(batch_angles)
    
    return predictions, true_angles


def visualize_steering(image_path, angle, true_angle=None, output_path=None):
    """Create visualization of steering angle prediction."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title(f"Input Image: {os.path.basename(image_path)}")
    ax1.axis('off')
    
    # Create steering wheel visualization
    wheel_img = plt.imread(os.path.join(os.path.dirname(__file__), 'steering_wheel.png')) if os.path.exists(os.path.join(os.path.dirname(__file__), 'steering_wheel.png')) else None
    
    # If we don't have a steering wheel image, create a simple bar visualization
    if wheel_img is None:
        # Create bar chart showing angle
        bar_width = 0.3
        x = [0, 1] if true_angle is not None else [0]
        y = [angle, true_angle] if true_angle is not None else [angle]
        labels = ['Predicted', 'Ground Truth'] if true_angle is not None else ['Predicted']
        colors = ['blue', 'green'] if true_angle is not None else ['blue']
        
        ax2.bar(x, y, width=bar_width, color=colors)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Steering Angle')
        ax2.set_title('Steering Prediction')
        ax2.grid(alpha=0.3)
        
        # Add limit lines for min/max range
        ax2.axhline(y=-0.04, color='r', linestyle='-', alpha=0.3)
        ax2.axhline(y=0.08, color='r', linestyle='-', alpha=0.3)
    else:
        # Create steering wheel visualization
        ax2.imshow(wheel_img)
        ax2.axis('off')
        
        # Calculate rotation angle (convert from steering angle to degrees)
        # Assuming steering angle range of [-0.04, 0.08] maps to [-30, 30] degrees
        min_angle, max_angle = -0.04, 0.08
        min_degrees, max_degrees = -30, 30
        
        rotation = ((angle - min_angle) / (max_angle - min_angle)) * (max_degrees - min_degrees) + min_degrees
        
        # Create a rotated version of the wheel
        ax2.set_title(f"Predicted Angle: {angle:.4f}" + 
                     (f"\nGround Truth: {true_angle:.4f}" if true_angle is not None else ""))
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def run_inference(args):
    """Run inference on images using either PyTorch or ONNX model."""
    # Create output directory if needed
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image paths
    image_paths = get_image_paths(args.input)
    if len(image_paths) == 0:
        print(f"No images found at {args.input}")
        return
    
    print(f"Running inference on {len(image_paths)} images")
    
    # Set up model
    if args.use_onnx:
        # Use ONNX Runtime
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available. Falling back to PyTorch.")
            args.use_onnx = False
        else:
            print(f"Loading ONNX model: {args.model_path}")
            session = load_onnx_model(args.model_path)
            print(f"ONNX model loaded. Input name: {session.get_inputs()[0].name}")
    
    if not args.use_onnx:
        # Use PyTorch
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        print(f"Loading PyTorch model: {args.model_path} (Device: {device})")
        model = load_pytorch_model(args.model_path, args, device)
        print(f"PyTorch model loaded")
    
    # Run inference
    start_time = time.time()
    
    if args.use_onnx:
        predictions, true_angles = batch_predict_onnx(
            session, image_paths, tuple(args.img_size), args.batch_size
        )
    else:
        predictions, true_angles = batch_predict_pytorch(
            model, image_paths, tuple(args.img_size), args.batch_size, device
        )
    
    inference_time = time.time() - start_time
    avg_time_per_image = inference_time / len(image_paths)
    
    print(f"Inference completed in {inference_time:.2f}s ({avg_time_per_image*1000:.2f}ms per image)")
    
    # Calculate metrics if ground truth is available
    if all(angle is not None for angle in true_angles):
        mse = np.mean((np.array(predictions) - np.array(true_angles)) ** 2)
        mae = np.mean(np.abs(np.array(predictions) - np.array(true_angles)))
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # Plot predictions vs ground truth
        if args.visualize:
            plt.figure(figsize=(10, 6))
            plt.scatter(true_angles, predictions, alpha=0.5)
            
            # Add y=x line
            min_val = min(min(predictions), min(true_angles))
            max_val = max(max(predictions), max(true_angles))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title('Predicted vs Actual Steering Angles')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(args.output_dir, 'predictions_vs_actual.png'))
            plt.close()
    
    # Create visualizations
    if args.visualize:
        print("Generating visualizations...")
        for i, (path, pred) in enumerate(zip(image_paths, predictions)):
            true_angle = true_angles[i] if true_angles[i] is not None else None
            
            output_filename = os.path.splitext(os.path.basename(path))[0] + '_viz.png'
            output_path = os.path.join(args.output_dir, output_filename)
            
            visualize_steering(path, pred, true_angle, output_path)
        
        print(f"Visualizations saved to {args.output_dir}")
    
    # Write predictions to file
    results = []
    for path, pred, true in zip(image_paths, predictions, true_angles):
        results.append({
            'image': os.path.basename(path),
            'predicted_angle': float(pred),
            'true_angle': float(true) if true is not None else None
        })
    
    with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Predictions saved to {os.path.join(args.output_dir, 'predictions.json')}")


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()