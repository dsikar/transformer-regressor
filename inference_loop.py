"""
Real-time inference script for Vision Transformer steering angle prediction.
Loads images, predicts steering angles, writes predictions, and deletes images.
"""

import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from model_implementation import SteeringAngleTransformer

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time inference for CARLA steering prediction')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint (.pth) or ONNX model (.onnx)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory to watch for CARLA images')
    parser.add_argument('--prediction_file', type=str, required=True,
                        help='File to write steering predictions')
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
    parser.add_argument('--use_onnx', action='store_true',
                        help='Use ONNX Runtime for inference')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA inference')
    return parser.parse_args()

def load_pytorch_model(model_path, args, device):
    model = SteeringAngleTransformer(
        image_size=tuple(args.img_size),
        patch_size=args.patch_size,
        num_classes=1,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def load_onnx_model(model_path):
    if not ONNX_AVAILABLE:
        raise ImportError("Install onnxruntime for ONNX support")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    return ort.InferenceSession(model_path, sess_options, providers=providers)

def preprocess_image(image_path, img_size):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)

def predict_pytorch(model, img_tensor, device):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    return float(output.cpu().numpy()[0, 0])

def predict_onnx(session, img_tensor):
    input_name = session.get_inputs()[0].name
    input_batch = img_tensor.unsqueeze(0).numpy()
    output = session.run(None, {input_name: input_batch})[0]
    return float(output[0, 0])

def wait_for_file_stability(file_path, timeout=1.0, check_interval=0.05):
    """Wait until the file size stops changing or timeout is reached."""
    start_time = time.time()
    last_size = -1
    
    while time.time() - start_time < timeout:
        if not os.path.exists(file_path):
            return False  # File was deleted or never existed
        current_size = os.path.getsize(file_path)
        if current_size == last_size and current_size > 0:
            return True  # File size stabilized
        last_size = current_size
        time.sleep(check_interval)
    
    print(f"Warning: File {file_path} did not stabilize within {timeout}s")
    return False

def main():
    args = parse_args()
    
    # Setup model
    if args.use_onnx and ONNX_AVAILABLE:
        print(f"Loading ONNX model: {args.model_path}")
        session = load_onnx_model(args.model_path)
        predict_fn = lambda img: predict_onnx(session, img)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        print(f"Loading PyTorch model: {args.model_path} (Device: {device})")
        model = load_pytorch_model(args.model_path, args, device)
        predict_fn = lambda img: predict_pytorch(model, img, device)
    
    # Ensure image directory exists
    os.makedirs(args.image_dir, exist_ok=True)
    
    print(f"Watching {args.image_dir} for images...")
    
    while True:
        # Check for new images
        image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if image_files:
            filename = image_files[0]  # Process one at a time
            img_path = os.path.join(args.image_dir, filename)
            print(f"Processing {filename}")
            
            # Wait for file to stabilize
            if wait_for_file_stability(img_path):
                try:
                    # Preprocess and predict
                    img_tensor = preprocess_image(img_path, tuple(args.img_size))
                    steering = predict_fn(img_tensor)
                    
                    # Write prediction (overwrite file each time)
                    with open(args.prediction_file, "w") as f:
                        f.write(f"{steering}\n")
                    
                    # Delete the image
                    os.remove(img_path)
                    print(f"Predicted steering: {steering:.4f}, deleted {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    if os.path.exists(img_path):
                        os.remove(img_path)  # Clean up even on error
            else:
                print(f"Skipping {filename}: File not fully written")
                if os.path.exists(img_path):
                    os.remove(img_path)  # Clean up if stuck
            
        time.sleep(0.01)  # Small delay to avoid CPU overload

if __name__ == "__main__":
    main()

"""
Example usage:
python inference_loop.py --model_path ~/git/carla-driver-data/models/best_model.pth \
    --image_dir /home/daniel/dev/claude-dr/transformer-regressor/input_dir/ \
    --prediction_file /home/daniel/dev/claude-dr/transformer-regressor/output_dir/prediction.txt
"""