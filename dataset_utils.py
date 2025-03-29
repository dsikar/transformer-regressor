"""
Dataset utilities for processing and loading steering angle prediction dataset.
The dataset consists of images with steering angles encoded in filenames.
Example filename: 20250314_094716_119918_steering_0.0117.jpg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import re
from tqdm import tqdm
import random
import pandas as pd
import shutil
from pathlib import Path


class SteeringAngleDataset(Dataset):
    """
    Dataset for loading images and corresponding steering angles from filenames.
    The steering angle is extracted from the image filename.
    """
    def __init__(self, root_dir, transform=None, horizontal_flip=False, angle_range=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Directory containing the image files
            transform (callable, optional): Transform to apply to the images
            horizontal_flip (bool): If True, allow horizontal flipping with angle negation
            angle_range (tuple, optional): If provided, filter angles to this range (min, max)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.horizontal_flip = horizontal_flip
        self.image_paths = []
        self.steering_angles = []
        
        # Compile regex pattern for extracting steering angle from filename
        self.pattern = re.compile(r'.*_steering_([-+]?\d*\.\d+)\.jpg$')
        
        # Load all image paths and extract steering angles
        for filename in sorted(os.listdir(root_dir)):  # Sort filenames in ascending order
            if filename.endswith('.jpg'):
                match = self.pattern.match(filename)
                if match:
                    steering_angle = float(match.group(1))
                    
                    # Filter by angle range if specified
                    if angle_range is not None:
                        min_angle, max_angle = angle_range
                        if steering_angle < min_angle or steering_angle > max_angle:
                            continue
                    
                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.steering_angles.append(steering_angle)
        
        # Sort by filename (to ensure reproducibility)
        paired_data = sorted(zip(self.image_paths, self.steering_angles), key=lambda x: x[0])
        self.image_paths, self.steering_angles = zip(*paired_data) if paired_data else ([], [])
        
        # Convert to lists
        self.image_paths = list(self.image_paths)
        self.steering_angles = list(self.steering_angles)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        steering_angle = self.steering_angles[idx]
        
        # Handle horizontal flips
        if self.horizontal_flip and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            steering_angle = -steering_angle
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([steering_angle], dtype=torch.float32)
    
    def get_stats(self):
        """
        Return statistics about the dataset
        """
        angles = np.array(self.steering_angles)
        
        stats = {
            'count': len(angles),
            'min': np.min(angles),
            'max': np.max(angles),
            'mean': np.mean(angles),
            'median': np.median(angles),
            'std': np.std(angles)
        }
        
        return stats
    
    def plot_distribution(self, save_path=None):
        """
        Plot the distribution of steering angles
        """
        angles = np.array(self.steering_angles)
        
        plt.figure(figsize=(10, 6))
        plt.hist(angles, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=np.mean(angles), color='red', linestyle='--', label=f'Mean: {np.mean(angles):.4f}')
        plt.axvline(x=np.median(angles), color='green', linestyle='-.', label=f'Median: {np.median(angles):.4f}')
        plt.xlabel('Steering Angle')
        plt.ylabel('Count')
        plt.title('Distribution of Steering Angles')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def create_data_loaders(data_dir, batch_size=8, val_split=0.2, test_split=0.1, 
                        img_size=(480, 640), augment=False, seed=42):
    """
    Create training, validation, and test data loaders.
    
    Args:
        data_dir (str): Directory containing the images
        batch_size (int): Batch size for training
        val_split (float): Fraction of data to use for validation
        test_split (float): Fraction of data to use for testing
        img_size (tuple): Target image size (height, width)
        augment (bool): Whether to apply data augmentation
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Define transformations
    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_transform = base_transform

    # Create dataset
    full_dataset = SteeringAngleDataset(
        data_dir,
        transform=None,  # No transform here, will be applied in the subset classes
        horizontal_flip=False  # Handle horizontal flipping separately
    )
    
    # Print dataset statistics
    stats = full_dataset.get_stats()
    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Plot distribution
    full_dataset.plot_distribution(save_path="angle_distribution.png")
    
    # Calculate splits
    dataset_size = len(full_dataset)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # If no validation or test split, use the full dataset without shuffling
    if val_size == 0 and test_size == 0:
        train_dataset = TransformDataset(full_dataset, list(range(dataset_size)), train_transform, horizontal_flip=augment)
        val_dataset = None
        test_dataset = None
        print(f"Dataset split: Train: {len(train_dataset)}, Validation: 0, Test: 0")
    else:
        # Create random splits
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create dataset wrappers with transforms
        train_dataset = TransformDataset(full_dataset, train_dataset.indices, train_transform, horizontal_flip=augment)
        val_dataset = TransformDataset(full_dataset, val_dataset.indices, base_transform)
        test_dataset = TransformDataset(full_dataset, test_dataset.indices, base_transform)
        print(f"Dataset split: Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=False,  # Disable shuffling to preserve sorted order
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = None if test_dataset is None else DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
class TransformDataset(Dataset):
    """
    Wrapper dataset that applies transforms to a subset of another dataset
    """
    def __init__(self, dataset, indices, transform=None, horizontal_flip=False):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.horizontal_flip = horizontal_flip

    def __getitem__(self, idx):
        img_path = self.dataset.image_paths[self.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        steering_angle = self.dataset.steering_angles[self.indices[idx]]
        
        # Apply horizontal flip with probability 0.5 for training
        if self.horizontal_flip and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            steering_angle = -steering_angle
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, torch.tensor([steering_angle], dtype=torch.float32)

    def __len__(self):
        return len(self.indices)


def create_dataset_csv(data_dir, output_csv='dataset_info.csv'):
    """
    Create a CSV file with information about each image in the dataset
    """
    pattern = re.compile(r'.*_steering_([-+]?\d*\.\d+)\.jpg$')
    
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            match = pattern.match(filename)
            if match:
                steering_angle = float(match.group(1))
                
                # Get image creation time (or modification time as fallback)
                img_path = os.path.join(data_dir, filename)
                try:
                    creation_time = os.path.getctime(img_path)
                except:
                    creation_time = os.path.getmtime(img_path)
                
                # Get image size
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except:
                    width, height = None, None
                
                data.append({
                    'filename': filename,
                    'steering_angle': steering_angle,
                    'creation_time': creation_time,
                    'width': width,
                    'height': height,
                    'file_size_kb': os.path.getsize(img_path) / 1024
                })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Dataset information saved to {output_csv}")
    
    return df


def verify_dataset_integrity(data_dir):
    """
    Verify that all images in the dataset can be opened and have valid steering angles
    """
    pattern = re.compile(r'.*_steering_([-+]?\d*\.\d+)\.jpg$')
    
    invalid_files = []
    valid_count = 0
    
    for filename in tqdm(os.listdir(data_dir), desc="Verifying dataset"):
        if filename.endswith('.jpg'):
            img_path = os.path.join(data_dir, filename)
            
            # Check if filename contains valid steering angle
            match = pattern.match(filename)
            if not match:
                invalid_files.append((filename, "Invalid filename format"))
                continue
            
            # Check if image can be opened
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_count += 1
            except Exception as e:
                invalid_files.append((filename, f"Cannot open image: {str(e)}"))
    
    # Print results
    print(f"Dataset verification complete:")
    print(f"  Valid files: {valid_count}")
    print(f"  Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print("\nInvalid files:")
        for filename, reason in invalid_files[:10]:  # Show only first 10
            print(f"  {filename}: {reason}")
        
        if len(invalid_files) > 10:
            print(f"  ... and {len(invalid_files) - 10} more")
    
    return valid_count, invalid_files


def visualize_sample_images(data_loader, num_samples=5, save_path=None):
    """
    Visualize sample images from a data loader with their steering angles
    """
    # Get a batch of samples
    images, angles = next(iter(data_loader))
    
    # Limit to requested number of samples
    images = images[:num_samples]
    angles = angles[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 4, 4))
    if num_samples == 1:
        axes = [axes]
    
    # Denormalization function
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ])
    
    for i, (img, angle) in enumerate(zip(images, angles)):
        # Denormalize image
        img = denormalize(img)
        img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
        img = np.clip(img, 0, 1)
        
        # Display image
        axes[i].imshow(img)
        axes[i].set_title(f"Angle: {angle.item():.4f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    
def create_balanced_subset(data_dir, output_dir, bins=20, samples_per_bin=100, seed=42):
    """
    Create a balanced subset of the dataset with an equal number of samples per angle bin
    
    Args:
        data_dir (str): Original dataset directory
        output_dir (str): Directory to save the balanced subset
        bins (int): Number of bins to divide the angle range into
        samples_per_bin (int): Maximum number of samples to take from each bin
        seed (int): Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the original dataset
    dataset = SteeringAngleDataset(data_dir, transform=None)
    
    # Create angle bins
    angles = np.array(dataset.steering_angles)
    min_angle, max_angle = np.min(angles), np.max(angles)
    bin_edges = np.linspace(min_angle, max_angle, bins + 1)
    
    # Group images by bin
    binned_images = [[] for _ in range(bins)]
    
    for img_path, angle in zip(dataset.image_paths, dataset.steering_angles):
        bin_idx = np.digitize(angle, bin_edges) - 1
        if bin_idx >= bins:
            bin_idx = bins - 1
        binned_images[bin_idx].append((img_path, angle))
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Select random samples from each bin
    selected_images = []
    
    for bin_idx, images in enumerate(binned_images):
        bin_start = bin_edges[bin_idx]
        bin_end = bin_edges[bin_idx + 1]
        
        # Shuffle and select samples
        # random.shuffle(images)
        bin_samples = images[:samples_per_bin]
        selected_images.extend(bin_samples)
        
        print(f"Bin {bin_idx}: {bin_start:.4f} to {bin_end:.4f} - Selected {len(bin_samples)} out of {len(images)} images")
    
    # Copy selected images to output directory
    print(f"Copying {len(selected_images)} images to {output_dir}...")
    
    for img_path, angle in tqdm(selected_images):
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(output_dir, filename))
    
    print("Done!")
    
    # Create visualization of the balance
    original_counts, _ = np.histogram(angles, bins=bin_edges)
    
    selected_angles = [angle for _, angle in selected_images]
    balanced_counts, _ = np.histogram(selected_angles, bins=bin_edges)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(bin_centers, original_counts, width=(max_angle-min_angle)/bins, alpha=0.7)
    plt.title('Original Dataset')
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.bar(bin_centers, balanced_counts, width=(max_angle-min_angle)/bins, alpha=0.7)
    plt.title('Balanced Subset')
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'balance_comparison.png'))
    plt.close()


if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/your/dataset"  # Update with actual dataset path
    
    # Verify dataset integrity
    verify_dataset_integrity(data_dir)
    
    # Create dataset information CSV
    create_dataset_csv(data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data_dir)
    
    # Visualize sample images
    visualize_sample_images(train_loader, save_path="sample_images.png")