"""
Vision Transformer implementation for steering angle prediction in autonomous vehicles.
This implementation is designed for a dataset of 15k images with steering angles in the filename.
The model is optimized to run on a 6GB GPU with CUDA 6.1.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import re
from tqdm import tqdm
import random


class SteeringAngleTransformer(nn.Module):
    """
    Vision Transformer model for steering angle prediction.
    Optimized for memory efficiency to run on GPUs with limited memory (6GB).
    """
    def __init__(self, image_size=(480, 640), patch_size=32, num_classes=1, 
                 dim=384, depth=4, heads=6, mlp_dim=1536, dropout=0.1, emb_dropout=0.1):
        super(SteeringAngleTransformer, self).__init__()
        
        # Calculate the number of patches
        h, w = image_size
        assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by patch size"
        num_patches = (h // patch_size) * (w // patch_size)
        patch_dim = 3 * patch_size ** 2  # 3 channels * patch area
        
        # Define the embedding layer
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        )
        
        # Position embedding and cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=mlp_dim, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Layer norm
        self.ln = nn.LayerNorm(dim)
        
        # Regression head
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Tanh()  # Constrain output to [-1, 1] range
        )

    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # Convert image to patches
        # [batch_size, dim, patches_h, patches_w]
        x = self.to_patch_embedding(x)
        
        # Flatten patches to sequence
        # [batch_size, dim, patches_h * patches_w] -> [batch_size, patches_h * patches_w, dim]
        x = x.flatten(2).transpose(1, 2)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Use cls token for prediction
        x = x[:, 0]
        x = self.ln(x)
        
        # MLP head for regression
        return self.mlp_head(x)


class SteeringAngleDataset(Dataset):
    """
    Dataset for loading images and corresponding steering angles from filenames.
    Example filename: 20250314_094716_119918_steering_0.0117.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.steering_angles = []
        
        # Compile regex pattern for extracting steering angle from filename
        self.pattern = re.compile(r'.*_steering_([-+]?\d*\.\d+)\.jpg$')
        
        # Load all image paths and extract steering angles
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg'):
                match = self.pattern.match(filename)
                if match:
                    steering_angle = float(match.group(1))
                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.steering_angles.append(steering_angle)
        
        # Print dataset statistics
        if len(self.steering_angles) > 0:
            print(f"Loaded {len(self.steering_angles)} images")
            print(f"Steering angle range: [{min(self.steering_angles)}, {max(self.steering_angles)}]")
            print(f"Mean steering angle: {sum(self.steering_angles) / len(self.steering_angles)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        steering_angle = self.steering_angles[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([steering_angle], dtype=torch.float32)


def create_data_loaders(data_dir, batch_size=8, val_split=0.2):
    """
    Create training and validation data loaders with appropriate transforms
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Flip with 50% probability, will need to negate steering angle
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = SteeringAngleDataset(data_dir, transform=None)  # No transform yet
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    # Shuffle indices
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create custom datasets with correct transforms
    train_dataset = Subset(full_dataset, train_indices, train_transform)
    val_dataset = Subset(full_dataset, val_indices, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


class Subset(Dataset):
    """
    Subset of a dataset with a specific transform
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.dataset.image_paths[self.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        steering_angle = self.dataset.steering_angles[self.indices[idx]]
        
        # Handle horizontal flips with angle negation for training
        if self.transform:
            # Check if transform includes horizontal flip and save the random decision
            do_flip = False
            if isinstance(self.transform, transforms.Compose):
                for t in self.transform.transforms:
                    if isinstance(t, transforms.RandomHorizontalFlip) and random.random() < t.p:
                        do_flip = True
                        break
            
            # Apply transform
            image = self.transform(image)
            
            # If image was flipped, negate the steering angle
            if do_flip:
                steering_angle = -steering_angle
        else:
            image = transforms.ToTensor()(image)
        
        return image, torch.tensor([steering_angle], dtype=torch.float32)

    def __len__(self):
        return len(self.indices)


def train_model(model, train_loader, val_loader, num_epochs=30, 
                learning_rate=1e-4, device='cuda', weight_decay=1e-4):
    """
    Train the model with early stopping and learning rate scheduling
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5  # for early stopping
    patience_counter = 0
    
    # For plotting
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, angles in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)
            angles = angles.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, angles)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        # Calculate average training loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, angles in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.to(device)
                angles = angles.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, angles)
                
                running_loss += loss.item() * images.size(0)
        
        # Calculate average validation loss
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
        
        # Update learning rate
        scheduler.step()
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    
    return model


def export_onnx_model(model, output_path='model.onnx'):
    """
    Export the PyTorch model to ONNX format for deployment
    """
    model.eval()
    
    # Create a dummy input for tracing
    dummy_input = torch.randn(1, 3, 480, 640, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported to {output_path}")


def visualize_predictions(model, val_loader, device='cuda', num_samples=5):
    """
    Visualize some predictions against ground truth
    """
    model.eval()
    
    # Get random samples from validation set
    dataiter = iter(val_loader)
    images, angles = next(dataiter)
    
    # Make predictions
    images = images.to(device)
    with torch.no_grad():
        predictions = model(images)
    
    # Convert to numpy for plotting
    images = images.cpu().numpy()
    predictions = predictions.cpu().numpy()
    angles = angles.cpu().numpy()
    
    # Plot results
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Display image (need to transpose and denormalize)
        img = np.transpose(images[i], (1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i+1}")
        axes[i, 0].axis('off')
        
        # Plot actual vs predicted steering angle
        axes[i, 1].bar(['Actual', 'Predicted'], [angles[i, 0], predictions[i, 0]], color=['blue', 'red'])
        axes[i, 1].set_ylim([-0.05, 0.09])  # Set y-limits based on dataset range
        axes[i, 1].set_title(f"Actual: {angles[i, 0]:.4f}, Predicted: {predictions[i, 0]:.4f}")
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    plt.close()


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    data_dir = "path/to/dataset"  # Update with actual dataset path
    train_loader, val_loader = create_data_loaders(data_dir, batch_size=8)
    
    # Create model
    model = SteeringAngleTransformer(image_size=(480, 640), patch_size=32, num_classes=1)
    
    # Print model summary
    print(model)
    
    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {param_count:,}")
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=30, device=device)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Visualize predictions
    visualize_predictions(model, val_loader, device=device)
    
    # Export to ONNX for deployment
    export_onnx_model(model)


if __name__ == "__main__":
    main()