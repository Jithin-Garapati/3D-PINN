import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import torch.multiprocessing as mp

from data_utils import create_dataloader
from fno_model import FNO3D

# Set multiprocessing start method to spawn
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

def custom_loss(pred, target, fluid_mask, building_mask, fluid_weight=1.0, building_weight=2.0):
    """Custom loss function that handles fluid and building points separately"""
    # MSE for fluid points
    fluid_loss = ((pred * fluid_mask.unsqueeze(-1) - target * fluid_mask.unsqueeze(-1)) ** 2).mean()
    
    # MSE for building points (should be close to zero)
    building_loss = ((pred * building_mask.unsqueeze(-1)) ** 2).mean()
    
    # Weighted combination
    total_loss = fluid_weight * fluid_loss + building_weight * building_loss
    return total_loss, fluid_loss, building_loss

def train_model(
    data_dir,
    model_dir='models',
    epochs=100,
    learning_rate=1e-3,
    modes=(8, 8, 4),
    width=32,
    batch_size=16
):
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloader with single worker for CUDA compatibility
    train_loader, dataset = create_dataloader(
        data_dir,
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers to avoid CUDA issues
        shuffle=True
    )
    
    # Initialize model
    model = FNO3D(
        modes1=modes[0],
        modes2=modes[1],
        modes3=modes[2],
        width=width,
        in_channels=4,
        out_channels=4
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    train_losses = []
    fluid_losses = []
    building_losses = []
    best_loss = float('inf')
    
    print(f"Training on {len(dataset)} scenarios with batch size {batch_size}")
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_fluid_losses = []
        epoch_building_losses = []
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to GPU
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                fluid_mask = batch['fluid_mask'].to(device)
                building_mask = batch['building_mask'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss, fluid_loss, building_loss = custom_loss(
                    outputs, targets, fluid_mask, building_mask
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_losses.append(loss.item())
                epoch_fluid_losses.append(fluid_loss.item())
                epoch_building_losses.append(building_loss.item())
                
                if batch_idx % 10 == 0:  # Update less frequently
                    pbar.set_postfix({
                        'batch': f'{batch_idx}/{len(train_loader)}',
                        'loss': f'{loss.item():.6f}',
                        'fluid_loss': f'{fluid_loss.item():.6f}',
                        'building_loss': f'{building_loss.item():.6f}'
                    })
        
        # Step scheduler
        scheduler.step()
        
        # Calculate average losses
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_fluid_loss = sum(epoch_fluid_losses) / len(epoch_fluid_losses)
        avg_building_loss = sum(epoch_building_losses) / len(epoch_building_losses)
        
        train_losses.append(avg_loss)
        fluid_losses.append(avg_fluid_loss)
        building_losses.append(avg_building_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'coord_scaler': dataset.coord_scaler,
                'velocity_scaler': dataset.velocity_scaler,
                'pressure_scaler': dataset.pressure_scaler
            }, os.path.join(model_dir, 'best_model.pth'))
        
        print(f'Epoch {epoch+1}:')
        print(f'  Average Loss: {avg_loss:.6f}')
        print(f'  Fluid Loss: {avg_fluid_loss:.6f}')
        print(f'  Building Loss: {avg_building_loss:.6f}')
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(train_losses)
    plt.title('Total Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(fluid_losses)
    plt.title('Fluid Region Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(building_losses)
    plt.title('Building Region Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'))
    plt.close()

if __name__ == '__main__':
    train_model('filtered_dataset')