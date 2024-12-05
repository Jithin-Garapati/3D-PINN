import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from fno_model import FNO3D
import os
import glob
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals

# Add StandardScaler to safe globals
add_safe_globals([StandardScaler])

def list_available_csvs(data_dir):
    """List all available CSV files in the directory"""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return []
    
    print("\nAvailable CSV files:")
    for i, file in enumerate(csv_files):
        print(f"{i+1}. {os.path.basename(file)}")
    return csv_files

def load_model(model_path, device=None):
    """Load trained model and scalers"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print("Loading model from:", model_path)
    print(f"Using device: {device}")
    
    try:
        # Try loading with weights_only first
        checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    except Exception as e:
        print("Warning: Failed to load with weights_only=True, falling back to full load")
        checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = FNO3D(
        modes1=8, modes2=8, modes3=4,
        width=32,
        in_channels=4,
        out_channels=4
    ).to(device)
    
    # Load state and scalers
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['coord_scaler'], checkpoint['velocity_scaler'], checkpoint['pressure_scaler'], device

def normalize_coordinates(df):
    """Normalize coordinates to fit within grid size"""
    # Get min and max for each coordinate
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()
    
    # Create normalized coordinates
    df_normalized = df.copy()
    df_normalized['x'] = ((df['x'] - x_min) * 23 / (x_max - x_min)).astype(int)
    df_normalized['y'] = ((df['y'] - y_min) * 23 / (y_max - y_min)).astype(int)
    df_normalized['z'] = ((df['z'] - z_min) * 7 / (z_max - z_min)).astype(int)
    
    return df_normalized

def prepare_input(df, grid_size, coord_scaler):
    """Prepare input grid from CSV data"""
    input_grid = np.zeros((*grid_size, 4))
    
    # Normalize coordinates to fit grid
    df_normalized = normalize_coordinates(df)
    
    for i in range(len(df_normalized)):
        x, y, z = df_normalized['x'].iloc[i], df_normalized['y'].iloc[i], df_normalized['z'].iloc[i]
        if (0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]):
            coords = np.array([[df['x'].iloc[i], df['y'].iloc[i], df['z'].iloc[i]]])  # Use original coords for scaling
            coords_scaled = coord_scaler.transform(coords)[0]
            input_grid[x, y, z] = np.append(coords_scaled, df['bm'].iloc[i])
    
    return input_grid, df_normalized

def analyze_predictions(csv_path, model_path, save_dir='analysis_results'):
    """Perform detailed analysis of model predictions"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, coord_scaler, velocity_scaler, pressure_scaler, device = load_model(model_path, device)
    df = pd.read_csv(csv_path)
    print(f"\nAnalyzing file: {csv_path}")
    
    # Prepare input with normalized coordinates
    grid_size = (24, 24, 8)
    input_grid, df_normalized = prepare_input(df, grid_size, coord_scaler)
    
    # Get predictions
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_grid).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred = output.cpu().squeeze().numpy()
    
    # Extract ground truth and predictions
    fluid_points = df[df['bm'] == 0]
    building_points = df[df['bm'] == 1]
    fluid_points_norm = df_normalized[df_normalized['bm'] == 0]
    
    # Initialize arrays for analysis
    true_velocities = []
    pred_velocities = []
    true_pressures = []
    pred_pressures = []
    
    for i in range(len(fluid_points)):
        x = fluid_points_norm['x'].iloc[i]
        y = fluid_points_norm['y'].iloc[i]
        z = fluid_points_norm['z'].iloc[i]
        
        if (0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]):
            # Get predicted values
            pred_vel = velocity_scaler.inverse_transform(pred[x, y, z, :3].reshape(1, -1))[0]
            pred_p = pressure_scaler.inverse_transform(pred[x, y, z, 3:].reshape(1, -1))[0]
            
            # Get true values
            true_vel = [fluid_points['u'].iloc[i], fluid_points['v'].iloc[i], fluid_points['w'].iloc[i]]
            true_p = fluid_points['p'].iloc[i]
            
            true_velocities.append(true_vel)
            pred_velocities.append(pred_vel)
            true_pressures.append(true_p)
            pred_pressures.append(pred_p)
    
    # Convert to numpy arrays
    true_velocities = np.array(true_velocities)
    pred_velocities = np.array(pred_velocities)
    true_pressures = np.array(true_pressures)
    pred_pressures = np.array(pred_pressures)
    
    if len(true_velocities) == 0:
        print("No valid predictions found! Check if coordinates are within grid bounds.")
        return None
    
    # Calculate errors
    velocity_errors = np.abs(true_velocities - pred_velocities)
    pressure_errors = np.abs(np.array(true_pressures) - np.array(pred_pressures))
    
    # Print statistics
    print("\nStatistical Analysis:")
    print("-" * 50)
    print("Velocity Components (u, v, w):")
    print(f"Mean Absolute Error: {velocity_errors.mean(axis=0)}")
    print(f"Max Absolute Error: {velocity_errors.max(axis=0)}")
    print(f"Standard Deviation of Error: {velocity_errors.std(axis=0)}")
    
    print("\nPressure:")
    print(f"Mean Absolute Error: {pressure_errors.mean()}")
    print(f"Max Absolute Error: {pressure_errors.max()}")
    print(f"Standard Deviation of Error: {pressure_errors.std()}")
    
    # Plot velocity comparisons
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = ['u', 'v', 'w']
    
    for i, (ax, comp) in enumerate(zip(axes, components)):
        ax.scatter(true_velocities[:, i], pred_velocities[:, i], alpha=0.5)
        ax.plot([-20, 20], [-20, 20], 'r--')  # Perfect prediction line
        ax.set_xlabel(f'True {comp}')
        ax.set_ylabel(f'Predicted {comp}')
        ax.set_title(f'{comp} Velocity Component')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'velocity_comparison.png'))
    plt.close()
    
    # Plot pressure comparison
    plt.figure(figsize=(8, 8))
    plt.scatter(true_pressures, pred_pressures, alpha=0.5)
    plt.plot([-100, 100], [-100, 100], 'r--')  # Perfect prediction line
    plt.xlabel('True Pressure')
    plt.ylabel('Predicted Pressure')
    plt.title('Pressure Comparison')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pressure_comparison.png'))
    plt.close()
    
    # Plot error distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    for i, (ax, comp) in enumerate(zip(axes.flat[:3], components)):
        sns.histplot(velocity_errors[:, i], ax=ax, bins=50)
        ax.set_title(f'{comp} Velocity Error Distribution')
        ax.set_xlabel('Absolute Error')
    
    sns.histplot(pressure_errors, ax=axes[1, 1], bins=50)
    axes[1, 1].set_title('Pressure Error Distribution')
    axes[1, 1].set_xlabel('Absolute Error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distributions.png'))
    plt.close()
    
    # Create 3D visualization of largest errors
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with largest velocity errors
    total_vel_error = np.linalg.norm(velocity_errors, axis=1)
    error_threshold = np.percentile(total_vel_error, 90)  # Top 10% of errors
    high_error_points = total_vel_error > error_threshold
    
    scatter = ax.scatter(fluid_points_norm['x'][high_error_points], 
                        fluid_points_norm['y'][high_error_points], 
                        fluid_points_norm['z'][high_error_points],
                        c=total_vel_error[high_error_points],
                        cmap='hot',
                        s=100)
    
    # Plot building points
    building_points_norm = df_normalized[df_normalized['bm'] == 1]
    ax.scatter(building_points_norm['x'], 
              building_points_norm['y'], 
              building_points_norm['z'],
              color='gray', alpha=0.3, s=50)
    
    plt.colorbar(scatter, label='Velocity Error Magnitude')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('High Error Regions (Top 10%)')
    
    plt.savefig(os.path.join(save_dir, '3d_error_visualization.png'))
    plt.close()
    
    # Add boundary flow analysis
    print("\nAnalyzing flow near building boundaries...")
    boundary_points = find_boundary_points(df_normalized, grid_size)
    boundary_analysis = analyze_boundary_flow(
        fluid_points_norm, true_velocities, pred_velocities,
        boundary_points, grid_size, save_dir
    )
    
    return {
        'velocity_errors': velocity_errors,
        'pressure_errors': pressure_errors,
        'true_velocities': true_velocities,
        'pred_velocities': pred_velocities,
        'true_pressures': true_pressures,
        'pred_pressures': pred_pressures
    }

def find_boundary_points(df_normalized, grid_size):
    """Find fluid points adjacent to building boundaries"""
    building_points = set()
    for _, row in df_normalized[df_normalized['bm'] == 1].iterrows():
        building_points.add((int(row['x']), int(row['y']), int(row['z'])))
    
    # Find fluid points adjacent to buildings
    boundary_adjacent = set()
    directions = [
        (1,0,0), (-1,0,0),  # x direction
        (0,1,0), (0,-1,0),  # y direction
        (0,0,1), (0,0,-1)   # z direction
    ]
    
    for x, y, z in building_points:
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < grid_size[0] and 
                0 <= ny < grid_size[1] and 
                0 <= nz < grid_size[2]):
                point = (nx, ny, nz)
                if point not in building_points:
                    boundary_adjacent.add(point)
    
    return boundary_adjacent

def analyze_boundary_flow(fluid_points_norm, true_velocities, pred_velocities, 
                        boundary_points, grid_size, save_dir):
    """Analyze flow characteristics near building boundaries"""
    boundary_indices = []
    flow_angles = []
    errors = []
    
    # First find all building points
    building_points_set = set()
    for point in boundary_points:
        building_points_set.add(point)
    
    for i in range(len(fluid_points_norm)):
        point = (int(fluid_points_norm['x'].iloc[i]), 
                int(fluid_points_norm['y'].iloc[i]), 
                int(fluid_points_norm['z'].iloc[i]))
        
        if point in boundary_points:
            # Calculate flow angle relative to nearest building surface
            true_vel = true_velocities[i]
            pred_vel = pred_velocities[i]
            
            # Skip points with zero velocity
            if np.linalg.norm(true_vel) < 1e-6:
                continue
                
            # Find nearest building direction
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nx, ny, nz = point[0] + dx, point[1] + dy, point[2] + dz
                if (nx, ny, nz) in building_points_set:
                    normal = np.array([dx, dy, dz])
                    # Calculate angle between flow and surface normal
                    cos_angle = np.abs(np.dot(true_vel, normal) / 
                                     (np.linalg.norm(true_vel) * np.linalg.norm(normal)))
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                    
                    boundary_indices.append(i)
                    flow_angles.append(angle)
                    errors.append(np.linalg.norm(true_vel - pred_vel))
                    break
    
    if not errors:
        print("No boundary points found!")
        return
    
    print(f"\nFound {len(boundary_indices)} boundary points with valid flow")
    
    # Convert to numpy arrays for analysis
    flow_angles = np.array(flow_angles)
    errors = np.array(errors)
    
    # Analyze relationship between flow angle and error
    plt.figure(figsize=(10, 6))
    plt.scatter(flow_angles, errors, alpha=0.5)
    plt.xlabel('Flow Angle to Surface (degrees)')
    plt.ylabel('Prediction Error Magnitude')
    plt.title('Error vs Flow Angle near Boundaries')
    plt.grid(True)
    
    # Add trend line
    z = np.polyfit(flow_angles, errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(flow_angles), max(flow_angles), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, 
             label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'boundary_flow_analysis.png'))
    plt.close()
    
    # Print statistics
    print("\nBoundary Flow Analysis:")
    print("-" * 50)
    print(f"Number of boundary points: {len(boundary_indices)}")
    print("\nError by flow angle:")
    angle_ranges = [(0,30), (30,60), (60,90)]
    for start, end in angle_ranges:
        mask = (flow_angles >= start) & (flow_angles < end)
        if np.any(mask):
            mean_error = np.mean(errors[mask])
            std_error = np.std(errors[mask])
            count = np.sum(mask)
            print(f"  {start:2d}-{end:2d} degrees: {mean_error:.4f} ± {std_error:.4f} ({count} points)")
    
    # Create 3D visualization of boundary errors
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all boundary points colored by error
    scatter = ax.scatter(
        [fluid_points_norm['x'].iloc[i] for i in boundary_indices],
        [fluid_points_norm['y'].iloc[i] for i in boundary_indices],
        [fluid_points_norm['z'].iloc[i] for i in boundary_indices],
        c=errors,
        cmap='hot',
        s=100
    )
    
    plt.colorbar(scatter, label='Error Magnitude')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Boundary Point Errors')
    
    plt.savefig(os.path.join(save_dir, 'boundary_errors_3d.png'))
    plt.close()
    
    # Additional analysis: Correlation between angle and error
    correlation = np.corrcoef(flow_angles, errors)[0,1]
    print(f"\nCorrelation between flow angle and error: {correlation:.4f}")
    
    # Analyze high error points
    error_threshold = np.percentile(errors, 90)
    high_error_mask = errors > error_threshold
    if np.any(high_error_mask):
        print("\nHigh Error Analysis (top 10%):")
        print(f"Mean angle: {np.mean(flow_angles[high_error_mask]):.2f}°")
        print(f"Mean error: {np.mean(errors[high_error_mask]):.4f}")
    
    return boundary_indices, flow_angles, errors

if __name__ == "__main__":
    # Check for model file
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)
    
    # Analyze specific file
    csv_path = "filtered_dataset/-25_0_-17_8_3_13_result_preprocessed_50.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        exit(1)
    
    # Run analysis
    print(f"\nAnalyzing file: {os.path.basename(csv_path)}")
    results = analyze_predictions(csv_path, model_path)
    
    if results is not None:
        print("\nAnalysis complete! Results saved in 'analysis_results' directory.")
        print("Generated files:")
        print("- velocity_comparison.png")
        print("- pressure_comparison.png")
        print("- error_distributions.png")
        print("- 3d_error_visualization.png")
        print("- boundary_flow_analysis.png")
        print("- boundary_errors_3d.png")