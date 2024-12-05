import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import glob
from tqdm import tqdm
import concurrent.futures

class FlowDataset(Dataset):
    def __init__(self, data_dir, grid_size=(24, 24, 8), transform=None):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.transform = transform
        self.file_list = glob.glob(os.path.join(data_dir, "*.csv"))
        
        # Fit scalers on all data
        print("Fitting scalers on all data...")
        self._fit_scalers()
        
        # Process files in parallel but keep on CPU
        self.preprocessed_data = self._parallel_preprocess()
        
    def _fit_scalers(self):
        """Fit scalers on all data points"""
        coords, velocities, pressures = [], [], []
        
        # Process files in parallel for fitting scalers
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for file_path in self.file_list:
                futures.append(executor.submit(pd.read_csv, file_path))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(self.file_list), desc="Fitting scalers"):
                df = future.result()
                fluid_points = df[df['bm'] == 0]
                if not fluid_points.empty:
                    coords.append(fluid_points[['x', 'y', 'z']].values)
                    velocities.append(fluid_points[['u', 'v', 'w']].values)
                    pressures.append(fluid_points[['p']].values)
        
        # Initialize and fit scalers
        self.coord_scaler = StandardScaler().fit(np.concatenate(coords))
        self.velocity_scaler = StandardScaler().fit(np.concatenate(velocities))
        self.pressure_scaler = StandardScaler().fit(np.concatenate(pressures))
    
    def _process_chunk(self, file_paths):
        """Process a chunk of files"""
        chunk_data = []
        
        for file_path in file_paths:
            # Initialize grids
            input_grid = np.zeros((*self.grid_size, 4))
            output_grid = np.zeros((*self.grid_size, 4))
            fluid_mask = np.zeros(self.grid_size)
            building_mask = np.zeros(self.grid_size)
            
            # Read and process file
            df = pd.read_csv(file_path)
            
            # Process points efficiently using numpy operations
            coords = df[['x', 'y', 'z']].values
            valid_mask = (coords >= 0).all(axis=1) & (coords < [self.grid_size[0], self.grid_size[1], self.grid_size[2]]).all(axis=1)
            
            coords = coords[valid_mask].astype(int)
            bm = df['bm'].values[valid_mask]
            
            # Handle fluid points (bm == 0)
            fluid_idx = bm == 0
            if np.any(fluid_idx):
                fluid_coords = coords[fluid_idx]
                
                # Scale coordinates
                coords_scaled = self.coord_scaler.transform(fluid_coords.astype(float))
                
                # Scale velocities and pressure
                velocities = df.loc[valid_mask][['u', 'v', 'w']].values[fluid_idx]
                pressure = df.loc[valid_mask][['p']].values[fluid_idx]
                
                vel_scaled = self.velocity_scaler.transform(velocities)
                p_scaled = self.pressure_scaler.transform(pressure)
                
                # Assign to grids
                input_grid[fluid_coords[:, 0], fluid_coords[:, 1], fluid_coords[:, 2]] = np.column_stack([coords_scaled, np.zeros(len(coords_scaled))])
                output_grid[fluid_coords[:, 0], fluid_coords[:, 1], fluid_coords[:, 2]] = np.column_stack([vel_scaled, p_scaled])
                fluid_mask[fluid_coords[:, 0], fluid_coords[:, 1], fluid_coords[:, 2]] = 1
            
            # Handle building points (bm == 1)
            building_idx = bm == 1
            if np.any(building_idx):
                building_coords = coords[building_idx]
                coords_scaled = self.coord_scaler.transform(building_coords.astype(float))
                
                input_grid[building_coords[:, 0], building_coords[:, 1], building_coords[:, 2]] = np.column_stack([coords_scaled, np.ones(len(coords_scaled))])
                building_mask[building_coords[:, 0], building_coords[:, 1], building_coords[:, 2]] = 1
            
            # Convert to tensors (keep on CPU)
            data = {
                'input': torch.FloatTensor(input_grid),
                'target': torch.FloatTensor(output_grid),
                'fluid_mask': torch.FloatTensor(fluid_mask),
                'building_mask': torch.FloatTensor(building_mask)
            }
            chunk_data.append(data)
            
        return chunk_data
    
    def _parallel_preprocess(self):
        """Preprocess files in parallel"""
        print("Preprocessing files in parallel...")
        preprocessed = []
        
        # Process in chunks
        chunk_size = 32
        chunks = [self.file_list[i:i + chunk_size] for i in range(0, len(self.file_list), chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self._process_chunk, chunk))
            
            with tqdm(total=len(self.file_list), desc="Preprocessing") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    chunk_data = future.result()
                    preprocessed.extend(chunk_data)
                    pbar.update(len(chunk_data))
        
        return preprocessed
    
    def __len__(self):
        return len(self.preprocessed_data)
    
    def __getitem__(self, idx):
        # Return data directly (will be moved to GPU in training loop)
        data = self.preprocessed_data[idx].copy()
        data['scenario_idx'] = idx
        return data

def create_dataloader(data_dir, batch_size=16, num_workers=0, shuffle=True):
    """Create dataloader with optimized settings"""
    dataset = FlowDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, dataset 