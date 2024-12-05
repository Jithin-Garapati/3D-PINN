import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QSlider, QLabel, QPushButton, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import torch
from fno_model import FNO3D
import time
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals

# Add StandardScaler to safe globals for model loading
add_safe_globals([StandardScaler])

class FlowPredictor:
    def __init__(self, model_path, csv_path, grid_size=(24, 24, 8)):
        self.grid_size = grid_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load original data
        print(f"Loading original data from {csv_path}")
        self.original_data = pd.read_csv(csv_path)
        
        # Create 3D arrays for original velocities
        self.u = np.zeros(grid_size)
        self.v = np.zeros(grid_size)
        self.w = np.zeros(grid_size)
        self.bm = np.zeros(grid_size)
        
        # Fill arrays with original data
        print("Initializing velocity and building mask arrays...")
        for _, row in self.original_data.iterrows():
            x, y, z = int(row['x']), int(row['y']), int(row['z'])
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]:
                if not np.isnan(row['u']):
                    self.u[x, y, z] = row['u']
                    self.v[x, y, z] = row['v']
                    self.w[x, y, z] = row['w']
                self.bm[x, y, z] = row['bm']
        
        # Load model and scalers
        print(f"Loading model from {model_path}")
        try:
            checkpoint = torch.load(model_path, weights_only=True, map_location=self.device)
        except Exception as e:
            print("Warning: Failed to load with weights_only=True, falling back to full load")
            checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = FNO3D(
            modes1=8, modes2=8, modes3=4,
            width=32,
            in_channels=4,
            out_channels=4
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.coord_scaler = checkpoint['coord_scaler']
        self.velocity_scaler = checkpoint['velocity_scaler']
        self.pressure_scaler = checkpoint['pressure_scaler']
        
        # Create empty grid for model input
        self.input_grid = np.zeros((*grid_size, 4))
        
        # Initialize input grid
        self.initialize_grids()
        print("Initialization complete!")
        
        # Add interpolation for building mask
        self.interpolate_building_mask()
        
        # Store min/max velocities for normalization
        self.min_vel = float('inf')
        self.max_vel = float('-inf')
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    if self.bm[x, y, z] == 0:  # Only consider fluid points
                        vel_mag = np.sqrt(self.u[x,y,z]**2 + self.v[x,y,z]**2 + self.w[x,y,z]**2)
                        self.min_vel = min(self.min_vel, vel_mag)
                        self.max_vel = max(self.max_vel, vel_mag)
        
        print(f"Velocity range in original data: {self.min_vel:.3f} to {self.max_vel:.3f}")
    
    def initialize_grids(self):
        """Initialize input grid with scaled coordinates and building mask"""
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    # Scale coordinates
                    coords = np.array([[x/self.grid_size[0], y/self.grid_size[1], z/self.grid_size[2]]])
                    coords_scaled = self.coord_scaler.transform(coords)
                    self.input_grid[x, y, z, :3] = coords_scaled[0]
                    self.input_grid[x, y, z, 3] = self.bm[x, y, z]
    
    def interpolate_building_mask(self):
        """Create a finer building mask for better collision detection"""
        self.fine_grid_factor = 4  # Subdivide each grid cell into 4x4x4
        fine_shape = tuple(s * self.fine_grid_factor for s in self.grid_size)
        self.fine_bm = np.zeros(fine_shape)
        
        # Interpolate building mask to finer grid
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    if self.bm[x, y, z] == 1:
                        # Fill the corresponding fine grid cells
                        x_start = x * self.fine_grid_factor
                        y_start = y * self.fine_grid_factor
                        z_start = z * self.fine_grid_factor
                        self.fine_bm[x_start:x_start + self.fine_grid_factor,
                                   y_start:y_start + self.fine_grid_factor,
                                   z_start:z_start + self.fine_grid_factor] = 1
    
    def check_building_collision(self, pos, next_pos):
        """Check if path between pos and next_pos intersects with building"""
        # Convert positions to fine grid coordinates
        pos_fine = np.array([
            int(pos[0] * (self.grid_size[0] * self.fine_grid_factor - 1)),
            int(pos[1] * (self.grid_size[1] * self.fine_grid_factor - 1)),
            int(pos[2] * (self.grid_size[2] * self.fine_grid_factor - 1))
        ])
        next_pos_fine = np.array([
            int(next_pos[0] * (self.grid_size[0] * self.fine_grid_factor - 1)),
            int(next_pos[1] * (self.grid_size[1] * self.fine_grid_factor - 1)),
            int(next_pos[2] * (self.grid_size[2] * self.fine_grid_factor - 1))
        ])
        
        # Get points along the path
        path_points = self.get_line_points(pos_fine, next_pos_fine)
        
        # Check each point for collision
        for point in path_points:
            x, y, z = point
            if (0 <= x < self.fine_bm.shape[0] and 
                0 <= y < self.fine_bm.shape[1] and 
                0 <= z < self.fine_bm.shape[2]):
                if self.fine_bm[x, y, z] == 1:
                    return True
        return False
    
    def get_line_points(self, start, end):
        """Get all points along a 3D line using Bresenham's algorithm"""
        points = []
        x1, y1, z1 = start
        x2, y2, z2 = end
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1
        
        # Driving axis is X
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x1 != x2:
                points.append((x1, y1, z1))
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
        # Driving axis is Y
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y1 != y2:
                points.append((x1, y1, z1))
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
        # Driving axis is Z
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z1 != z2:
                points.append((x1, y1, z1))
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
        
        points.append((x2, y2, z2))
        return points
    
    def get_velocity(self, pos, use_original=False):
        """Get velocity at position, either from original data or model prediction"""
        # Convert to grid indices
        x = int(pos[0] * (self.grid_size[0] - 1))
        y = int(pos[1] * (self.grid_size[1] - 1))
        z = int(pos[2] * (self.grid_size[2] - 1))
        
        # Ensure within bounds
        x = max(0, min(x, self.grid_size[0] - 1))
        y = max(0, min(y, self.grid_size[1] - 1))
        z = max(0, min(z, self.grid_size[2] - 1))
        
        # Check if inside building
        if self.bm[x, y, z] == 1:
            return np.zeros(3)
        
        if use_original:
            velocity = np.array([
                self.u[x, y, z],
                self.v[x, y, z],
                self.w[x, y, z]
            ])
        else:
            # Get model prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(self.input_grid).unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                pred = output.cpu().squeeze().numpy()
            
            # Get velocity at position
            velocity_scaled = pred[x, y, z, :3]
            velocity = self.velocity_scaler.inverse_transform(velocity_scaled.reshape(1, -1))[0]
        
        # Print debug info occasionally
        if np.random.random() < 0.001:  # Print for ~0.1% of predictions
            vel_mag = np.linalg.norm(velocity)
            print(f"\nVelocity at pos {pos}:")
            print(f"{'Original' if use_original else 'Predicted'} velocity: {velocity}")
            print(f"Magnitude: {vel_mag:.3f}")
            if not use_original:
                orig_vel = np.array([self.u[x, y, z], self.v[x, y, z], self.w[x, y, z]])
                orig_mag = np.linalg.norm(orig_vel)
                print(f"Original velocity at same point: {orig_vel}")
                print(f"Original magnitude: {orig_mag:.3f}")
                print(f"Relative error: {abs(vel_mag - orig_mag)/orig_mag:.2%}")
        
        return velocity

class Particle:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.velocity = np.zeros(3)
        self.history = [pos.copy()]
        self.alive = True
        self.age = 0
        self.lifetime = 100
        self.collision_count = 0
        self.max_collisions = 3  # Maximum number of collisions before dying

    def update(self, velocity, dt=0.01, flow_predictor=None):
        if not self.alive:
            return

        self.velocity = velocity
        next_pos = self.pos + velocity * dt
        
        # Check for building collision if flow_predictor is provided
        if flow_predictor is not None:
            if flow_predictor.check_building_collision(self.pos, next_pos):
                self.collision_count += 1
                if self.collision_count >= self.max_collisions:
                    self.alive = False
                return
        
        self.pos = next_pos
        self.history.append(self.pos.copy())
        
        if len(self.history) > 20:
            self.history.pop(0)
            
        self.age += 1
        if self.age > self.lifetime:
            self.alive = False

class FlowVisualizer(QMainWindow):
    def __init__(self, flow_predictor):
        super().__init__()
        self.flow_predictor = flow_predictor
        self.particles = []
        self.max_particles = 300
        self.view_elevation = 20
        self.view_azimuth = 45
        self.is_rotating = False
        self.last_mouse_pos = None
        self.rotation_speed = 5
        self.use_original = False
        self.zoom_level = 1.0
        self.zoom_speed = 0.1
        self.show_velocity_info = True
        self.velocity_text = None
        
        # Emission parameters
        self.emission_x = 0.1  # Starting x position
        self.emission_y_min = 0.2  # Minimum y position
        self.emission_y_max = 0.8  # Maximum y position
        self.emission_z_min = 0.1  # Minimum z position
        self.emission_z_max = 0.9  # Maximum z position
        self.emission_layers = 3  # Number of emission layers in x direction
        self.emission_spacing = 0.05  # Spacing between emission layers
        
        # Set default font
        self.default_font = QFont('Arial', 10)
        QApplication.setFont(self.default_font)
        
        # Initialize plot elements
        self.building_artists = None
        self.particle_artists = []
        
        self.initUI()
        
    def create_rotation_controls(self):
        rotation_group = QWidget()
        rotation_layout = QGridLayout()
        
        # Create rotation buttons with arrows
        self.up_button = QPushButton('↑')
        self.down_button = QPushButton('↓')
        self.left_button = QPushButton('←')
        self.right_button = QPushButton('→')
        self.clockwise_button = QPushButton('↻')
        self.counterclockwise_button = QPushButton('↺')
        self.reset_view_button = QPushButton('Reset View')
        
        # Style buttons
        button_style = """
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 8px;
                font-size: 16px;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:pressed {
                background-color: #454545;
            }
        """
        for button in [self.up_button, self.down_button, self.left_button, 
                      self.right_button, self.clockwise_button, 
                      self.counterclockwise_button, self.reset_view_button]:
            button.setStyleSheet(button_style)
        
        # Add buttons to grid
        rotation_layout.addWidget(self.up_button, 0, 1)
        rotation_layout.addWidget(self.left_button, 1, 0)
        rotation_layout.addWidget(self.right_button, 1, 2)
        rotation_layout.addWidget(self.down_button, 2, 1)
        rotation_layout.addWidget(self.clockwise_button, 1, 3)
        rotation_layout.addWidget(self.counterclockwise_button, 1, 4)
        rotation_layout.addWidget(self.reset_view_button, 1, 1)
        
        # Connect button signals
        self.up_button.clicked.connect(lambda: self.rotate_view(elevation=self.rotation_speed))
        self.down_button.clicked.connect(lambda: self.rotate_view(elevation=-self.rotation_speed))
        self.left_button.clicked.connect(lambda: self.rotate_view(azimuth=-self.rotation_speed))
        self.right_button.clicked.connect(lambda: self.rotate_view(azimuth=self.rotation_speed))
        self.clockwise_button.clicked.connect(lambda: self.rotate_view(roll=self.rotation_speed))
        self.counterclockwise_button.clicked.connect(lambda: self.rotate_view(roll=-self.rotation_speed))
        self.reset_view_button.clicked.connect(self.reset_view)
        
        # Add rotation speed slider
        speed_layout = QHBoxLayout()
        speed_label = QLabel('Rotation Speed:')
        speed_label.setStyleSheet('color: white;')
        self.rotation_speed_slider = QSlider(Qt.Horizontal)
        self.rotation_speed_slider.setMinimum(1)
        self.rotation_speed_slider.setMaximum(20)
        self.rotation_speed_slider.setValue(5)
        self.rotation_speed_slider.valueChanged.connect(self.update_rotation_speed)
        
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.rotation_speed_slider)
        
        # Create vertical layout for all rotation controls
        rotation_controls = QVBoxLayout()
        rotation_controls.addLayout(rotation_layout)
        rotation_controls.addLayout(speed_layout)
        
        rotation_group.setLayout(rotation_controls)
        return rotation_group
        
    def create_data_toggle(self):
        toggle_layout = QHBoxLayout()
        
        # Create toggle button for data source
        self.data_toggle = QPushButton('Using: Model Predictions')
        self.data_toggle.setCheckable(True)
        self.data_toggle.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:checked {
                background-color: #4a4a4a;
            }
        """)
        self.data_toggle.clicked.connect(self.toggle_data_source)
        
        # Add velocity info toggle
        self.velocity_toggle = QPushButton('Show Velocity Info')
        self.velocity_toggle.setCheckable(True)
        self.velocity_toggle.setChecked(True)
        self.velocity_toggle.setStyleSheet(self.data_toggle.styleSheet())
        self.velocity_toggle.clicked.connect(self.toggle_velocity_info)
        
        toggle_layout.addWidget(self.data_toggle)
        toggle_layout.addWidget(self.velocity_toggle)
        toggle_layout.addStretch()
        
        return toggle_layout
    
    def toggle_velocity_info(self):
        self.show_velocity_info = self.velocity_toggle.isChecked()
        if not self.show_velocity_info and self.velocity_text:
            self.velocity_text.remove()
            self.velocity_text = None
        self.canvas.draw()
    
    def toggle_data_source(self):
        self.use_original = self.data_toggle.isChecked()
        self.data_toggle.setText('Using: ' + ('Original Data' if self.use_original else 'Model Predictions'))

    def create_zoom_controls(self):
        zoom_group = QWidget()
        zoom_layout = QVBoxLayout()
        
        # Create zoom buttons
        buttons_layout = QHBoxLayout()
        
        self.zoom_in_button = QPushButton('🔍+')
        self.zoom_out_button = QPushButton('🔍-')
        self.reset_zoom_button = QPushButton('Reset Zoom')
        
        # Style buttons
        button_style = """
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 8px;
                font-size: 16px;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:pressed {
                background-color: #454545;
            }
        """
        
        for button in [self.zoom_in_button, self.zoom_out_button, self.reset_zoom_button]:
            button.setStyleSheet(button_style)
        
        # Connect button signals
        self.zoom_in_button.clicked.connect(lambda: self.zoom_view(1.1))
        self.zoom_out_button.clicked.connect(lambda: self.zoom_view(0.9))
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        
        # Add zoom speed slider
        speed_layout = QHBoxLayout()
        speed_label = QLabel('Zoom Speed:')
        speed_label.setStyleSheet('color: white;')
        self.zoom_speed_slider = QSlider(Qt.Horizontal)
        self.zoom_speed_slider.setMinimum(1)
        self.zoom_speed_slider.setMaximum(20)
        self.zoom_speed_slider.setValue(10)
        self.zoom_speed_slider.valueChanged.connect(self.update_zoom_speed)
        
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.zoom_speed_slider)
        
        # Add buttons to layout
        buttons_layout.addWidget(self.zoom_out_button)
        buttons_layout.addWidget(self.zoom_in_button)
        buttons_layout.addWidget(self.reset_zoom_button)
        
        zoom_layout.addLayout(buttons_layout)
        zoom_layout.addLayout(speed_layout)
        
        zoom_group.setLayout(zoom_layout)
        return zoom_group

    def update_zoom_speed(self):
        self.zoom_speed = self.zoom_speed_slider.value() / 100.0

    def zoom_view(self, factor):
        # Get current view limits
        x_center = np.mean(self.ax.get_xlim())
        y_center = np.mean(self.ax.get_ylim())
        z_center = np.mean(self.ax.get_zlim())
        
        # Calculate new ranges
        x_range = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) / 2
        y_range = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) / 2
        z_range = (self.ax.get_zlim()[1] - self.ax.get_zlim()[0]) / 2
        
        # Update limits around center
        self.ax.set_xlim(x_center - x_range * factor, x_center + x_range * factor)
        self.ax.set_ylim(y_center - y_range * factor, y_center + y_range * factor)
        self.ax.set_zlim(z_center - z_range * factor, z_center + z_range * factor)
        
        # Update axis scaling and appearance
        self.ax.set_box_aspect([1, 1, 1])
        
        # Scale the axis elements with zoom
        scale_factor = 1.0 / factor if factor > 1 else factor
        self.ax.tick_params(axis='x', labelsize=10 * scale_factor, pad=8 * scale_factor)
        self.ax.tick_params(axis='y', labelsize=10 * scale_factor, pad=8 * scale_factor)
        self.ax.tick_params(axis='z', labelsize=10 * scale_factor, pad=8 * scale_factor)
        
        # Scale labels
        self.ax.set_xlabel('X', color='white', fontsize=12 * scale_factor, labelpad=10 * scale_factor)
        self.ax.set_ylabel('Y', color='white', fontsize=12 * scale_factor, labelpad=10 * scale_factor)
        self.ax.set_zlabel('Z', color='white', fontsize=12 * scale_factor, labelpad=10 * scale_factor)
        
        # Update camera position to maintain proper perspective
        self.ax.dist = 7 + 3 * np.log(1/scale_factor)
        
        # Ensure building points stay on top
        if self.building_artists:
            self.building_artists.set_zorder(10)
        
        self.zoom_level *= factor
        self.canvas.draw()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(0, 1)
        
        # Reset axis scaling and appearance
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.tick_params(axis='x', labelsize=10, pad=8)
        self.ax.tick_params(axis='y', labelsize=10, pad=8)
        self.ax.tick_params(axis='z', labelsize=10, pad=8)
        
        self.ax.set_xlabel('X', color='white', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y', color='white', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z', color='white', fontsize=12, labelpad=10)
        
        # Reset camera distance
        self.ax.dist = 7
        
        self.canvas.draw()

    def on_scroll(self, event):
        if event.button == 'up':
            self.zoom_view(1 - self.zoom_speed)
        else:
            self.zoom_view(1 + self.zoom_speed)

    def create_emission_controls(self):
        emission_group = QWidget()
        emission_layout = QVBoxLayout()
        
        # Emission rate slider
        rate_layout = QHBoxLayout()
        rate_label = QLabel('Emission Rate:')
        rate_label.setStyleSheet('color: white;')
        self.emission_slider = QSlider(Qt.Horizontal)
        self.emission_slider.setMinimum(1)
        self.emission_slider.setMaximum(20)
        self.emission_slider.setValue(10)
        rate_layout.addWidget(rate_label)
        rate_layout.addWidget(self.emission_slider)
        
        # Flow speed slider
        speed_layout = QHBoxLayout()
        speed_label = QLabel('Flow Speed:')
        speed_label.setStyleSheet('color: white;')
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        
        # Width control slider
        width_layout = QHBoxLayout()
        width_label = QLabel('Emission Width:')
        width_label.setStyleSheet('color: white;')
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setMinimum(1)
        self.width_slider.setMaximum(10)
        self.width_slider.setValue(6)
        self.width_slider.valueChanged.connect(self.update_emission_width)
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.width_slider)
        
        # Layers control slider
        layers_layout = QHBoxLayout()
        layers_label = QLabel('Emission Layers:')
        layers_label.setStyleSheet('color: white;')
        self.layers_slider = QSlider(Qt.Horizontal)
        self.layers_slider.setMinimum(1)
        self.layers_slider.setMaximum(5)
        self.layers_slider.setValue(3)
        self.layers_slider.valueChanged.connect(self.update_emission_layers)
        layers_layout.addWidget(layers_label)
        layers_layout.addWidget(self.layers_slider)
        
        # Add all controls to layout
        emission_layout.addLayout(rate_layout)
        emission_layout.addLayout(speed_layout)
        emission_layout.addLayout(width_layout)
        emission_layout.addLayout(layers_layout)
        
        emission_group.setLayout(emission_layout)
        
        # Style the widget
        emission_group.setStyleSheet("""
            QWidget {
                background-color: black;
                color: white;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)
        
        return emission_group

    def update_emission_width(self):
        width_factor = self.width_slider.value() / 10.0
        self.emission_y_min = 0.5 - (0.3 * width_factor)
        self.emission_y_max = 0.5 + (0.3 * width_factor)
        self.emission_z_min = 0.5 - (0.4 * width_factor)
        self.emission_z_max = 0.5 + (0.4 * width_factor)

    def update_emission_layers(self):
        self.emission_layers = self.layers_slider.value()
        self.emission_spacing = 0.05 * (3 / self.emission_layers)  # Adjust spacing based on layers

    def initUI(self):
        self.setWindowTitle('Flow Visualization')
        self.setGeometry(100, 100, 1200, 800)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add data source toggle
        toggle_layout = self.create_data_toggle()
        layout.addLayout(toggle_layout)

        # Create controls container
        controls_container = QHBoxLayout()
        
        # Add rotation controls
        rotation_controls = self.create_rotation_controls()
        controls_container.addWidget(rotation_controls)
        
        # Add zoom controls
        zoom_controls = self.create_zoom_controls()
        controls_container.addWidget(zoom_controls)
        
        # Add emission controls
        emission_controls = self.create_emission_controls()
        controls_container.addWidget(emission_controls)
        
        layout.addLayout(controls_container)

        # Set up the plot
        self.setup_plot()
        
        # Create timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(50)  # 20 FPS

    def setup_plot(self):
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.figure.patch.set_facecolor('black')
        
        # Plot building points and store the artist with higher zorder
        building_points = np.where(self.flow_predictor.bm == 1)
        if len(building_points[0]) > 0:
            x_scaled = building_points[0] / (self.flow_predictor.grid_size[0] - 1)
            y_scaled = building_points[1] / (self.flow_predictor.grid_size[1] - 1)
            z_scaled = building_points[2] / (self.flow_predictor.grid_size[2] - 1)
            self.building_artists = self.ax.scatter(x_scaled, y_scaled, z_scaled,
                                                  c='white', alpha=0.8, marker='s', s=50,
                                                  zorder=10)  # Higher zorder to stay on top
        
        # Set initial view
        self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)
        
        # Set labels and limits with initial styling
        self.ax.set_xlabel('X', color='white', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y', color='white', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z', color='white', fontsize=12, labelpad=10)
        self.ax.tick_params(axis='x', colors='white', labelsize=10, pad=8)
        self.ax.tick_params(axis='y', colors='white', labelsize=10, pad=8)
        self.ax.tick_params(axis='z', colors='white', labelsize=10, pad=8)
        
        # Set axis limits and aspect
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(0, 1)
        self.ax.set_box_aspect([1, 1, 1])
        
        # Ensure grid is behind all plot elements
        self.ax.grid(False)
        
        # Set initial camera distance
        self.ax.dist = 7
        
        self.canvas.draw()

    def on_mouse_press(self, event):
        if event.button == 1:  # Left click
            self.is_rotating = True
            self.last_mouse_pos = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.button == 1:  # Left click
            self.is_rotating = False

    def on_mouse_move(self, event):
        if self.is_rotating and event.xdata is not None and event.ydata is not None:
            dx = event.xdata - self.last_mouse_pos[0]
            dy = event.ydata - self.last_mouse_pos[1]
            
            self.view_azimuth += dx * 2.0
            self.view_elevation = np.clip(self.view_elevation + dy * 2.0, -90, 90)
            
            self.last_mouse_pos = (event.xdata, event.ydata)
            
            # Update view without clearing
            self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)
            self.canvas.draw()

    def update_simulation(self):
        # Emit new particles
        if len(self.particles) < self.max_particles:
            particles_per_layer = self.emission_slider.value() // self.emission_layers
            if particles_per_layer < 1:
                particles_per_layer = 1
                
            for layer in range(self.emission_layers):
                x_pos = self.emission_x + (layer * self.emission_spacing)
                
                for _ in range(particles_per_layer):
                    if len(self.particles) >= self.max_particles:
                        break
                        
                    pos = np.array([
                        x_pos,
                        np.random.uniform(self.emission_y_min, self.emission_y_max),
                        np.random.uniform(self.emission_z_min, self.emission_z_max)
                    ])
                    self.particles.append(Particle(pos))

        # Update particles
        speed_factor = self.speed_slider.value() / 100.0
        new_particles = []
        
        for particle in self.particles:
            if particle.alive:
                velocity = self.flow_predictor.get_velocity(particle.pos, self.use_original)
                particle.update(velocity * speed_factor, flow_predictor=self.flow_predictor)
                
                if (0 <= particle.pos[0] <= 1 and 
                    0 <= particle.pos[1] <= 1 and 
                    0 <= particle.pos[2] <= 1):
                    new_particles.append(particle)
        
        self.particles = new_particles
        self.update_plot()

    def update_plot(self):
        # Remove old particle trails and text
        for artist in self.particle_artists:
            artist.remove()
        self.particle_artists.clear()
        
        if self.velocity_text:
            self.velocity_text.remove()
            self.velocity_text = None
        
        # Plot new particle trails
        max_vel = float('-inf')
        min_vel = float('inf')
        avg_vel = 0
        active_particles = 0
        
        for particle in self.particles:
            if len(particle.history) > 1:
                history = np.array(particle.history)
                velocity_mag = np.linalg.norm(particle.velocity)
                max_vel = max(max_vel, velocity_mag)
                min_vel = min(min_vel, velocity_mag)
                avg_vel += velocity_mag
                active_particles += 1
                
                # Use velocity magnitude for color
                normalized_vel = (velocity_mag - self.flow_predictor.min_vel) / (self.flow_predictor.max_vel - self.flow_predictor.min_vel)
                color = plt.cm.coolwarm(normalized_vel)
                
                line = self.ax.plot3D(history[:, 0], history[:, 1], history[:, 2],
                                    color=color, alpha=0.8, linewidth=2,
                                    zorder=5)[0]
                self.particle_artists.append(line)
        
        # Update velocity info text
        if self.show_velocity_info and active_particles > 0:
            avg_vel /= active_particles
            info_text = f"Velocity (m/s)\nMin: {min_vel:.2f}\nMax: {max_vel:.2f}\nAvg: {avg_vel:.2f}"
            self.velocity_text = self.ax.text2D(0.02, 0.98, info_text,
                                              transform=self.ax.transAxes,
                                              color='white',
                                              verticalalignment='top',
                                              fontsize=10)
        
        # Ensure building points stay on top
        if self.building_artists:
            self.building_artists.set_zorder(10)
        
        # Update the canvas
        self.canvas.draw()

    def rotate_view(self, elevation=0, azimuth=0, roll=0):
        self.view_elevation = np.clip(self.view_elevation + elevation, -90, 90)
        self.view_azimuth = (self.view_azimuth + azimuth) % 360
        
        self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)
        if roll != 0:
            # Rotate the up direction
            self.ax.set_box_aspect(None)  # Reset aspect to allow rotation
            current_up = self.ax.zaxis._axinfo['juggled'][0]
            new_up = (current_up + roll) % 360
            self.ax.zaxis._axinfo['juggled'] = (new_up, 2)
        
        self.canvas.draw()

    def update_rotation_speed(self):
        self.rotation_speed = self.rotation_speed_slider.value()

    def reset_view(self):
        self.view_elevation = 20
        self.view_azimuth = 45
        self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)
        self.ax.set_box_aspect(None)
        self.ax.zaxis._axinfo['juggled'] = (0, 2)  # Reset roll
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    
    # Initialize flow predictor with both model and original data
    flow_predictor = FlowPredictor(
        model_path='models/best_model.pth',
        csv_path='filtered_dataset/-25_0_-17_8_3_13_result_preprocessed_50.csv'
    )
    
    # Create visualization window
    window = FlowVisualizer(flow_predictor)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 