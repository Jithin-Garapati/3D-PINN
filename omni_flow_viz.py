import os
import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt, QTimer
import omni.kit.app
import omni.usd
import omni.kit.viewport
import carb

class OmniFlowVisualizer(QMainWindow):
    def __init__(self, flow_data_path):
        super().__init__()
        
        # Initialize Omniverse Kit
        self.kit = omni.kit.app.get_app()
        
        # Load flow data
        self.data = pd.read_csv(flow_data_path)
        self.setup_flow_field()
        
        # Initialize UI
        self.initUI()
        
        # Start Omniverse update loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # 60 FPS target
        
    def setup_flow_field(self):
        # Create USD stage
        self.stage = omni.usd.get_context().get_stage()
        
        # Create particle system
        self.particles_path = "/World/ParticleSystem"
        self.stage.DefinePrim(self.particles_path, "Points")
        
        # Create building geometry from bm=1 points
        building_points = self.data[self.data['bm'] == 1][['x', 'y', 'z']].values
        if len(building_points) > 0:
            building_mesh = self.stage.DefinePrim("/World/Building", "Mesh")
            # Set building points as vertices
            building_mesh.GetAttribute("points").Set(building_points)
            
    def initUI(self):
        self.setWindowTitle('Omniverse Flow Visualization')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create Omniverse viewport
        self.viewport = omni.kit.viewport.get_viewport_interface()
        viewport_widget = self.viewport.create_viewport_window(
            title="Flow Visualization",
            width=1100,
            height=700
        )
        layout.addWidget(viewport_widget)
        
        # Create control panel
        control_panel = QHBoxLayout()
        
        # Emission rate slider
        emission_layout = QVBoxLayout()
        emission_layout.addWidget(QLabel('Emission Rate:'))
        self.emission_slider = QSlider(Qt.Horizontal)
        self.emission_slider.setMinimum(1)
        self.emission_slider.setMaximum(50)
        self.emission_slider.setValue(20)
        emission_layout.addWidget(self.emission_slider)
        
        # Flow speed slider
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel('Flow Speed:'))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        speed_layout.addWidget(self.speed_slider)
        
        # Add layouts to control panel
        control_panel.addLayout(emission_layout)
        control_panel.addLayout(speed_layout)
        layout.addLayout(control_panel)
        
    def update(self):
        # Update particle system
        speed_factor = self.speed_slider.value() / 100.0
        emission_rate = self.emission_slider.value()
        
        # Emit new particles
        for _ in range(emission_rate):
            pos = np.array([
                0.1,  # x (inlet)
                np.random.uniform(0.3, 0.7),  # y
                np.random.uniform(0.1, 0.9)   # z
            ])
            
            # Get velocity at position
            x, y, z = pos
            velocity = self.get_interpolated_velocity(x, y, z)
            
            # Create particle in USD
            particle = self.stage.DefinePrim(
                f"{self.particles_path}/particle_{_}",
                "Sphere"
            )
            particle.GetAttribute("radius").Set(0.01)
            particle.GetAttribute("position").Set(pos)
            
            # Add velocity as custom attribute
            velocity_attr = particle.CreateAttribute(
                "velocity",
                Sdf.ValueTypeNames.Vector3f
            )
            velocity_attr.Set(velocity)
        
        # Update viewport
        self.viewport.update()
        
    def get_interpolated_velocity(self, x, y, z):
        # Find nearest data points and interpolate velocity
        x_idx = int(x * (self.nx - 1))
        y_idx = int(y * (self.ny - 1))
        z_idx = int(z * (self.nz - 1))
        
        # Get velocity from data
        point_data = self.data[
            (self.data['x'] == x_idx) &
            (self.data['y'] == y_idx) &
            (self.data['z'] == z_idx)
        ]
        
        if len(point_data) > 0:
            return np.array([
                point_data['u'].values[0],
                point_data['v'].values[0],
                point_data['w'].values[0]
            ])
        else:
            return np.zeros(3)

def main():
    # Initialize Omniverse Kit
    kit = omni.kit.app.get_app("Flow Visualization")
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create visualization window
    flow_viz = OmniFlowVisualizer('filtered_dataset/-25_0_-17_8_3_13_result_preprocessed_50.csv')
    flow_viz.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 