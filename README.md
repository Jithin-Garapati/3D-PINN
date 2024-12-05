# 3D Flow Field Prediction with Fourier Neural Operator

This project implements a Fourier Neural Operator (FNO) model for predicting 3D flow fields around buildings in an urban environment. The model takes coordinates (x, y, z) as input and predicts velocity components (u, v, w) and pressure (p) while handling solid boundaries.

## Data Structure

The dataset consists of CSV files containing 3D flow data with the following properties:
- Grid size: 23x23x8 (4,232 points per file)
- Features per point:
  - Coordinates (x, y, z)
  - Velocities (u, v, w)
  - Pressure (p)
  - Boundary marker (bm): 0 for fluid, 1 for solid

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your CSV data files in the `filtered_dataset` directory.

## Training

To train the model:
```bash
python train.py
```

Training configuration can be modified in `train.py`:
- `batch_size`: Number of samples per batch
- `epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `modes`: Fourier modes for each dimension
- `width`: Width of the FNO model

The model automatically:
- Handles solid boundaries (bm=1)
- Normalizes input and output features
- Saves the best model and training curves
- Implements learning rate scheduling

## Model Architecture

The FNO model consists of:
- 3D Fourier layers for spectral convolution
- Linear layers for lifting and projection
- Batch normalization and GELU activation
- Skip connections

## Prediction

To use the trained model for predictions:
```python
from train import predict

# Load model and make predictions
predictions = predict(
    model_path='models/best_model.pth',
    input_data=your_input_data
)
```

## Output

The model saves:
- Best model weights: `models/best_model.pth`
- Training loss curve: `models/training_curve.png`
- Scalers for input/output normalization (saved with model) 