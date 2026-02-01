# CIFAR-10 Classification with ResNet18
## Lab 2 Worksheet - 31 Jan 2026

This repository contains a complete implementation of a CNN-based image classification system for the CIFAR-10 dataset with comprehensive analysis and visualizations.

## Project Overview

This project implements:
1. **Custom CNN Model**: ResNet18 architecture adapted for CIFAR-10
2. **Custom DataLoader**: CIFAR-10 dataloader with data augmentation
3. **FLOPs Counter**: Computational complexity analysis
4. **Training Pipeline**: 25-30 epoch training with monitoring
5. **Gradient Flow Visualization**: Real-time gradient tracking
6. **Weight Update Visualization**: Weight change monitoring
7. **Wandb Integration**: All metrics and visualizations logged to Wandb

## Project Structure

```
Sonam_Sikarwar_B23CM1060_lab2_Worksheet/
├── model.py              # ResNet18 model architecture
├── dataloader.py         # Custom CIFAR-10 dataloader
├── flops_counter.py      # FLOPs calculation utilities
├── visualization.py      # Gradient and weight visualization
├── train.py             # Main training script
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── report          # Detailed findings and observations
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU

### Setup

1. Clone the repository:
```bash
git clone -b Sonam_Sikarwar_B23CM1060_lab2_Worksheet \
  https://github.com/isonam16/MLOps_Sonam_Sikarwar_B23CM1060.git
cd Sonam_Sikarwar_B23CM1060_lab2_Worksheet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Login to Wandb:
```bash
wandb login
```

## Usage

### Basic Training

Run training with default parameters:
```bash
python train.py
```

### Custom Parameters

Train with custom configuration:
```bash
python train.py \
    --batch-size 64 \
    --epochs 25 \
    --lr 0.15 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --wandb-project "cifar10-lab2" \
    --wandb-name "my-experiment"
```

### Without Wandb

To train without Wandb logging:
```bash
python train.py --no-wandb
```

## Key Features

### 1. Model Architecture (model.py)
- **ResNet18**: Adapted for CIFAR-10 (32x32 images)
- **Parameters**: ~11M trainable parameters
- **Skip Connections**: Residual blocks for better gradient flow
- **Batch Normalization**: Improved training stability

### 2. Custom DataLoader (dataloader.py)
- **Data Augmentation**: 
  - Random cropping with padding
  - Random horizontal flipping
  - Color jittering
- **Normalization**: Channel-wise mean and std normalization
- **Efficient Loading**: Multi-worker support with pin memory

### 3. FLOPs Counter (flops_counter.py)
- **Layer-wise Analysis**: FLOPs breakdown by layer
- **Total Computation**: ~1.1 GFLOPs per forward pass
- **Optimization Insights**: Identify computational bottlenecks

### 4. Gradient Flow Visualization (visualization.py)
- **Real-time Tracking**: Monitor gradient magnitudes during training
- **Vanishing/Exploding Detection**: Early warning system
- **Layer-wise Analysis**: Identify problematic layers
- **Logged to Wandb**: Every 5 epochs

### 5. Weight Update Visualization (visualization.py)
- **Weight Change Tracking**: Monitor parameter updates
- **Distribution Analysis**: Weight magnitude and variance
- **Learning Dynamics**: Visualize training progress
- **Logged to Wandb**: Every 5 epochs

## Results


## Outputs

The training process generates:
1. **Wandb Logs**: Real-time metrics and visualizations
2. **Gradient Flow Plots**: Saved every 5 epochs
3. **Weight Update Plots**: Saved every 5 epochs
4. **Training Curves**: Loss and accuracy over time

## Analysis Components

### FLOPs Analysis
- Total FLOPs for forward pass
- Layer-wise computational breakdown
- Identification of expensive operations

### Gradient Flow Analysis
- Mean and max gradient per layer
- Gradient distribution over training
- Detection of vanishing/exploding gradients

### Weight Update Analysis
- Weight change magnitude over time
- Layer-wise update patterns
- Learning rate effectiveness

## Report

See report for detailed findings and observations including:
- Model performance analysis
- FLOPs breakdown and insights
- Gradient flow observations
- Weight update patterns
- Training dynamics
- Conclusions and recommendations

## Technical Details

### Hyperparameters
- **Batch Size**: 64
- **Learning Rate**: 0.15 (initial)
- **Optimizer**: SGD with momentum (0.9)
- **Weight Decay**: 5e-4
- **LR Scheduler**: Cosine Annealing
- **Epochs**: 25

### Data Augmentation
- Random Crop (32x32 with padding=4)
- Random Horizontal Flip (p=0.5)
- Color Jitter (brightness, contrast, saturation ±0.2)
- Normalization (CIFAR-10 statistics)

### Monitoring
- Gradient norms logged every 100 batches
- Weight updates tracked every 100 batches
- Visualizations generated every 5 epochs
- Metrics logged to Wandb in real-time
