# Drivable Corridor Detection for Autonomous Vehicles

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

A PyTorch implementation of CNN-based drivable corridor detection inspired by Thomas Michalke's research on autonomous vehicle perception. This project demonstrates deep learning techniques for road segmentation using the Berkeley DeepDrive (BDD100K) dataset.

## 📋 Overview

This project implements a classic computer vision approach to drivable area segmentation, a critical component for autonomous vehicle navigation. The system uses convolutional neural networks to identify safe driving corridors from single camera images, providing pixel-level semantic segmentation of drivable vs non-drivable regions.

### Key Features

- **U-Net inspired architecture** with encoder-decoder design and skip connections
- **Multi-scale feature extraction** using progressively larger receptive fields
- **Custom loss functions** including BCE, Dice Loss, and IoU metrics
- **Berkeley DeepDrive (BDD100K) dataset integration** with custom data loaders
- **Comprehensive training pipeline** with validation, checkpointing, and Weights & Biases logging
- **Real-time inference capabilities** for deployment scenarios

## 🛠️ Architecture

### Core CNN Model (`models/cnn.py`)
- **Encoder**: 10 convolutional layers with max pooling for feature extraction
- **Decoder**: 4 transpose convolutional layers with skip connections for upsampling
- **Regularization**: Batch normalization, dropout (0.2), and ReLU activations
- **Output**: Single-channel probability map for drivable area classification

### Advanced Architecture (`models/bisenet.py`) - In Development
Implementing BiSeNet-inspired dual-path architecture:
- **Spatial Path**: Preserves high-resolution spatial details
- **Context Path**: Leverages ResNet50 backbone for semantic context
- **Feature Fusion Network (FFN)**: Combines spatial and contextual features
- **Attention Refinement Module (ARM)**: Enhances feature representation

## 📊 Dataset & Performance

**Dataset**: Berkeley DeepDrive (BDD100K)
- 100K+ diverse driving scenarios
- Multi-weather and lighting conditions
- Pixel-wise drivable area annotations

**Metrics**:
- Binary Cross-Entropy Loss with Logits
- Intersection over Union (IoU) accuracy
- Dice coefficient for segmentation quality

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### Training
```bash
cd scripts
python train.py
```

### Inference
```bash
cd scripts
python inference.py
```

## 📁 Project Structure

```
Drivable-Corridor/
├── src/drivable_corridor/
│   ├── models/
│   │   ├── cnn.py              # Main U-Net style CNN architecture
│   │   └── bisenet.py          # Advanced BiSeNet implementation
│   ├── data/
│   │   └── dataloaders.py      # BDD100K dataset handling
│   └── utils/                  # Utility functions
├── scripts/
│   ├── train.py               # Training pipeline with W&B logging
│   └── inference.py           # Model evaluation and visualization
├── tests/                     # Unit tests and model tests
├── models/                    # Trained model checkpoints
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks for experiments
├── data/                      # Dataset storage
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
└── README.md
```

## 🔬 Technical Highlights

### Data Pipeline
- **Efficient data loading** with PyTorch DataLoader
- **Image preprocessing**: Grayscale conversion, resizing (640x360), normalization
- **Label processing**: Binary mask generation for drivable/non-drivable areas
- **Train/validation split** with subset sampling for faster experimentation

### Model Training
- **Optimizers**: SGD with momentum and Adam adaptive learning rate
- **Learning rate scheduling**: StepLR with gamma decay
- **Checkpointing**: Automatic model state saving for resumable training
- **Experiment tracking**: Weights & Biases integration for metrics visualization

### Loss Functions
- **BCE with Logits**: Numerically stable binary classification
- **Dice Loss**: Handles class imbalance in segmentation tasks
- **Custom IoU**: Intersection over Union for segmentation accuracy

## 🎯 Results

The model successfully segments drivable corridors with high precision, demonstrating:
- Robust performance across diverse lighting conditions
- Accurate boundary detection between road and non-road areas
- Real-time inference capabilities suitable for autonomous vehicle deployment

## 🔮 Future Work: Modern BEV Transformer Integration

### Planned Enhancements

#### Bird's Eye View (BEV) Transformation
- **Multi-camera fusion**: Integrate front, side, and rear camera inputs
- **3D spatial understanding**: Transform perspective view to BEV representation
- **Temporal consistency**: Leverage video sequences for smoother predictions

#### Transformer-Based Architecture
- **Vision Transformers (ViTs)**: Replace CNN backbone with transformer encoders
- **Cross-attention mechanisms**: Enable better multi-modal feature fusion
- **Spatial-temporal transformers**: Process sequential BEV representations

#### Advanced Perception Stack
- **3D object detection**: Extend beyond segmentation to full scene understanding
- **Occupancy prediction**: Predict future states of the driving environment
- **End-to-end learning**: Joint training of perception and planning modules

#### Dataset Expansion
- **nuScenes integration**: Leverage 3D annotations and multi-camera setup
- **Synthetic data**: Augment training with CARLA/AirSim simulated environments
- **Real-world deployment**: Validation on custom autonomous vehicle platform

### Research Directions
- **Uncertainty quantification**: Bayesian neural networks for safety-critical applications
- **Domain adaptation**: Robust performance across different geographical regions
- **Efficient architectures**: Mobile-friendly models for edge deployment
- **Multi-task learning**: Joint semantic segmentation, depth estimation, and motion prediction

## 🤝 Contributing

This project represents an evolution from classical computer vision to modern autonomous vehicle perception. Contributions focusing on:
- BEV transformer implementations
- Multi-camera fusion techniques
- Real-time optimization
- Benchmark evaluations

...are particularly welcome!

## 📚 References

- Thomas Michalke et al. - "Drivable Corridor Detection for Autonomous Vehicles"
- Berkeley DeepDrive Dataset (BDD100K)
- BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation
- Vision Transformers for Autonomous Driving Applications

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🏷️ Tags

`#autonomous-vehicles` `#computer-vision` `#deep-learning` `#pytorch` `#semantic-segmentation` `#bev-perception` `#transformer` `#bdd100k` `#road-detection` `#self-driving-cars`
