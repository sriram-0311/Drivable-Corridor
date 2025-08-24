# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-24

### Added
- Initial implementation of CNN-based drivable corridor detection
- U-Net inspired architecture with encoder-decoder design
- Berkeley DeepDrive (BDD100K) dataset integration
- Training pipeline with Weights & Biases logging
- Inference script for model evaluation
- BiSeNet architecture implementation (in development)
- Professional project structure following Python best practices
- Comprehensive documentation and README

### Changed
- Restructured project from "AI Corridor" to proper Python package layout
- Updated import statements for new module structure
- Improved gitignore for better repository hygiene

### Technical Features
- Binary Cross-Entropy loss with logits
- Dice loss for segmentation tasks
- IoU accuracy metrics
- Batch normalization and dropout regularization
- Skip connections for feature preservation
- Multi-scale feature extraction
