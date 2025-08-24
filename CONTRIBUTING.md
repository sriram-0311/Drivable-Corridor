# Contributing to Drivable Corridor Detection

Thank you for your interest in contributing to this project! This guide will help you get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sriram-0311/Drivable-Corridor.git
   cd Drivable-Corridor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

## Project Structure

```
Drivable-Corridor/
├── src/drivable_corridor/    # Main package
│   ├── models/              # Neural network architectures
│   ├── data/               # Data loading and preprocessing
│   └── utils/              # Utility functions
├── scripts/                # Training and inference scripts
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks for experiments
├── models/                 # Saved model checkpoints
└── data/                   # Dataset storage
```

## Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add type hints where appropriate
   - Include docstrings for new functions/classes

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Format code**
   ```bash
   black src/ scripts/ tests/
   isort src/ scripts/ tests/
   ```

## Areas for Contribution

- **BEV Transformer implementation**
- **Multi-camera fusion techniques**
- **Performance optimizations**
- **Additional loss functions**
- **Documentation improvements**
- **Unit test coverage**

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation if needed
3. Create a pull request with a clear description
4. Link any relevant issues

## Questions?

Feel free to open an issue for any questions or suggestions!
