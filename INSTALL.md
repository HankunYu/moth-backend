# Installation Guide

## System Requirements

- Python 3.8 or higher
- Operating System: macOS, Linux, or Windows

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd moth-backend
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv moth-env

# Activate virtual environment
# On macOS/Linux:
source moth-env/bin/activate
# On Windows:
moth-env\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install base requirements
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

### 4. Create Required Directories
```bash
mkdir -p example
mkdir -p offspring_textures
mkdir -p default_textures
```

### 5. Configuration
The system uses `config.json` for configuration. Default settings are already provided, but you can modify:

- `min_moths`: Minimum number of moths (default: 5)
- `max_moths`: Maximum number of moths (default: 200)
- `base_lifespan_seconds`: Base moth lifespan (default: 300 seconds)

## Quick Start

1. **Start the moth controller:**
   ```bash
   python moth_controller.py
   ```

2. **Create test directories with textures:**
   ```bash
   mkdir -p example/test1/mothtexture
   # Add your texture file: example/test1/mothtexture/generated_textures_000_2000.png
   ```

3. **Use interactive commands:**
   - `status` - View moth population status
   - `list` - List all moths with details
   - `mate <id1> <id2>` - Mate two moths
   - `kill <id>` - Kill a specific moth
   - `quit` - Exit

## Troubleshooting

### Common Issues:

1. **OpenCV installation issues:**
   ```bash
   pip install --upgrade pip
   pip install opencv-python-headless  # Use headless version if GUI issues
   ```

2. **Permission errors:**
   ```bash
   # Make sure you have write permissions to the project directory
   chmod -R 755 .
   ```

3. **Python version compatibility:**
   ```bash
   # Check Python version
   python --version
   # Should be 3.8 or higher
   ```

## Development Setup

For contributors and developers:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests (when available)
pytest

# Format code
black .
isort .

# Type checking
mypy moth_controller.py
```