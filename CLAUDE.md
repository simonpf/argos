# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Argos is an ML-based precipitation retrievals package for atmospheric science applications. It focuses on processing GOES satellite data and creating precipitation retrievals using machine learning techniques.

## Development Setup

### Environment Management
The project uses conda for environment management. **All code execution, testing, and development should be run within the argus conda environment.**

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate argos
```

**Important**: Always ensure the argos environment is activated before running any commands, tests, or code in this repository.

### Installation
```bash
# Install in development mode
pip install -e ".[dev]"
```

## Common Commands

### Development
```bash
# Run tests
pytest

# Code formatting
black .

# Import sorting
isort .

# Linting
flake8

# Type checking
mypy src/argos
```

### CLI Usage
The package provides a command-line interface via the `argos` command, built with Click:

```bash
# Show help
argos --help

# Extract GOES satellite data
argos goes <satellite> <year> <month> [options]

# Available satellites: goes16, goes18, goes19
# Examples:
argos goes goes16 2023 6                                    # Extract all days in June 2023
argos goes goes16 2023 6 --days 1 --days 2 --days 3       # Extract specific days
argos goes goes16 2023 6 --step 20 --output-path ./data   # Custom step and output directory
argos goes goes16 2023 6 --processes 4 --verbose          # Use 4 processes with verbose logging

# Show GOES command help
argos goes --help
```

## Architecture

### Core Components

1. **CLI Module** (`src/argos/cli/`):
   - `main.py`: Main CLI entry point with argument parsing
   - `goes.py`: GOES-specific data extraction commands
   - Entry point defined in pyproject.toml: `argos = "argos.cli:main"`

2. **Data Processing** (`src/argos/data/`):
   - `goes.py`: GOESObs class for satellite data extraction and processing
   - Uses pansat library for satellite data access
   - Uses satpy for scene processing
   - Integrates with pyresample for grid operations

3. **Grid Definitions** (`src/argos/grids.py`):
   - Standard grid definitions using pyresample
   - Default 0.025-degree global grid
   - Regular lat/lon grid creation utilities

4. **Channel Properties** (`src/argos/channel_properties.py`):
   - ChannelProperties dataclass for observation metadata
   - Polarization conversion utilities
   - Channel property definitions for GOES ABI

### Key Dependencies

- **Scientific computing**: numpy, scipy, pandas, scikit-learn
- **Geospatial**: xarray, netcdf4, cartopy, pyresample
- **Satellite data**: satpy, pansat
- **CLI framework**: click (>=8.0.0)
- **Development**: pytest, black, flake8, isort, mypy

### Data Processing Flow

1. **Data Extraction**: GOES satellite data is downloaded using pansat
2. **Scene Processing**: Raw data is processed using satpy Scene objects
3. **Grid Remapping**: Data is remapped to standard grids using pyresample
4. **Output**: Processed data is saved as NetCDF files

### Configuration

- **pyproject.toml**: Main project configuration, dependencies, tool settings
- **environment.yml**: Conda environment specification with additional satellite packages
- Tool configurations included for black (line length 88), isort (black profile), mypy (strict typing)

## Testing

Tests are located in the `tests/` directory. The project uses pytest with the following configuration:
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

## Code Style

- Line length: 88 characters (black)
- Import sorting: black profile (isort)
- Type hints: Required for all function definitions (mypy)
- Python version: 3.12+ required