# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a single Python script implementing a genetic algorithm for optimal image blending using evolutionary mask optimization.

## Architecture

The codebase consists of a single main module:

- `mask.py`: Contains the `GeneticMaskBlender` class that implements a genetic algorithm to evolve optimal blending masks for combining two images

### Core Components

**GeneticMaskBlender Class** (`mask.py:8-188`):
- Evolutionary algorithm for image blending optimization
- Key methods:
  - `initialize_population()`: Creates random mask population
  - `fitness_function()`: Evaluates blend quality using SSIM and edge preservation
  - `tournament_selection()`: Parent selection mechanism
  - `crossover()`: Two-point crossover for mask breeding
  - `mutate()`: Random mutations for genetic diversity
  - `run()`: Main evolution loop
  - `visualize_results()`: Display optimization results

**Algorithm Parameters**:
- Mask resolution: Configurable grid size for optimization (default 20x20)
- Population size: Number of candidate masks per generation (default 15)
- Generations: Evolution iterations (default 30)
- Mutation rate: Probability of random changes (default 0.15)

**Fitness Evaluation** (`mask.py:49-65`):
- Combines SSIM (Structural Similarity Index) with both source images
- Edge preservation using Laplacian variance
- Weighted combination: 40% SSIM1 + 40% SSIM2 + 20% edge score

## Dependencies

The script requires these Python packages:
- `numpy`: Numerical operations and array handling
- `opencv-python` (`cv2`): Image loading, processing, and saving
- `scikit-image`: SSIM calculation for fitness evaluation
- `matplotlib`: Visualization of results and fitness progression

## Usage

**Basic execution**:
```bash
python mask.py
```

**Input/Output Structure**:
- Input images: `./inputs/` directory
- Output blended images: `./out_textures/` directory (created automatically)
- Processes all pairwise combinations of input images

**Customization**:
The genetic algorithm parameters can be adjusted in the main execution block (`mask.py:203-207`):
- `mask_resolution`: Lower values = faster evolution, higher values = finer detail
- `population_size`: More individuals = better exploration, slower execution
- `generations`: More generations = better optimization, longer runtime
- `mutation_rate`: Higher rates = more exploration, potential instability

## Development Notes

- No build system or package management files present
- Single-file architecture suitable for research/experimentation
- Visualization components can be enabled by uncommenting `blender.visualize_results()` call
- Mask saving functionality available but commented out (`mask.py:216`)