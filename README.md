# ANN-PSO-Hybrid-F21BC

Implements a multi-layer Artificial Neural Network and Particle Swarm Optimization entirely from scratch. The ANN is trained using PSO to solve a target problem, with experiments analyzing the impact of ANN and PSO hyperparameters.

## Project Overview

### Implementation Philosophy
This project implements ANN training **without backpropagation**. Instead, PSO treats all neural network parameters (weights, biases, and activation functions) as a search space, optimizing them through swarm intelligence. Each particle in the swarm represents a complete neural network configuration.

### Codebase Structure

```
ANN-PSO-Hybrid-F21BC/
├── ANN/
│   ├── ann.py              # MultiLayerANN class: forward pass implementation
│   ├── activations.py      # Activation functions (ReLU, Sigmoid, Tanh, etc.)
│   └── loss_functions.py   # MAE, MSE, RMSE loss functions
├── PSO/
│   └── pso.py              # ParticleSwarm class: velocity updates, topology
├── BRIDGE/
│   └── bridge.py           # Bridge class: converts ANN params ↔ PSO particles
├── Utility/
│   ├── data_handler.py     # Data loading and preprocessing
│   ├── visualizer.py       # Convergence plots and test result tracking
│   └── model_utils.py      # Prediction visualization utilities
├── main.py                 # Single experiment runner
├── run_experiments.py      # Multiple run aggregator with statistics
└── concrete_data.csv       # Dataset: concrete compressive strength
```

### How ANN-PSO Works

**1. Initialization (Bridge)**
- `Bridge.initialize_particles()` creates a swarm of neural networks
- Each particle is a flattened vector: `[weights | biases | activation_indices]`
- Weights initialized using Xavier/He initialization (σ = √(4/(n_in + n_out)))
- Activation functions encoded as discrete probability distributions when `DISCRETE_PSO=True`

**2. Optimization Loop (PSO)**
- PSO evaluates each particle by:
  - `Bridge.reconstruct_params()` → unflatten particle into ANN parameters
  - `MultiLayerANN.evaluate_with_params()` → compute predictions
  - Loss function → fitness value
- Velocity update rule combines:
  - **Inertia (α)**: Previous velocity momentum
  - **Cognitive (β)**: Personal best attraction
  - **Social (γ)**: Local neighborhood best attraction  
  - **Global (δ)**: Global swarm best attraction
- Particles explore within velocity bounds (`v_max`) and position constraints

**3. Evaluation (ANN)**
- Forward pass: `X → [Linear → Activation] × L → Output`
- No gradient computation required
- Supports dynamic architectures (depth, width)
- Activation functions optimized per hidden layer

**4. Discrete Variables**
- Activation function selection uses probabilistic sampling
- Probability vectors updated via PSO dynamics
- Argmax selection during evaluation maintains differentiability in PSO space

## Quick Start

### Run Main Experiment

```
python main.py
```

This will:
- Train an ANN using PSO optimization
- Generate visualizations in `_Test_Results/Test_X/`
- Print training and test loss metrics

## Configuration Parameters

### Architecture

```python
LAYERS = [8, 32, 1]  # [input, hidden_layer(s), output]
```

- First value (8): Input features
- Middle values: Hidden layer sizes (add more for deeper networks)
- Last value (1): Output dimension

**Examples:**
```python
LAYERS = [8, 16, 1]      # Single hidden layer, 16 neurons
LAYERS = [8, 64, 32, 1]  # Two hidden layers, 64 and 32 neurons
LAYERS = [8, 1]          # No hidden layers (linear)
```

### PSO Parameters

```python
NUM_PARTICLES = 30       # Swarm size (more = better exploration, slower)
NUM_ITERATIONS = 2000    # Training iterations
NUM_INFORMANTS = 6       # Local topology neighbors
LOSS_FUNCTION = 'mae'    # Options: 'mae', 'mse', 'rmse'
DISCRETE_PSO = True      # Discrete activation functions
```

### PSO Behavior Presets

Choose one by setting: `PSO_PARAMS = PSO_PARAMS_[PRESET]`

**Available Presets:**

```python
PSO_PARAMS_GLOBAL        # Pure global best
PSO_PARAMS_LOCAL         # Pure local best
PSO_PARAMS_HYBRID        # Balanced global + local
PSO_PARAMS_BASELINE      # Standard PSO (recommended)
```

**Manual Tuning:**
```python
PSO_PARAMS = {
    'alpha': 0.729,      # Inertia weight (momentum)
    'beta': 1.49445,     # Cognitive coefficient (personal best)
    'gamma': 1.49445,    # Social coefficient (local best)
    'delta': 0.0,        # Global coefficient (global best)
    'epsilon': 0.75      # Topology connectivity (0-1)
}
```

### Data Configuration

```python
data_handler.transform_data(
    path="concrete_data.csv",
    train_split=0.7,     # 70% train, 30% test
    random_seed=1        # Reproducibility
)
```

## Output

### Console Output
- Real-time loss every 20 iterations
- Final train/test metrics
- Model architecture summary

### Generated Files
Location: `_Test_Results/Test_X/`
- `convergence.png` - Loss over iterations
- `predictions.png` - Predicted vs actual values
- `report.txt` - Full experiment details

## Experiment Scripts

### Single Run: `main.py`
Trains one ANN-PSO model with specified parameters.

```powershell
python main.py
```

**Returns:** Dictionary with trained model, optimizer, losses, and predictions.

**Use Case:** Quick prototyping, parameter testing, single experiment execution.

### Multiple Runs: `run_experiments.py`
Executes `main.py` multiple times to compute statistical metrics.

```powershell
python run_experiments.py
```

**Configuration:**
```python
NUM_RUNS = 10                    # Number of independent runs
BASE_DIR = "_Convergence_Test_3" # Output directory name
```

**Features:**
- Runs N independent experiments with same hyperparameters
- Tracks best model across all runs (lowest test loss)
- Computes mean and standard deviation for:
  - Final training loss
  - Test loss
- Saves best model visualizations and aggregated statistics
- Generates `_test_summary.txt` with statistical results

**Output Structure:**
```
_Convergence_Test_3/
└── Test_1/
    ├── convergence.png
    ├── predictions.png
    ├── report.txt
    └── _test_summary.txt    # Mean ± Std for train/test loss
```

**Use Case:** Statistical validation, robustness testing, publication-ready results.

**Note:** Update `layers` parameter in `visualizer` creation (line ~59) to match `main.py` LAYERS config.

## Other Utilities

### Architecture Search: `ann_experiment.py`
```powershell
python ann_experiment.py
```
Parallel grid search across depth/width combinations. Configure `SEARCH_CONFIG` for custom ranges.

### Visualization: `vizplot.py`
```powershell
python vizplot.py
```
Generates heatmaps and line plots from grid search CSV results.

## Performance Tips

**For better accuracy:**
- Increase `NUM_PARTICLES` (50-100)
- Increase `NUM_ITERATIONS` (2000-5000)
- Use `LOSS_FUNCTION = 'mae'` for robust training

**For faster experiments:**
- Decrease `NUM_PARTICLES` (10-20)
- Decrease `NUM_ITERATIONS` (500-1000)
- Use smaller `LAYERS` architectures

**For exploration:**
- Use `PSO_PARAMS_HYBRID` for balanced search
- Increase `epsilon` (0.8-1.0) for more connectivity
- Increase `NUM_INFORMANTS` for information sharing
