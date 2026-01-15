##  Project Structure

```
.
├── config.py         # Hyperparameters and global simulation settings
├── env.py            # Heterogeneous edge computing environment (System Model)
├── sac_cnn_gru.py    # Main entry point: SAC algorithm, Network definition (CNN+GRU), and Training loop
└── README.md         # Project documentation
```

## Requirements

The code is implemented in Python 3.8+ using PyTorch. It is recommended to use a virtual environment (Conda or venv).

```
pip install numpy torch
```

## Usage

### 1. Configuration

You can modify the simulation parameters in `config.py` to test different scenarios:

- `EDGE_NODE_NUM_CPU` / `EDGE_NODE_NUM_GPU`: Adjust the ratio of heterogeneous nodes.
- `max_tasks` / `min_tasks`: Control the workload intensity (burstiness).
- `epoch`: Set the number of training episodes.

### 2. Run Training

To start the training process using the proposed SAC+CNN+GRU algorithm， simply:

```
python sac_cnn_gru.py
```
