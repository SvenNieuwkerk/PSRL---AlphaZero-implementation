# RL Challenge – MCTS + Actor-Critic Exploration

**Status: Work in progress.**  
This repository contains experimental code for solving the exploration challenge using Monte Carlo Tree Search (MCTS) combined with a learned Actor-Critic policy. The codebase is functional for training, evaluation, and analysis, but is not fully cleaned up. Some notebooks and utilities may be outdated.

---

# Overview

The core idea is:

- Learn a policy and value network
- Use the learned policy to guide MCTS planning
- Train using self-generated trajectories
- Evaluate checkpoints across multiple seeds
- Analyze performance and inspect MCTS trees and agent behavior

Training and evaluation are managed via config-based experiment sweeps.

---

# Folder structure

## Core algorithm and models

### `MCTS.py`

Contains the base Monte Carlo Tree Search implementation.

Implements:

- Tree structure
- Node expansion
- Selection using UCB / PUCT
- Backup / value propagation
- Simulation logic

This is the unconstrained version.

---

### `MCTS_AC.py`

Action-constrained MCTS variant.

Adds support for:

- Progressive widening
- Action sampling constraints
- Policy-guided action selection
- Integration with Actor-Critic network

This is the version used for experiments.

---

### `network.py`

Defines the Actor-Critic neural network.

Contains:

- Policy head
- Value head
- Forward pass
- Network architecture definition

Network size and hyperparameters are controlled via experiment configs.

---

### `utils.py`

Utility functions used across training and evaluation.

Includes:

- Environment decoding helpers
- Evaluation rollout helpers
- Trace handling
- Misc experiment utilities

---

## Training pipeline

### `train.py`

Main training script.

Responsible for:

- Loading experiment config
- Creating run directory
- Initializing environment, network, and MCTS
- Running training loop
- Saving checkpoints
- Saving config and metadata

Outputs are stored in:
runs/<experiment_name>seed<seed><hash>/<2d or 3d>/

---

### `train_sweep.py`

Runs multiple training experiments automatically.

Features:

- Reads configs from `experiment_configs/`
- Launches multiple training jobs in parallel
- Skips already completed runs
- Supports multi-core execution

This is the main entry point for running batches of experiments.

---

### `experiment_config.py`

Defines experiment configuration system.

Contains:

- Dataclasses defining all config parameters
- Config loading logic
- Validation logic
- Hash generation for reproducible run directories

Configs are defined as JSON files in:
experiment_configs/

---

### `experiment_configs/`

Contains JSON experiment definitions.

Each file specifies:

- Training hyperparameters
- MCTS parameters
- Network architecture
- Environment parameters
- Schedules and exploration settings

Used by `train.py` and `train_sweep.py`.

---

## Evaluation pipeline

### `eval.py`

Evaluates a single checkpoint.

Responsible for:

- Loading trained checkpoint
- Running evaluation episodes across seeds
- Saving:

  - summary statistics (`summary.json`)
  - slim trajectory traces (`slim_traces.json`)
  - full MCTS trees (`trees_seed_*.npz`)

Outputs stored in:
runs/<experiment>/<env>/eval/<checkpoint_name>/

---

### `eval_sweep.py`

Runs evaluation across all experiments and checkpoints.

Features:

- Automatically discovers checkpoints
- Runs evaluation in parallel
- Skips already evaluated checkpoints
- Supports evaluating every Nth checkpoint

This is the main evaluation entry point.

---

## Plotting and analysis

### `plot_utils.py`

Original plotting utilities.

Contains functions for:

- Plotting agent trajectories
- Plotting MCTS trees
- Debug visualization

May rely on older file formats.

---

### `plot_utils_eval.py`

Updated plotting utilities compatible with new evaluation outputs.

Supports plotting using:

- slim traces
- tree NPZ files
- evaluation summaries

Recommended plotting utilities going forward.

---

### `Confidence intervals.py`

Utility functions for computing confidence intervals for evaluation statistics.

Used for comparing experiments.

---

### `jupyter_notebooks/`

Contains analysis notebooks.

Used for:

- Loading evaluation results
- Comparing experiments
- Visualizing trajectories and trees
- Computing summary statistics

⚠️ Some notebooks may be outdated and require adjustments to work with the current evaluation format.

---

## Experiment outputs

### `runs/`

Contains all training and evaluation results.

Structure:
runs/
    experiment_name__seed42__hash/
        2d/
            checkpoints/
                ckpt_step_XXXX.pt
            eval/
                ckpt_step_XXXX/
                summary.json
                slim_traces.json
                trees_seed_XXXX.npz
            config.json
            DONE.json
            meta.json

This directory contains all data needed for analysis.

---

# Typical workflow

## Training

Run experiment sweep:

```bash
python train_sweep.py --env 2d --max-workers 10
```

---

# Evaluation

Evaluate checkpoints:

```bash
python eval_sweep.py --env 2d --max-workers 10 --every-n 2
```

---

# Analysis

Use notebooks in:

jupyter_notebooks/

and plotting functions in:

plot_utils_eval.py

---

# Important notes

This repository is experimental and still under development.

Some characteristics:
- Code may not be fully cleaned or documented
- Some plotting utilities and notebooks may be outdated
- File formats evolved during development
- Evaluation pipeline was recently rewritten
- Not intended as a polished software package
- However, the training and evaluation pipelines are functional and reproducible.

---

# Key entry points

## Main training entry point:

train_sweep.py

## Main evaluation entry point:

eval_sweep.py

## Main algorithm implementation:

MCTS_AC.py