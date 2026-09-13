# Experiment 02: DGL Alpha Tuning

## Overview

This experiment investigates the effect of the regularization strength (alpha)
in the proposed Dual Graph Laplacian (DGL) module.

This branch is based on:

- Parent experiment: `exp01-dgl-baseline`
- Backbone: IPG Super-Resolution Network
- Task: Single Image Super-Resolution (x4)
- Dataset: DF2K
- Baseline DGL configuration:
  - Dual Graph Laplacian inserted into IPG Grapher
  - Initial alpha = 0.01


## Motivation

The Dual Graph Laplacian module introduces an additional feature regularization
term:

    X_out = X + alpha * R(X)

where alpha controls the strength of the graph-based regularization.

A proper alpha value is important because:

- Small alpha values may produce insufficient graph regularization.
- Large alpha values may over-constrain feature representations and reduce
  reconstruction capability.

Therefore, this experiment performs an alpha sensitivity analysis.


## Experimental Setup

All experiments keep the same configuration as the DGL baseline:

- Same IPG architecture
- Same DGL placement
- Same training dataset
- Same optimization settings
- Same validation protocol

Only the alpha parameter is modified.


## Tested Alpha Values

Planned configurations:

| Experiment | Alpha |
|------------|-------|
| DGL-A1 | 0.001 |
| DGL-A2 | 0.005 |
| DGL-A3 | 0.010 |
| DGL-A4 | 0.050 |
| DGL-A5 | 0.100 |


## Evaluation

Models are evaluated using:

- PSNR
- SSIM
- Validation convergence behavior

Validation datasets:

- DF2K
- Set5


## Expected Analysis

The goal is to determine:

1. The optimal regularization strength for DGL.
2. Whether stronger graph constraints improve reconstruction quality.
3. The trade-off between feature smoothing and detail preservation.


## Results

Results will be added after completing all alpha configurations.

| Alpha | DF2K PSNR | Set5 PSNR | Notes |
|------|-----------|-----------|------|
|0.001 | - | - | |
|0.005 | - | - | |
|0.010 | - | - | Baseline |
|0.050 | - | - | |
|0.100 | - | - | |


## Relation to Previous Experiment

This experiment extends `exp01-dgl-baseline`.

No architectural changes are introduced.
The objective is only to analyze the influence of the DGL regularization strength.
