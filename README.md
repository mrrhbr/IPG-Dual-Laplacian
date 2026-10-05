## Experiment: Initial Dual Graph Laplacian (DGL) Integration

The goal of this experiment is to investigate whether a lightweight graph-based feature regularization before the original IPG graph construction can improve feature representation and adaptive graph generation.

### Motivation
The original IPG pipeline constructs graphs based on feature similarity. However, before graph construction, no explicit spatial structural prior is introduced. The Dual Graph Laplacian (DGL) module is designed to provide additional row-wise and column-wise spatial relationships, allowing the network to exploit directional structures and repeated patterns in images.

### Current Implementation
- Added a `DualGraphLaplacian` module at the beginning of `MGB.forward()`.
- The module receives image tokens with shape `(B, H*W, C)` and restores the spatial structure.
- Two simple graphs are considered:
  - **Row graph:** models relationships between tokens located in the same image row.
  - **Column graph:** models relationships between tokens located in the same image column.
- Feature cosine similarity is used as edge weighting.
- Normalized Laplacian responses are computed for both graphs.
- The refined feature is obtained using residual regularization:

  `out = feature + alpha * dual_response`

### Integration Point
The DGL module is applied before `calc_graph()`. Therefore, the original IPG graph sampling and aggregation mechanisms remain unchanged. The experiment only evaluates whether feature refinement can improve the following graph construction stage.

### Current Status
This is an initial validation implementation. The current version uses a dense formulation to verify the effectiveness of dual graph regularization.

### Future Work
The main limitation is the memory cost of dense affinity computation. Future improvements will focus on implicit/sparse graph construction and more efficient Laplacian operations without explicitly creating large NxN matrices.
