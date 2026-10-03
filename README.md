### Experiment 03 — DGL-Guided Graph Construction

In this experiment, the Dual Graph Laplacian (DGL) is moved from the post-aggregation stage to the graph construction stage.

Instead of refining the output features after graph reasoning, DGL is applied to the input features before graph construction. The DGL-processed features are used only to build the graph, while the original features are still used for graph aggregation.

The goal is to investigate whether row- and column-aware structural information from DGL can improve the topology of the graph constructed by IPG.

Initial setting:
- Scale: ×4
- Initial α: 0.02
- Short screening training: 9K iterations
