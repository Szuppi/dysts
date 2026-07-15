# Recovered Dysts 2021 clustering artifacts

This directory contains the clustering code and stored cluster memberships extracted from the public notebook:

`GilpinLab/dysts_data/dysts_data/benchmarks/figure_descriptive_statistics.ipynb`

## Recovered pipeline

The notebook loads the precomputed `feat_arr_all.pkl`, takes the median feature vector across trajectory replicates, embeds the system centroids with

```python
umap.UMAP(random_state=15, n_neighbors=5)
```

and applies

```python
AffinityPropagation().fit_predict(embedding_mean)
```

The notebook's stored output contains labels `0` through `6`, i.e. **seven clusters**, covering 131 systems. This differs from the statement of **eight clusters** in the 2021 paper/supplementary methods. The stored notebook output is therefore preserved here as its own reproducible artifact rather than silently relabeled as the paper's eight-cluster result.

## Files

- `clustering_related_cells.txt`: clustering-related notebook cells and their stored text outputs.
- `figure_descriptive_statistics_code.py`: all code cells extracted from the notebook.
- `dysts_2021_affinity_clusters.csv`: recovered 131-system mapping from the notebook's stored outputs.
- `dysts_polynomial_71_cluster_mapping.csv`: mapping for the 71 systems in the supplied `dysts_coefficients.py` registry.

## Coverage of the 71 polynomial systems

| Cluster | Number of systems |
|---:|---:|
| 0 | 9 |
| 1 | 12 |
| 2 | 9 |
| 3 | 6 |
| 4 | 10 |
| 5 | 14 |
| 6 | 10 |
| Unassigned | 1 |

`AtmosphericRegime` is the only one of the 71 systems absent from the recovered 131-system partition and is therefore left unassigned rather than being assigned heuristically.

## Joining to experiment results

```python
import pandas as pd

clusters = pd.read_csv(
    "analysis/dysts_2021_clusters/dysts_polynomial_71_cluster_mapping.csv"
)

results = results.merge(
    clusters[["system", "cluster"]],
    on="system",
    how="left",
    validate="many_to_one",
)
```

The integer labels are nominal identifiers only; their numerical order has no interpretation.
