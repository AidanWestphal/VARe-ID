# Forward Clustering

Propagates individual identity assignments from representative annotations back to all other annotations in their encounter group.

## Why this step exists

The inter-encounter LCA only clusters representative annotations (one per encounter group). That leaves all non-representative annotations without a final `cluster_id`. Forward clustering fills in the gaps: it takes the identity assigned to each representative and copies it to every other annotation that shares the same `encounter_id`.

## How it works

1. Group all annotations by `encounter_id`
2. In each group, find the annotation marked `representative=True`
3. Read its `cluster_id` (set by inter-encounter LCA)
4. Assign that same `cluster_id` to every other annotation in the group

That's it. No computation, no similarity checks. Just ID propagation.

## Example

```
Encounter group 42:
  Ann_A  (representative=True,  cluster_id=7)  -> keeps cluster_id=7
  Ann_B  (representative=False, cluster_id=?)  -> gets cluster_id=7
  Ann_C  (representative=False, cluster_id=?)  -> gets cluster_id=7
```

All three annotations now belong to individual 7.

## Where it fits in the pipeline

```
Intra-LCA -> Representative Selection -> Inter-LCA -> Forward Clustering -> Final Output
```

Forward clustering is the last step before the final output. Its output is the `lca_out_path` used by downstream stages.
