# LCA (Locality-based Clustering Algorithm)

Clusters photo annotations into individual identities using MiewID embeddings, a graph-based stability algorithm, and optional human review.

## How it works

1. Load MiewID embeddings for each annotation
2. Compute cosine similarity between all pairs; keep top-K neighbors as candidate edges
3. Fit a 2-component GMM to the similarity score distribution to find an automatic threshold separating same-individual scores from different-individual scores
4. Label edges above threshold as positive, below as negative
5. Run the stability algorithm:
   - Phase 0 (automatic): resolve obvious contradictions by deactivating weak positive edges
   - Phase 1 (human-guided): select the most unstable remaining edges, ask a human reviewer, update labels, re-stabilize
6. Final clustering = connected components of active positive edges

## Single-stage vs. two-stage mode

When `encounter_grouping` is disabled in the config, LCA runs once on all annotations (single-stage). When enabled, LCA runs in two stages: intra-encounter and inter-encounter.

## Intra-encounter LCA (`--intra`)

Clusters annotations within each encounter group separately.

- **Input**: encounter-grouped annotations + MiewID embeddings
- **Scope**: only compares annotations that share the same `occurence_id`
- **Human review**: none (`max_human_reviews=0`); fully automatic
- **Output field**: `encounter_id` (local cluster assignment within the encounter)
- **Config**: `lca_intra.yaml`

This step is fast because encounter groups are small (a handful of photos from the same time and place). It groups co-occurring detections of the same individual without needing human help.

## Representative selection (between intra and inter)

After intra-LCA, each encounter cluster picks one representative: the annotation with the highest IA (identifiable annotation) score. Only representatives move forward to inter-encounter clustering. This keeps the inter-LCA input small and focused on the best-quality images.

## Inter-encounter LCA (`--inter`)

Clusters representative annotations across different encounters to link the same individual seen at different times and places.

- **Input**: representative annotations only + MiewID embeddings
- **Scope**: compares representatives from all encounters
- **Human review**: up to 500 reviews (`max_human_reviews=500`); uses UI-based human reviewer
- **Output field**: `cluster_id` (global individual identity)
- **Config**: `lca_inter.yaml`

This is the harder matching problem: the same animal photographed weeks or months apart, possibly from different angles. Human review is concentrated here where it has the most impact.

## Summary of differences

| | Intra-LCA | Inter-LCA |
|---|---|---|
| Scope | Within one encounter | Across all encounters |
| Input | All annotations in encounter | Representatives only |
| Human review | None | Up to 500 |
| Output field | `encounter_id` | `cluster_id` |
| Goal | Group co-occurring detections | Link same individual over time |

## Configuration files

- `lca_intra.yaml`: intra-encounter settings (stability algorithm, no human)
- `lca_inter.yaml`: inter-encounter settings (stability algorithm, with human)
- `lca_image.yaml`: single-stage mode (no encounter grouping)
- `lca_drone.yaml`: drone-specific settings
