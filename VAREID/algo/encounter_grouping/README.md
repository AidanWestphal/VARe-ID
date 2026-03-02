# Encounter Grouping

Groups annotations into encounters based on when and where photos were taken.

## What is an encounter?

An encounter is a set of images captured close together in time and space, likely depicting the same sighting event. For example, a camera trap firing 10 times over 2 minutes at the same waterhole produces one encounter.

## How it works

1. Compute pairwise GPS distance (geodesic, in km) between all annotations
2. Compute pairwise time difference (in hours) between all annotations
3. Build a connectivity matrix where two annotations are connected only if BOTH conditions hold:
   - Distance < `max_distance_km` (default: 1.0 km)
   - Time gap < `max_time_hours` (default: 0.5 hours)
4. Run DBSCAN on this connectivity matrix to form encounter groups
5. Annotations that don't meet either threshold with any other annotation get their own singleton encounter
6. Annotations missing GPS or timestamp metadata also become singletons

Each annotation receives an `occurence_id` field identifying its encounter.

## Why this step exists

Without encounter grouping, LCA would try to cluster all annotations at once. That is expensive and unnecessary. Photos taken seconds apart at the same location almost certainly show the same animal(s), so grouping them first lets the pipeline:

- Run intra-encounter LCA on small, local groups (fast, no human review needed)
- Pick one representative per group for inter-encounter matching
- Reserve human review budget for the harder cross-encounter comparisons

## Configuration

See `encounter_grouping_config.yaml`:

- `max_distance_km`: spatial radius (default 1.0 km)
- `max_time_hours`: temporal window (default 0.5 hours)
- `min_samples_per_encounter`: minimum group size (default 1)
