import pandas as pd
import json
import os

def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def save_lca_results(input_dir, anno_file, output_dir, viewpoint=None):
    clustering_file = os.path.join(input_dir, "clustering.json")
    node2uuid_file = os.path.join(input_dir, "node2uuid_file.json")

    # Load original annotation file
    data = load_json(anno_file)

    # Load clustering results
    clusters = load_json(clustering_file)
    node2uuid = load_json(node2uuid_file)

    # Build mapping from UUID to cluster ID
    uuid_to_cluster = {}
    for cluster_id, nodes in clusters.items():
        for node in nodes:
            uuid = node2uuid.get(str(node))
            if uuid:
                uuid_to_cluster[uuid] = cluster_id

    # Create a lookup of image_uuid -> viewpoint (if exists)
    image_viewpoints = {
        img['uuid']: img.get('viewpoint')
        for img in data['annotations']
    }

    # Filter annotations based on viewpoint if provided
    if viewpoint is not None:
        filtered_annotations = [
            ann for ann in data['annotations']
            if ann.get('viewpoint', '').strip().lower() == viewpoint.strip().lower()
        ]
        print(f"Filtered {len(filtered_annotations)} annotations with viewpoint='{viewpoint}' "
            f"out of {len(data['annotations'])}")
    else:
        filtered_annotations = data['annotations']

    # Add LCA_clustering_id to each annotation
    for ann in filtered_annotations:
        ann['LCA_clustering_id'] = uuid_to_cluster.get(ann['uuid'], None)
        # Modify output file name
        suffix = f"LCA_{viewpoint}" if viewpoint else "LCA"
        name, ext = os.path.splitext(os.path.basename(anno_file))
        output_filename = f"{name}_{suffix}{ext}"
        output_path = os.path.join(output_dir, output_filename)

    # Save final result with same categories/images, modified annotations
    result_dict = {
        'categories': data['categories'],
        'images': data['images'],
        'annotations': filtered_annotations
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
        print('Data saved to:', output_path)