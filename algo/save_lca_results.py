import pandas as pd
import json
import os

def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def save_lca_results(input_dir, anno_file, output_dir, viewpoint=None):
    clustering_file = os.path.join(input_dir, "clustering.json")
    node2uuid_file = os.path.join(input_dir, "node2uuid_file.json")

    data = load_json(anno_file)

    df_categories = pd.DataFrame(data['categories'])
    df_annotations = pd.DataFrame(data['annotations'])
    df_images = pd.DataFrame(data['images'])

    # Merge annotations with image metadata
    df = df_annotations.merge(df_images, left_on='image_uuid', right_on='uuid')

    # Optionally filter by viewpoint
    if viewpoint is not None:
        df = df[df['viewpoint'] == viewpoint]

    # Load clustering results
    clusters = load_json(clustering_file)
    node2uuid = load_json(node2uuid_file)

    # Map UUIDs to cluster IDs
    uuid_to_cluster = {}
    for cluster_id, nodes in clusters.items():
        for node in nodes:
            uuid = node2uuid.get(str(node))
            if uuid:
                uuid_to_cluster[uuid] = cluster_id

    # Ensure UUID column exists
    if 'uuid' not in df.columns:
        df['uuid'] = df['uuid_x']

    # Assign cluster IDs
    df['LCA_clustering_id'] = df['uuid'].map(uuid_to_cluster).where(df['uuid'].isin(uuid_to_cluster), None)

    # Modify output file name
    suffix = f"LCA_{viewpoint}" if viewpoint else "LCA"
    name, ext = os.path.splitext(os.path.basename(anno_file))
    output_filename = f"{name}_{suffix}{ext}"
    output_path = os.path.join(output_dir, output_filename)

    # Prepare final annotation data
    annotations_fields = [
        'uuid', 'image_uuid', 'bbox', 'viewpoint', 'tracking_id',
        'individual_id', 'confidence', 'detection_class', 'species',
        'CA_score', 'category_id', 'LCA_clustering_id'
    ]
    df_annotations = df[annotations_fields]

    # Prepare image data
    image_fields = ['image_uuid', 'file_name']
    image_fields = df.columns.intersection(image_fields)
    df_images = df[image_fields].drop_duplicates(keep='first').reset_index(drop=True)
    df_images = df_images.rename(columns={'image_uuid': 'uuid'})

    # Prepare category data
    category_fields = ['category_id', 'species']
    df_categories = df[category_fields].drop_duplicates(keep='first').reset_index(drop=True)
    df_categories = df_categories.rename(columns={'category_id': 'id'})

    # Assemble final JSON structure
    result_dict = {
        'categories': df_categories.to_dict(orient='records'),
        'images': df_images.to_dict(orient='records'),
        'annotations': df_annotations.to_dict(orient='records')
    }

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
        print('Data saved to:', output_path)
