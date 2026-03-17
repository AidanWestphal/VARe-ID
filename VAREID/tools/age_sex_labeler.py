import os
import sqlite3
import json
import warnings
import yaml
from PIL import Image

import gradio as gr
import pandas as pd

from VAREID.libraries.io.format_funcs import load_dataframe, split_dataframe, save_json

warnings.filterwarnings("ignore")

# Define cache directory for bounding box crops
os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/gradio_cache")
CACHE_DIR = os.path.expanduser("~/gradio_cache/zebra_crops")

# Load configuration
try:
    with open('age_sex_labeler.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'age_sex_labeler.yaml'. Please ensure it is in the same directory.")

CONFIG = {
    "image_data": config_data.get("image_data", "lca_annots.json"),
    "db": config_data.get("db_path", "cluster_labels.db"),
    "out_json": config_data.get("out_json", "lca_annots_labeled.json"),
    "port": config_data.get("port", 7860),
    "load_unlabeled_only": config_data.get("load_unlabeled_only", True)
}


def setup_data_and_db(json_path, db_path):
    print(f"Loading annotation data from {json_path}...")
    df = load_dataframe(json_path)
    
    if 'age' not in df.columns:
        df['age'] = ""
    if 'sex' not in df.columns:
        df['sex'] = ""
        
    df['age'] = df['age'].fillna("")
    df['sex'] = df['sex'].fillna("")
    
    db_exists = os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # If starting fresh, initialize schema and populate based on config rules
    if not db_exists:
        print(f"Initializing new database at {db_path}...")
        c.execute('''
            CREATE TABLE cluster_labels (
                cluster_id TEXT PRIMARY KEY,
                age TEXT,
                sex TEXT,
                status TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE annotation_status (
                uuid TEXT PRIMARY KEY,
                cluster_id TEXT,
                keep INTEGER
            )
        ''')
        
        valid_clusters = df.dropna(subset=['cluster_id'])
        cluster_groups = valid_clusters.groupby('cluster_id')
        
        for cluster_id, group in cluster_groups:
            existing_age = ""
            existing_sex = ""
            
            # Only propagate labels if user requested to load unlabeled data only
            if CONFIG["load_unlabeled_only"]:
                for _, row in group.iterrows():
                    if str(row.get('age', '')).strip() != "":
                        existing_age = str(row['age']).strip()
                    if str(row.get('sex', '')).strip() != "":
                        existing_sex = str(row['sex']).strip()
            
            # Determine status. If we have both, and we only wanted unlabeled, skip it by marking 'labeled'.
            # If load_unlabeled_only is False, existing_age/sex are empty, status is 'pending' for all.
            status = "labeled" if (CONFIG["load_unlabeled_only"] and existing_age and existing_sex) else "pending"
            
            c.execute('INSERT INTO cluster_labels (cluster_id, age, sex, status) VALUES (?, ?, ?, ?)', 
                      (str(cluster_id), existing_age, existing_sex, status))
            
            # Add all individual annotations to the tracking table
            for idx, row in group.iterrows():
                uuid_val = str(row.get('uuid', f"nouuid_{idx}"))
                # default keep = 1 (True)
                c.execute('INSERT INTO annotation_status (uuid, cluster_id, keep) VALUES (?, ?, 1)', 
                          (uuid_val, str(cluster_id)))
                          
        conn.commit()
    else:
        print(f"Resuming from existing database at {db_path}...")

    conn.close()
    return df


def get_cluster_images(df, cluster_id, cache_dir):
    """Retrieves cached bounding box crops for a given cluster."""
    rows = df[df['cluster_id'].astype(str) == str(cluster_id)]
    gallery_items = []
    uuid_list = []
    
    os.makedirs(cache_dir, exist_ok=True)
    
    for idx, row in rows.iterrows():
        img_path = row.get('image_path')
        bbox = row.get('bbox')
        uuid_val = str(row.get('uuid', f"nouuid_{idx}"))
        
        if not img_path or pd.isna(img_path) or not os.path.exists(str(img_path)):
            continue
            
        try:
            img = Image.open(str(img_path))
            if pd.notna(bbox):
                if isinstance(bbox, str):
                    try:
                        bbox = json.loads(bbox)
                    except json.JSONDecodeError:
                        pass
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = [float(c) for c in bbox]
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(img.width, int(x2)), min(img.height, int(y2))
                    img = img.crop((x1, y1, x2, y2))
            
            temp_path = os.path.join(cache_dir, f"crop_{uuid_val}.jpg")
            img.save(temp_path)
            
            # Store tuple of (image_path, caption) for the gallery
            gallery_items.append((temp_path, f"UUID: {uuid_val}"))
            uuid_list.append(uuid_val)
            
        except Exception as e:
            print(f"Warning: Error processing image {img_path}: {e}")
            
    return gallery_items, uuid_list


def export_data():
    """Generates the new annotation file, applying labels and marking excluded annotations with cluster_id = -1."""
    out_json = CONFIG["out_json"]
    df = CONFIG["df"].copy()
    
    conn = sqlite3.connect(CONFIG["db"])
    labels_df = pd.read_sql_query("SELECT cluster_id, age, sex FROM cluster_labels", conn)
    excluded_df = pd.read_sql_query("SELECT uuid FROM annotation_status WHERE keep = 0", conn)
    conn.close()
    
    label_dict = labels_df.set_index('cluster_id').to_dict('index')
    excluded_uuids = set(excluded_df['uuid'].astype(str).tolist())
    
    def apply_labels(row):
        uuid_str = str(row.get('uuid', ''))
        
        # 1. If this annotation was marked for exclusion, reassign to cluster -1 and clear labels
        if uuid_str in excluded_uuids:
            row['cluster_id'] = -1
            row['age'] = ""
            row['sex'] = ""
            return row
            
        # 2. Otherwise, propagate cluster-level labels normally
        cid = str(row.get('cluster_id', ''))
        if cid in label_dict:
            db_a = label_dict[cid]['age']
            db_s = label_dict[cid]['sex']
            if db_a: row['age'] = db_a
            if db_s: row['sex'] = db_s
        return row
        
    df = df.apply(apply_labels, axis=1)
    
    # 3. Export
    out_dict = split_dataframe(df)
    save_json(out_dict, out_json)
    
    return f"Data successfully exported to `{out_json}`. Reassigned {len(excluded_uuids)} bad annotations to cluster `-1`."


def load_ui_data():
    """Fetches the next cluster requiring labeling."""
    conn = sqlite3.connect(CONFIG["db"])
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM cluster_labels WHERE status = "pending"')
    pending_count = c.fetchone()[0]
    
    if pending_count == 0:
        # Trigger automatic export if nothing is left
        conn.close()
        export_msg = export_data()
        final_text = f"**✅ All clusters labeled!**\n\n{export_msg}"
        
        # Hide interactive elements via gr.update
        return (None, final_text, gr.update(visible=False), gr.update(visible=False), 
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
                
    c.execute('SELECT cluster_id, age, sex FROM cluster_labels WHERE status = "pending" LIMIT 1')
    row = c.fetchone()
    conn.close()
    
    cluster_id, age, sex = row
    
    gallery_items, uuid_list = get_cluster_images(CONFIG["df"], cluster_id, CACHE_DIR)
    status_text = f"**Labeling Cluster:** `{cluster_id}` | **Pending:** {pending_count}"
    
    ui_age = age if age else None
    ui_sex = sex if sex else None
    
    # Return UI elements. Make sure controls are visible in case we restarted.
    return (cluster_id, status_text, 
            gr.update(value=ui_age, visible=True), 
            gr.update(value=ui_sex, visible=True), 
            gr.update(value=gallery_items, visible=True), 
            gr.update(choices=uuid_list, value=[], visible=True),
            gr.update(visible=True))


def on_submit(cid, age_val, sex_val, excluded_uuids):
    """Saves the given selections/exclusions to the database and fetches the next cluster."""
    if cid is None:
        return load_ui_data()
        
    a = age_val if age_val else ""
    s = sex_val if sex_val else ""
    
    conn = sqlite3.connect(CONFIG["db"])
    c = conn.cursor()
    
    # Update Cluster Labels
    c.execute('UPDATE cluster_labels SET age=?, sex=?, status="labeled" WHERE cluster_id=?', 
              (a, s, cid))
              
    # Process excluded annotations
    if excluded_uuids:
        for uuid_val in excluded_uuids:
            c.execute('UPDATE annotation_status SET keep=0 WHERE uuid=?', (str(uuid_val),))
            
    conn.commit()
    conn.close()
    
    return load_ui_data()


def build_interface():
    with gr.Blocks(title="Zebra Annotation GUI") as demo:
        gr.Markdown("# Zebra Cluster Age & Sex Labeling")
        
        cluster_state = gr.State()
        status_md = gr.Markdown("**Status:** Initializing...")
        
        gallery = gr.Gallery(
            label="Cluster Detections (Cropped) - Note the UUID on each image.", 
            show_label=True, 
            elem_id="gallery", 
            columns=[4], 
            rows=[1], 
            object_fit="contain", 
            height="auto"
        )
        
        with gr.Row():
            age_radio = gr.Radio(
                choices=["0-2", "3-5", "6-11", "12-23", "24-35", "36+"], 
                label="Age (Months)",
                interactive=True
            )
            sex_radio = gr.Radio(
                choices=["Male", "Female"], 
                label="Sex",
                interactive=True
            )
            
        exclude_checkboxes = gr.CheckboxGroup(
            label="❌ EXCLUDE Bad Annotations", 
            info="Select the UUIDs of any images that do NOT belong in this cluster (incorrect detection, bad quality, etc.). They will be reassigned to cluster '-1' for later review.",
            choices=[],
            interactive=True
        )
            
        submit_btn = gr.Button("Submit & Next ➡", variant="primary", size="lg")
        
        # Load the initial data, hooking up visibility toggles for the auto-save hide mechanic
        demo.load(
            load_ui_data, 
            outputs=[cluster_state, status_md, age_radio, sex_radio, gallery, exclude_checkboxes, submit_btn]
        )
        
        submit_btn.click(
            on_submit, 
            inputs=[cluster_state, age_radio, sex_radio, exclude_checkboxes], 
            outputs=[cluster_state, status_md, age_radio, sex_radio, gallery, exclude_checkboxes, submit_btn]
        )
        
    return demo


if __name__ == "__main__":
    CONFIG["df"] = setup_data_and_db(CONFIG["image_data"], CONFIG["db"])
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=CONFIG["port"])