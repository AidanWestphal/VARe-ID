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
        config_data = yaml.load(f, Loader=yaml.SafeLoader)
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'config_labeling.yaml'. Please ensure it is in the same directory.")

CONFIG = {
    "image_data": config_data.get("image_data", "lca_annots.json"),
    "db": config_data.get("db_path", "cluster_labels.db"),
    "out_json": config_data.get("out_json", "lca_annots_labeled.json"),
    "port": config_data.get("port", 7860)
}


def setup_data_and_db(json_path, db_path):
    print(f"Loading annotation data from {json_path}...")
    df = load_dataframe(json_path)
    
    # Requirement 2: Dynamically create 'age' and 'sex' columns if they don't exist at all
    if 'age' not in df.columns:
        print("Creating missing 'age' column...")
        df['age'] = ""
    if 'sex' not in df.columns:
        print("Creating missing 'sex' column...")
        df['sex'] = ""
        
    # Standardize empty or NaN values just in case they exist but are null
    df['age'] = df['age'].fillna("")
    df['sex'] = df['sex'].fillna("")
    
    print(f"Initializing database at {db_path}...")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS cluster_labels (
            cluster_id TEXT PRIMARY KEY,
            age TEXT,
            sex TEXT,
            status TEXT
        )
    ''')
    
    # Parse through the data to establish consensus for each cluster
    # We drop NA cluster_ids to prevent aggregating non-clustered annotations
    valid_clusters = df.dropna(subset=['cluster_id'])
    cluster_groups = valid_clusters.groupby('cluster_id')
    
    for cluster_id, group in cluster_groups:
        existing_age = ""
        existing_sex = ""
        
        for _, row in group.iterrows():
            if str(row['age']).strip() != "":
                existing_age = str(row['age']).strip()
            if str(row['sex']).strip() != "":
                existing_sex = str(row['sex']).strip()
                
        # Register or update in the database
        c.execute('SELECT age, sex, status FROM cluster_labels WHERE cluster_id = ?', (str(cluster_id),))
        db_row = c.fetchone()
        
        if db_row is None:
            # If it already has both labels, mark as labeled so we skip it in the UI queue
            status = "labeled" if existing_age and existing_sex else "pending"
            c.execute('INSERT INTO cluster_labels (cluster_id, age, sex, status) VALUES (?, ?, ?, ?)', 
                      (str(cluster_id), existing_age, existing_sex, status))
                      
    conn.commit()
    conn.close()
    return df


def get_cluster_images(df, cluster_id, cache_dir):
    """Retrieves and locally caches the cropped bounding boxes for a given cluster."""
    rows = df[df['cluster_id'].astype(str) == str(cluster_id)]
    image_paths = []
    
    os.makedirs(cache_dir, exist_ok=True)
    
    for _, row in rows.iterrows():
        img_path = row.get('image_path')
        bbox = row.get('bbox')
        uuid_val = row.get('uuid', 'nouuid')
        
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
            image_paths.append(temp_path)
        except Exception as e:
            print(f"Warning: Error processing image {img_path}: {e}")
            
    return image_paths


def load_ui_data():
    """Fetches the next cluster requiring labeling."""
    conn = sqlite3.connect(CONFIG["db"])
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM cluster_labels WHERE status = "pending"')
    pending_count = c.fetchone()[0]
    
    c.execute('SELECT cluster_id, age, sex FROM cluster_labels WHERE status = "pending" LIMIT 1')
    row = c.fetchone()
    conn.close()
    
    if not row:
        return None, f"**✅ All clusters labeled!** (Pending: 0)", None, None, []
        
    cluster_id, age, sex = row
    
    crops = get_cluster_images(CONFIG["df"], cluster_id, CACHE_DIR)
    status_text = f"**Labeling Cluster:** `{cluster_id}` | **Pending:** {pending_count}"
    
    # Pre-select radio buttons if data already partially exists
    ui_age = age if age else None
    ui_sex = sex if sex else None
    
    return cluster_id, status_text, ui_age, ui_sex, crops


def on_submit(cid, age_val, sex_val):
    """Saves the given selections to the database and fetches the next cluster."""
    if cid is None:
        return load_ui_data()
        
    # Re-cast unselected radios (None) to our target schema empty strings
    a = age_val if age_val else ""
    s = sex_val if sex_val else ""
    
    conn = sqlite3.connect(CONFIG["db"])
    c = conn.cursor()
    # Mark 'labeled' once submitted.
    c.execute('UPDATE cluster_labels SET age=?, sex=?, status="labeled" WHERE cluster_id=?', 
              (a, s, cid))
    conn.commit()
    conn.close()
    
    return load_ui_data()


def export_data():
    """Generates the new annotation file by mapping the DB state back onto the DataFrame."""
    out_json = CONFIG["out_json"]
    df = CONFIG["df"].copy()
    
    # Load authoritative labels from SQLite
    conn = sqlite3.connect(CONFIG["db"])
    labels_df = pd.read_sql_query("SELECT cluster_id, age, sex FROM cluster_labels", conn)
    conn.close()
    
    label_dict = labels_df.set_index('cluster_id').to_dict('index')
    
    def apply_labels(row):
        cid = str(row.get('cluster_id', ''))
        if cid in label_dict:
            db_a = label_dict[cid]['age']
            db_s = label_dict[cid]['sex']
            if db_a: row['age'] = db_a
            if db_s: row['sex'] = db_s
        return row
        
    # Propagate to all rows in the dataframe matching the cluster_id
    df = df.apply(apply_labels, axis=1)
    
    # Decompose and save back to original format
    out_dict = split_dataframe(df)
    save_json(out_dict, out_json)
    
    return f"✅ Data successfully exported to `{out_json}`"


def build_interface():
    with gr.Blocks(title="Zebra Annotation GUI") as demo:
        gr.Markdown("# Zebra Cluster Age & Sex Labeling")
        gr.Markdown("Identify the age and sex for each cluster. Partially populated items are automatically pre-filled.")
        
        cluster_state = gr.State()
        
        with gr.Row():
            status_md = gr.Markdown("**Status:** Initializing...")
            export_btn = gr.Button("💾 Export Results to JSON", variant="secondary")
            
        export_status = gr.Markdown("")
        
        # Horizontal scroll achieved by placing images directly in a column-based Gradio Gallery
        gallery = gr.Gallery(
            label="Cluster Detections (Cropped)", 
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
            
        submit_btn = gr.Button("Submit & Next ➡", variant="primary", size="lg")
        
        # UI Callbacks
        demo.load(
            load_ui_data, 
            outputs=[cluster_state, status_md, age_radio, sex_radio, gallery]
        )
        
        submit_btn.click(
            on_submit, 
            inputs=[cluster_state, age_radio, sex_radio], 
            outputs=[cluster_state, status_md, age_radio, sex_radio, gallery]
        )
        
        export_btn.click(
            export_data, 
            outputs=[export_status]
        )
        
    return demo


if __name__ == "__main__":
    # Join DataFrame and structure starting environment map prior to launching the UI
    CONFIG["df"] = setup_data_and_db(CONFIG["image_data"], CONFIG["db"])
    
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=CONFIG["port"])