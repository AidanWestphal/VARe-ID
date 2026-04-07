import os
import sqlite3
import json
import warnings
import yaml
from PIL import Image

import gradio as gr
import pandas as pd

from VAREID.libraries.io.format_funcs import load_json, join_dataframe, split_dataframe, save_json

warnings.filterwarnings("ignore")

# Define cache directory for bounding box crops
os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/gradio_cache")
CACHE_DIR = os.path.expanduser("~/gradio_cache/zebra_crops")

# Define maximum safe image threshold per cluster for the UI grid
MAX_IMAGES = 100

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

    df = load_json(json_path)
    df = join_dataframe(df)
    
    if 'age' not in df.columns:
        df['age'] = ""
    if 'sex' not in df.columns:
        df['sex'] = ""
        
    df['age'] = df['age'].fillna("")
    df['sex'] = df['sex'].fillna("")
    
    db_exists = os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
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
            
            if CONFIG["load_unlabeled_only"]:
                for _, row in group.iterrows():
                    a_val = str(row.get('age', '')).strip()
                    if a_val and a_val.lower() != "unknown":
                        existing_age = a_val
                        
                    s_val = str(row.get('sex', '')).strip()
                    if s_val and s_val.lower() != "unknown":
                        existing_sex = s_val
            
            status = "labeled" if (CONFIG["load_unlabeled_only"] and existing_age and existing_sex) else "pending"
            
            c.execute('INSERT INTO cluster_labels (cluster_id, age, sex, status) VALUES (?, ?, ?, ?)', 
                      (str(cluster_id), existing_age, existing_sex, status))
            
            for idx, row in group.iterrows():
                uuid_val = str(row.get('uuid', f"nouuid_{idx}"))
                c.execute('INSERT INTO annotation_status (uuid, cluster_id, keep) VALUES (?, ?, 1)', 
                          (uuid_val, str(cluster_id)))
                          
        conn.commit()
    else:
        print(f"Resuming from existing database at {db_path}...")

    conn.close()
    return df


def get_cluster_images_and_captions(df, cluster_id, cache_dir):
    """Retrieves cached bounding box crops for a given cluster along with original image paths."""
    rows = df[df['cluster_id'].astype(str) == str(cluster_id)]
    crop_paths = []
    orig_paths = []
    uuid_mapping = []
    
    os.makedirs(cache_dir, exist_ok=True)
    
    for idx, row in rows.iterrows():
        if len(crop_paths) >= MAX_IMAGES:
            print(f"Warning: Cluster {cluster_id} exceeds {MAX_IMAGES} images. Truncating UI view.")
            break

        img_path = row.get('image_path')
        bbox = row.get('bbox')
        uuid_val = str(row.get('uuid', f"nouuid_{idx}"))
        
        if not img_path or pd.isna(img_path) or not os.path.exists(str(img_path)):
            continue
            
        try:
            temp_path = os.path.join(cache_dir, f"crop_{uuid_val}.jpg")
            
            if not os.path.exists(temp_path):
                img = Image.open(str(img_path))
                
                if isinstance(bbox, str):
                    try:
                        bbox = json.loads(bbox)
                    except json.JSONDecodeError:
                        pass
                
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x, y, w, h = [float(c) for c in bbox]
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2 = min(img.width, int(x + w))
                    y2 = min(img.height, int(y + h))
                    img = img.crop((x1, y1, x2, y2))
                
                img.save(temp_path)
            
            orig_paths.append(str(img_path))
            crop_paths.append(temp_path)
            uuid_mapping.append(uuid_val)
            
        except Exception as e:
            print(f"Warning: Error processing image {img_path}: {e}")
            
    return crop_paths, orig_paths, uuid_mapping


def export_data():
    """Generates the new annotation file, routing excluded annots to cluster_id = -1."""
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
        
        # Priority 1: Exclusions become -1 and get scrubbed
        if uuid_str in excluded_uuids:
            row['cluster_id'] = -1
            row['age'] = ""
            row['sex'] = ""
            return row
            
        # Priority 2: Valid cluster images inherit consensus
        cid = str(row.get('cluster_id', ''))
        if cid in label_dict:
            db_a = label_dict[cid]['age']
            db_s = label_dict[cid]['sex']
            if db_a: row['age'] = db_a
            if db_s: row['sex'] = db_s
        return row
        
    df = df.apply(apply_labels, axis=1)
    
    out_dict = split_dataframe(df)
    save_json(out_dict, out_json)
    
    return f"Data successfully exported to `{out_json}`. Reassigned {len(excluded_uuids)} bad annotations to cluster `-1`."


def load_ui_data():
    """Builds the dynamic arrays for our fixed-grid layout and fetches the next cluster."""
    conn = sqlite3.connect(CONFIG["db"])
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM cluster_labels WHERE status = "pending"')
    pending_count = c.fetchone()[0]
    
    # --- END STATE HANDLING ---
    if pending_count == 0:
        conn.close()
        export_msg = export_data()
        final_text = f"**✅ All clusters labeled!**\n\n{export_msg}"
        
        # Hide all controls and columns
        col_updates = [gr.update(visible=False)] * MAX_IMAGES
        img_updates = [gr.update(value=None)] * MAX_IMAGES
        html_updates = [gr.update(value="")] * MAX_IMAGES
        chk_updates = [gr.update(value=False)] * MAX_IMAGES
        zoom_updates = [gr.update(value=False)] * MAX_IMAGES
        
        return [None, final_text, gr.update(visible=False), gr.update(visible=False)] + \
               col_updates + img_updates + html_updates + chk_updates + zoom_updates + \
               [{}, gr.update(visible=False)]
               
    # --- NEXT CLUSTER HANDLING ---
    c.execute('SELECT cluster_id, age, sex FROM cluster_labels WHERE status = "pending" LIMIT 1')
    row = c.fetchone()
    cluster_id, age, sex = row
    
    crop_paths, orig_paths, uuid_mapping = get_cluster_images_and_captions(CONFIG["df"], cluster_id, CACHE_DIR)
    
    # Query any previously saved exclusion states
    all_cluster_uuids = ', '.join([f"'{u}'" for u in uuid_mapping])
    existing_exclusions = set()
    if all_cluster_uuids:
        c.execute(f"SELECT uuid FROM annotation_status WHERE uuid IN ({all_cluster_uuids}) AND keep = 0")
        existing_exclusions = set(r[0] for r in c.fetchall())
    conn.close()
    
    # Build exact length arrays for our fixed grid UI
    col_updates, img_updates, html_updates, chk_updates, zoom_updates = [], [], [], [], []
    for i in range(MAX_IMAGES):
        zoom_updates.append(gr.update(value=False)) # Always reset full image toggle on new cluster
        if i < len(crop_paths):
            u_val = uuid_mapping[i]
            col_updates.append(gr.update(visible=True))
            img_updates.append(gr.update(value=crop_paths[i]))
            html_updates.append(gr.update(value=f"<div style='font-size: 0.8em; word-break: break-all; color: gray; margin-bottom: -10px;'>UUID: {u_val}</div>"))
            chk_updates.append(gr.update(value=(u_val in existing_exclusions)))
        else:
            col_updates.append(gr.update(visible=False))
            img_updates.append(gr.update(value=None))
            html_updates.append(gr.update(value=""))
            chk_updates.append(gr.update(value=False))

    cluster_json = {
        "uuid_mapping": uuid_mapping,
        "crop_paths": crop_paths,
        "orig_paths": orig_paths
    }
    
    status_text = f"**Labeling Cluster:** `{cluster_id}` | **Pending:** {pending_count}"
    
    # Translate empty string back to "Unknown" for UI if needed
    ui_age = "Unknown" if (age == "" and age is not None) else age
    ui_sex = "Unknown" if (sex == "" and sex is not None) else sex
    
    return [cluster_id, status_text, gr.update(value=ui_age if ui_age else None, visible=True), gr.update(value=ui_sex if ui_sex else None, visible=True)] + \
           col_updates + img_updates + html_updates + chk_updates + zoom_updates + \
           [cluster_json, gr.update(visible=True)]


def on_submit(cid, age_val, sex_val, cluster_data_json, *checkbox_values):
    """Validates inputs, commits changes to SQLite, and cycles queue."""
    if cid is None:
        return load_ui_data()
        
    uuid_mapping = cluster_data_json.get("uuid_mapping", [])
    total_active_images = len(uuid_mapping)
    
    excluded_count = 0
    # The first total_active_images items in checkbox_values map to our 'Remove' checkboxes
    for i in range(total_active_images):
        if checkbox_values[i]:
            excluded_count += 1
            
    if excluded_count < total_active_images:
        if not age_val or not sex_val:
            raise gr.Error("⚠️ You must select both an Age and a Sex. If unsure, select 'Unknown'.")
    else:
        age_val = ""
        sex_val = ""

    db_age = "" if age_val == "Unknown" else age_val
    db_sex = "" if sex_val == "Unknown" else sex_val

    conn = sqlite3.connect(CONFIG["db"])
    c = conn.cursor()
    
    c.execute('UPDATE cluster_labels SET age=?, sex=?, status="labeled" WHERE cluster_id=?', (db_age, db_sex, cid))
              
    for i in range(total_active_images):
        u_val = uuid_mapping[i]
        is_keep = 0 if checkbox_values[i] else 1
        c.execute('UPDATE annotation_status SET keep=? WHERE uuid=?', (is_keep, u_val))
            
    conn.commit()
    conn.close()
    
    return load_ui_data()


def make_toggle_fn(idx):
    """Closure factory to safely handle Gradio's event binding inside a loop."""
    def toggle(is_full, c_data):
        if not c_data: return gr.update()
        crop_paths = c_data.get("crop_paths", [])
        orig_paths = c_data.get("orig_paths", [])
        if idx < len(crop_paths) and idx < len(orig_paths):
            return orig_paths[idx] if is_full else crop_paths[idx]
        return gr.update()
    return toggle


def build_interface():
    with gr.Blocks(title="Zebra Annotation GUI") as demo:
        gr.Markdown("# Zebra Cluster Age & Sex Labeling")
        
        cluster_state = gr.State()
        cluster_data_state = gr.JSON({}, visible=False) 
        
        status_md = gr.Markdown("**Status:** Initializing...")
        
        with gr.Row():
            age_radio = gr.Radio(
                choices=["0-2", "3-5", "6-11", "12-23", "24-35", "36+", "Unknown"], 
                label="Age (Months)",
                interactive=True
            )
            sex_radio = gr.Radio(
                choices=["Male", "Female", "Unknown"], 
                label="Sex",
                interactive=True
            )
            
        submit_btn = gr.Button("Submit & Next ➡", variant="primary", size="lg")
        
        gr.Markdown("---")
        gr.Markdown("### Cluster Detections\nCheck `❌ Remove` below any image that is a bad detection. Check `🔍 Full Image` to zoom out.")
        
        # --- FIXED UI GRID GENERATOR ---
        box_columns = []
        image_components = []
        html_components = []
        checkbox_components = []
        zoom_checkbox_components = []
        
        with gr.Row():
            for i in range(MAX_IMAGES):
                with gr.Column(visible=False, min_width=220) as col:
                    img = gr.Image(interactive=False, show_label=False, show_download_button=False)
                    uuid_html = gr.HTML()
                    
                    with gr.Row():
                        zoom_chk = gr.Checkbox(label="🔍 Full Image")
                        chk = gr.Checkbox(label="❌ Remove")
                    
                    # Bind local scoped zoom event using our factory function
                    zoom_chk.change(
                        fn=make_toggle_fn(i),
                        inputs=[zoom_chk, cluster_data_state],
                        outputs=[img]
                    )
                    
                    box_columns.append(col)
                    image_components.append(img)
                    html_components.append(uuid_html)
                    checkbox_components.append(chk)
                    zoom_checkbox_components.append(zoom_chk)

        # Mapping I/O flows
        ui_outputs = [cluster_state, status_md, age_radio, sex_radio] + box_columns + image_components + html_components + checkbox_components + zoom_checkbox_components + [cluster_data_state, submit_btn]
        
        # submit_inputs doesn't need zoom_checkboxes since we don't save zoom state to the DB
        submit_inputs = [cluster_state, age_radio, sex_radio, cluster_data_state] + checkbox_components
        
        demo.load(load_ui_data, outputs=ui_outputs)
        submit_btn.click(on_submit, inputs=submit_inputs, outputs=ui_outputs)
        
        return demo


if __name__ == "__main__":
    CONFIG["df"] = setup_data_and_db(CONFIG["image_data"], CONFIG["db"])
    
    # Extract unique parent directories of the original images so Gradio knows they are safe to serve
    valid_paths = [str(p) for p in CONFIG["df"]['image_path'].dropna() if os.path.exists(str(p))]
    allowed_dirs = list(set(os.path.dirname(os.path.abspath(p)) for p in valid_paths))
    
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0", 
        server_port=CONFIG["port"], 
        share=True, 
        allowed_paths=allowed_dirs
    )