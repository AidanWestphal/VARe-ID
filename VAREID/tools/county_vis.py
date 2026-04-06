import sqlite3
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.colors as pc
import json
import gradio as gr
import numpy as np
import ast
import os
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# Define all your datasets here. 
# If a dataset lacks a database, set "db": None
# ==========================================
DATASETS = {
    "GGR2024": {
        "json": "/fs/ess/PAS2136/ggr_data/results/GGR2024_fixed_encounter/id_region/id_regions_filtered.json",
        "db": "/fs/ess/PAS2136/ggr_data/results/GGR2024_fixed_encounter/annots.db"
    },
    "GGR2018": {
        "json": "/fs/ess/PAS2136/ggr_data/wbia/GGR2018/gt_annots.json",
        "db": None
    }
}

AVAILABLE_COLORS = pc.qualitative.Plotly + pc.qualitative.Bold
DATASET_COLORS = {name: AVAILABLE_COLORS[i % len(AVAILABLE_COLORS)] for i, name in enumerate(DATASETS.keys())}

def get_zebra_mapping(db_path, all_uuids):
    uuid_to_zebra = {}
    zebra_counter = 0

    # Only attempt to process the DB if the path is provided and the file exists
    if db_path and os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            df_annots = pd.read_sql_query("SELECT uuid1, uuid2 FROM image_verification WHERE decision='correct'", conn)
            conn.close()

            G = nx.Graph()
            for _, row in df_annots.iterrows():
                G.add_edge(row['uuid1'], row['uuid2'])

            for comp in nx.connected_components(G):
                zebra_counter += 1
                for uuid in comp:
                    uuid_to_zebra[uuid] = f"Zebra_{zebra_counter}"
        except Exception as e:
            print(f"Warning: Could not process database at {db_path}. Error: {e}")

    # Any UUID not found in the graph (or if there is no DB) gets assigned "Unmatched"
    for uuid in all_uuids:
        if uuid not in uuid_to_zebra:
            uuid_to_zebra[uuid] = "Unmatched"
            
    return uuid_to_zebra

def get_database(dataset_name, id_json, db_path):
    if not os.path.exists(id_json):
        raise FileNotFoundError(f"JSON file not found: {id_json}")

    with open(id_json, 'r') as file:
        data = json.load(file)
        
    images = pd.DataFrame(data['images'])
    annotations = pd.DataFrame(data['annotations'])
    
    images = images.rename(columns={"uuid": "image_uuid"})
    zebra_ids = get_zebra_mapping(db_path, annotations['uuid'])
    
    big_data = pd.merge(images, annotations, on="image_uuid", how='inner')
    big_data["zebra_id"] = big_data['uuid'].map(zebra_ids)
    
    # Prepend dataset name to ID to prevent collisions across different datasets
    # e.g., "GGR2024_Zebra_1"
    mask = big_data["zebra_id"] != "Unmatched"
    big_data.loc[mask, "zebra_id"] = f"{dataset_name}_" + big_data.loc[mask, "zebra_id"]
    
    # Keep track of which dataset each row came from
    big_data["dataset"] = dataset_name
    
    return big_data

# --- Data Loading & Cleaning ---
print("Loading datasets...")
all_dfs = []
for name, paths in DATASETS.items():
    try:
        print(f" -> Processing {name}...")
        df_part = get_database(name, paths["json"], paths["db"])
        all_dfs.append(df_part)
    except Exception as e:
        print(f" -> Skipped {name}: {e}")

if not all_dfs:
    raise ValueError("No datasets could be loaded. Please check your DATASETS config paths.")

# Combine all datasets into one master dataframe
df = pd.concat(all_dfs, ignore_index=True)

# Ensure numeric types
df['gps_lat'] = pd.to_numeric(df['gps_lat'], errors='coerce')
df['gps_lon'] = pd.to_numeric(df['gps_lon'], errors='coerce')
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

# Strip out all the -1, -1 missing coordinates from the master dataset
df = df[(df['gps_lat'] != -1) & (df['gps_lon'] != -1)]
df = df[(df['gps_lat'] != -1.0) & (df['gps_lon'] != -1.0)] # Catch floats

# Now define map_df from this perfectly clean master dataframe
map_df = df.copy()
map_df = map_df.sort_values(by=['zebra_id', 'timestamp'])

# Jitter for burst photos
for zebra_id, group in map_df.groupby('zebra_id'):
    if zebra_id != "Unmatched" and len(group) > 1:
        if group['gps_lat'].max() == group['gps_lat'].min() and group['gps_lon'].max() == group['gps_lon'].min():
            map_df.loc[group.index, 'gps_lat'] += np.random.normal(0, 0.00002, size=len(group))
            map_df.loc[group.index, 'gps_lon'] += np.random.normal(0, 0.00002, size=len(group))

map_df = df.dropna(subset=['gps_lat', 'gps_lon']).copy()
map_df = map_df[(map_df['gps_lat'] != -1) & (map_df['gps_lon'] != -1)]
map_df = map_df.sort_values(by=['zebra_id', 'timestamp'])

# Setup Dropdown Choices
dataset_choices = ["All"] + list(DATASETS.keys())
initial_matched_zebras = sorted([z for z in map_df['zebra_id'].unique() if z != "Unmatched"])
initial_zebra_choices = ["All"] + initial_matched_zebras + ["Unmatched"]

def create_map(selected_dataset, selected_zebra):
    plot_df = map_df.copy()
    
    if selected_dataset != "All":
        plot_df = plot_df[plot_df['dataset'] == selected_dataset]
    if selected_zebra != "All":
        plot_df = plot_df[plot_df['zebra_id'] == selected_zebra]

    MAX_UNMATCHED = 2000 
    MAX_TRACKED_PER_ZEBRA = 1000 
    
    unmatched_mask = plot_df['zebra_id'] == "Unmatched"
    if unmatched_mask.sum() > MAX_UNMATCHED:
        tracked_df = plot_df[~unmatched_mask]
        sampled_unmatched = plot_df[unmatched_mask].sample(n=MAX_UNMATCHED, random_state=42)
        plot_df = pd.concat([tracked_df, sampled_unmatched])

    fig = go.Figure()

    for zebra_id, group in plot_df.groupby('zebra_id'):
        dataset_name = group['dataset'].iloc[0]
        
        if len(group) > MAX_TRACKED_PER_ZEBRA and zebra_id != "Unmatched":
            step = len(group) // MAX_TRACKED_PER_ZEBRA
            group = group.iloc[::step]
            
        mode = 'lines+markers' if (len(group) > 1 and zebra_id != "Unmatched") else 'markers'
        
        if zebra_id == "Unmatched":
            marker_style = dict(size=5, color='gray', opacity=0.3)
            line_style = dict(width=0)
        else:
            marker_style = dict(size=12 if selected_zebra != "All" else 8)
            line_style = dict(width=4)
            
            # COLOR LOGIC: Apply dataset color if looking at 'All' datasets
            if selected_dataset == "All":
                ds_color = DATASET_COLORS.get(dataset_name, 'blue')
                marker_style['color'] = ds_color
                line_style['color'] = ds_color
        
        hover_texts = [f"<b>Dataset:</b> {row['dataset']}<br><b>ID:</b> {zebra_id}<br><b>Image:</b> {row['image_uuid']}<br><b>Time:</b> {row['timestamp']}" for _, row in group.iterrows()]
        
        fig.add_trace(go.Scattermapbox(
            mode=mode, lon=group['gps_lon'], lat=group['gps_lat'], text=hover_texts, hoverinfo='text', 
            name=zebra_id, marker=marker_style, line=line_style
        ))

    center_lat = plot_df['gps_lat'].mean() if not plot_df.empty else 0
    center_lon = plot_df['gps_lon'].mean() if not plot_df.empty else 0
    zoom_level = 15 if selected_zebra not in ["All", "Unmatched"] else 7 

    fig.update_layout(
        mapbox=dict(style="open-street-map", zoom=zoom_level, center=dict(lat=center_lat, lon=center_lon)), 
        margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False, height=600
    )
    return fig

# --- Annotation & Image Cropping Function ---
def get_zebra_details(selected_dataset, selected_zebra):
    # Safeguard: Prevent freezing the server by loading thousands of unverified images
    if selected_zebra == "All" or selected_zebra == "Unmatched":
        msg = "Select a specific Tracked Zebra ID to view its cropped images (Unmatched/All contains too many images to safely load)."
        return pd.DataFrame([{"Message": msg}]), []
    
    subset = df.copy()
    if selected_dataset != "All":
        subset = subset[subset['dataset'] == selected_dataset]
        
    subset = subset[subset['zebra_id'] == selected_zebra]
    subset = subset.fillna("") 
    
    display_cols = ['image_path', 'bbox', 'uuid', 'image_uuid', 'timestamp', 'viewpoint', 'clarity_score']
    existing_cols = [c for c in display_cols if c in subset.columns]
    df_out = subset[existing_cols]

    crops = []
    bbox_col = 'bbox' if 'bbox' in subset.columns else None
    
    if bbox_col and 'image_path' in subset.columns:
        for _, row in subset.iterrows():
            path = row['image_path']
            b = row[bbox_col]
            
            if not path or not b:
                continue
                
            try:
                img = Image.open(path)
                if isinstance(b, str):
                    b = ast.literal_eval(b)
                
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    left, top, right, bottom = b[0], b[1], b[2], b[3]
                    if right <= left or bottom <= top:
                        right, bottom = left + right, top + bottom
                        
                    crop_img = img.crop((left, top, right, bottom))
                    crop_img.thumbnail((600, 600))
                    
                    caption = f"Dataset: {row.get('dataset')} | View: {row.get('viewpoint', 'N/A')} | Score: {row.get('clarity_score', 0):.2f}"
                    crops.append((crop_img, caption))
            except Exception as e:
                print(f"Failed to process crop for {path}: {e}")

    return df_out, crops

# --- Dynamic UI Updater ---
def update_zebra_dropdown(selected_dataset):
    """Updates the Zebra ID dropdown choices based on the selected dataset."""
    if selected_dataset == "All":
        subset = map_df
    else:
        subset = map_df[map_df['dataset'] == selected_dataset]
        
    matched = sorted([z for z in subset['zebra_id'].unique() if z != "Unmatched"])
    new_choices = ["All"] + matched + ["Unmatched"]
    return gr.update(choices=new_choices, value="All")


# --- Gradio Interface ---
with gr.Blocks() as app:
    gr.Markdown("# 🦓 Grevy's Zebra Analysis Toolkit")
    
    with gr.Row():
        dataset_dropdown = gr.Dropdown(choices=dataset_choices, value="All", label="1. Filter by Dataset", interactive=True)
        zebra_dropdown = gr.Dropdown(choices=initial_zebra_choices, value="All", label="2. Filter by Zebra ID", interactive=True)
    
    map_output = gr.Plot()
    
    gr.Markdown("### 📸 Sighting Verification")
    gallery_output = gr.Gallery(label="Zebra Crops", show_label=True, elem_id="gallery", columns=[3], rows=[2], object_fit="contain", height="auto")
    annot_output = gr.Dataframe(interactive=False, wrap=True)
    
    # Wire the dropdowns
    dataset_dropdown.change(fn=update_zebra_dropdown, inputs=dataset_dropdown, outputs=zebra_dropdown)
    
    # Both dropdowns trigger the map and gallery updates
    dataset_dropdown.change(fn=create_map, inputs=[dataset_dropdown, zebra_dropdown], outputs=map_output)
    zebra_dropdown.change(fn=create_map, inputs=[dataset_dropdown, zebra_dropdown], outputs=map_output)
    
    dataset_dropdown.change(fn=get_zebra_details, inputs=[dataset_dropdown, zebra_dropdown], outputs=[annot_output, gallery_output])
    zebra_dropdown.change(fn=get_zebra_details, inputs=[dataset_dropdown, zebra_dropdown], outputs=[annot_output, gallery_output])
    
    # Load initial view
    app.load(fn=create_map, inputs=[dataset_dropdown, zebra_dropdown], outputs=map_output)

if __name__ == "__main__":
    app.launch(share=True)