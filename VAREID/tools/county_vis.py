import sqlite3
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import json
import gradio as gr
import numpy as np
import ast
from PIL import Image

def get_zebra_mapping(db_path, all_uuids):
    conn = sqlite3.connect(db_path)
    df_annots = pd.read_sql_query("SELECT uuid1, uuid2 FROM image_verification WHERE decision='correct'", conn)
    conn.close()

    G = nx.Graph()
    for _, row in df_annots.iterrows():
        G.add_edge(row['uuid1'], row['uuid2'])

    uuid_to_zebra = {}
    zebra_counter = 0
    for comp in nx.connected_components(G):
        zebra_counter += 1
        for uuid in comp:
            uuid_to_zebra[uuid] = f"Zebra_{zebra_counter}"
            
    for uuid in all_uuids:
        if uuid not in uuid_to_zebra:
            uuid_to_zebra[uuid] = "Unmatched"
            
    return uuid_to_zebra

def get_database(id_json, db_path):
    with open(id_json, 'r') as file:
        data = json.load(file)
    images = pd.DataFrame(data['images'])
    annotations = pd.DataFrame(data['annotations'])
    
    images = images.rename(columns={"uuid": "image_uuid"})
    zebra_ids = get_zebra_mapping(db_path, annotations['uuid'])
    
    big_data = pd.merge(images, annotations, on="image_uuid", how='inner')
    big_data["zebra_id"] = big_data['uuid'].map(zebra_ids)
    
    return big_data

# --- Data Loading & Cleaning ---
id_json_path = "/fs/ess/PAS2136/ggr_data/results/GGR2024_encounter_grouping/id_region/id_regions_filtered.json"
db_path = "/fs/ess/PAS2136/ggr_data/results/GGR2024_encounter_grouping/annots.db"

print("Loading dataset...")
df = get_database(id_json_path, db_path)

# Ensure numeric GPS for the map
df['gps_lat'] = pd.to_numeric(df['gps_lat'], errors='coerce')
df['gps_lon'] = pd.to_numeric(df['gps_lon'], errors='coerce')
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

map_df = df.dropna(subset=['gps_lat', 'gps_lon']).copy()
map_df = map_df[(map_df['gps_lat'] != -1) & (map_df['gps_lon'] != -1)]
map_df = map_df.sort_values(by=['zebra_id', 'timestamp'])

# Jitter for burst photos
for zebra_id, group in map_df.groupby('zebra_id'):
    if zebra_id != "Unmatched" and len(group) > 1:
        if group['gps_lat'].max() == group['gps_lat'].min() and group['gps_lon'].max() == group['gps_lon'].min():
            map_df.loc[group.index, 'gps_lat'] += np.random.normal(0, 0.00002, size=len(group))
            map_df.loc[group.index, 'gps_lon'] += np.random.normal(0, 0.00002, size=len(group))

matched_zebras = sorted([z for z in map_df['zebra_id'].unique() if z != "Unmatched"])
unique_zebras = ["All"] + matched_zebras + ["Unmatched"]

def create_map(selected_zebra):
    plot_df = map_df.copy()
    if selected_zebra != "All":
        plot_df = plot_df[plot_df['zebra_id'] == selected_zebra]

    fig = go.Figure()

    for zebra_id, group in plot_df.groupby('zebra_id'):
        mode = 'lines+markers' if (len(group) > 1 and zebra_id != "Unmatched") else 'markers'
        if zebra_id == "Unmatched":
            marker_style = dict(size=5, color='gray', opacity=0.3)
            line_style = dict(width=0)
        else:
            marker_style = dict(size=12 if selected_zebra != "All" else 8)
            line_style = dict(width=4)
        
        hover_texts = [f"<b>ID:</b> {zebra_id}<br><b>Image:</b> {row['image_uuid']}<br><b>Time:</b> {row['timestamp']}" for _, row in group.iterrows()]
        
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

def get_zebra_details(selected_zebra):
    if selected_zebra == "All":
        # Return empty data when "All" is selected to prevent freezing the server
        return pd.DataFrame([{"Message": "Select a specific Zebra ID to view its cropped images and data."}]), []
    
    subset = df[df['zebra_id'] == selected_zebra].copy()
    subset = subset.fillna("") 
    
    display_cols = ['image_path', 'bbox', 'uuid', 'image_uuid', 'timestamp', 'viewpoint', 'CA_score']
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
                # 1. Open the original high-res image
                img = Image.open(path)
                
                # 2. Parse the bounding box list
                if isinstance(b, str):
                    b = ast.literal_eval(b)
                
                # 3. Crop the image based on coordinates
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    left, top, right, bottom = b[0], b[1], b[2], b[3]
                    
                    # Safety check: if width/height were provided instead of xmax/ymax
                    if right <= left or bottom <= top:
                        right = left + right
                        bottom = top + bottom
                        
                    crop_img = img.crop((left, top, right, bottom))
                    
                    # Resize the crop slightly so the web UI loads instantly
                    crop_img.thumbnail((600, 600))
                    
                    # Create a caption for the gallery image
                    caption = f"View: {row.get('viewpoint', 'N/A')} | Score: {row.get('CA_score', 0):.2f}"
                    crops.append((crop_img, caption))
                    
            except Exception as e:
                print(f"Failed to process crop for {path}: {e}")

    return df_out, crops

# --- Gradio Interface ---
with gr.Blocks() as app:
    gr.Markdown("# 🦓 Grevy's Zebra Analysis Toolkit")
    
    with gr.Row():
        zebra_dropdown = gr.Dropdown(choices=unique_zebras, value="All", label="Filter by Zebra ID", interactive=True)
    
    map_output = gr.Plot()
    
    gr.Markdown("### 📸 Sighting Verification")
    gr.Markdown("When a specific zebra is selected, its original images are loaded from the cluster, cropped to their exact bounding box, and displayed here.")
    
    # NEW: Gradio Gallery Component
    gallery_output = gr.Gallery(label="Zebra Crops", show_label=True, elem_id="gallery", columns=[3], rows=[2], object_fit="contain", height="auto")
    
    # Existing Dataframe Component
    annot_output = gr.Dataframe(interactive=False, wrap=True)
    
    # Wire the dropdown to update the Map, the Gallery, AND the Dataframe
    zebra_dropdown.change(fn=create_map, inputs=zebra_dropdown, outputs=map_output)
    zebra_dropdown.change(fn=get_zebra_details, inputs=zebra_dropdown, outputs=[annot_output, gallery_output])
    
    # Load initial view
    app.load(fn=create_map, inputs=zebra_dropdown, outputs=map_output)

if __name__ == "__main__":
    app.launch(share=True)