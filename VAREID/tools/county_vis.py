import sqlite3
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import json
import gradio as gr

def get_zebra_mapping(db_path, all_uuids):
    conn = sqlite3.connect(db_path)
    df_annots = pd.read_sql_query("SELECT uuid1, uuid2 FROM image_verification WHERE decision='correct'", conn)
    conn.close()

    G = nx.Graph()
    for _, row in df_annots.iterrows():
        G.add_edge(row['uuid1'], row['uuid2'])

    uuid_to_zebra = {}
    zebra_counter = 0
    
    # 1. Group the matched zebras
    for comp in nx.connected_components(G):
        zebra_counter += 1
        zebra_id = f"Zebra_{zebra_counter}"
        for uuid in comp:
            uuid_to_zebra[uuid] = zebra_id
            
    # 2. THE FIX: Assign all singletons to a single "Unmatched" group
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

# --- Data Loading ---
id_json_path = "/fs/ess/PAS2136/ggr_data/results/GGR2024_encounter_grouping/id_region/id_regions_filtered.json"
db_path = "/fs/ess/PAS2136/ggr_data/results/GGR2024_encounter_grouping/annots.db"

print("Loading and cleaning dataset...")
df = get_database(id_json_path, db_path)

# Format to numeric to prevent blank maps
df['gps_lat'] = pd.to_numeric(df['gps_lat'], errors='coerce')
df['gps_lon'] = pd.to_numeric(df['gps_lon'], errors='coerce')
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

# Clean and sort data
df = df.dropna(subset=['gps_lat', 'gps_lon'])
df = df[(df['gps_lat'] != -1) & (df['gps_lon'] != -1)]
df = df.sort_values(by=['zebra_id', 'timestamp'])

# Build dropdown choices (Filter out 'Unmatched' so it goes at the end)
matched_zebras = sorted([z for z in df['zebra_id'].unique() if z != "Unmatched"])
unique_zebras = ["All"] + matched_zebras + ["Unmatched"]

print(f"Loaded {len(df)} total sightings. {len(matched_zebras)} unique tracked zebras found.")

# --- Plotting Function ---
def create_map(selected_zebra):
    plot_df = df.copy()
    
    if selected_zebra != "All":
        plot_df = plot_df[plot_df['zebra_id'] == selected_zebra]

    fig = go.Figure()

    for zebra_id, group in plot_df.groupby('zebra_id'):
        # Only draw lines if it's a tracked zebra, otherwise just dots
        mode = 'lines+markers' if (len(group) > 1 and zebra_id != "Unmatched") else 'markers'
        
        # Style unmatched zebras differently (smaller, grey dots) to reduce clutter
        if zebra_id == "Unmatched":
            marker_style = dict(size=5, color='gray', opacity=0.5)
            line_style = dict(width=0)
        else:
            marker_style = dict(size=10 if selected_zebra != "All" else 7)
            line_style = dict(width=3)
        
        hover_texts = [
            f"<b>Zebra ID:</b> {zebra_id}<br>"
            f"<b>Image UUID:</b> {row['image_uuid']}<br>"
            f"<b>Viewpoint:</b> {row.get('viewpoint', 'N/A')}<br>"
            f"<b>CA Score:</b> {row.get('CA_score', 0):.2f}"
            for _, row in group.iterrows()
        ]
        
        fig.add_trace(go.Scattermapbox(
            mode=mode,
            lon=group['gps_lon'],
            lat=group['gps_lat'],
            text=hover_texts,
            hoverinfo='text',
            name=zebra_id,
            marker=marker_style,
            line=line_style
        ))

    center_lat = plot_df['gps_lat'].mean() if not plot_df.empty else 0
    center_lon = plot_df['gps_lon'].mean() if not plot_df.empty else 0
    zoom_level = 10 if selected_zebra not in ["All", "Unmatched"] else 7 

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=zoom_level, 
            center=dict(lat=center_lat, lon=center_lon)
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        height=700 
    )
    
    return fig

# --- Gradio Interface ---
with gr.Blocks() as app:
    gr.Markdown("# 🦓 Grevy's Zebra Trajectory Tracker")
    gr.Markdown("Select a specific individual from the dropdown below to isolate their movements across the counties.")
    
    with gr.Row():
        zebra_dropdown = gr.Dropdown(choices=unique_zebras, value="All", label="Filter by Zebra ID", interactive=True)
    
    map_output = gr.Plot()
    
    zebra_dropdown.change(fn=create_map, inputs=zebra_dropdown, outputs=map_output)
    app.load(fn=create_map, inputs=zebra_dropdown, outputs=map_output)

if __name__ == "__main__":
    app.launch(share=True)