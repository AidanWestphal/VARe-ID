import sqlite3
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as pc
import json
import gradio as gr
import numpy as np
import ast
import os
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASETS = {
    "GGR2024": {
        "json": "/fs/ess/PAS2136/ggr_data/results/GGR2024_fixed/counties.json",
        "db": "/fs/ess/PAS2136/ggr_data/results/GGR2024_fixed_encounter/annots.db"
    },
    "GGR2018": {
        "json": "/fs/ess/PAS2136/ggr_data/results/GGR2018_ilan/county.json",
        "db": None
    }
}

# Match colors to your provided map: Blue for 2024, Orange/Red for 2018
DATASET_COLORS = {
    "GGR2024": "#636EFA", # Blue
    "GGR2018": "#EF553B"  # Orange/Red
}

HEATMAP_SCALES = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']
DATASET_HEATMAP_SCALES = {name: HEATMAP_SCALES[i % len(HEATMAP_SCALES)] for i, name in enumerate(DATASETS.keys())}

def get_zebra_mapping(db_path, all_uuids):
    uuid_to_zebra = {}
    zebra_counter = 0
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
            print(f"Warning: DB Error: {e}")
    for uuid in all_uuids:
        if uuid not in uuid_to_zebra:
            uuid_to_zebra[uuid] = "Unmatched"
    return uuid_to_zebra

def get_database(dataset_name, id_json, db_path):
    with open(id_json, 'r') as file:
        data = json.load(file)
    images = pd.DataFrame(data['images'])
    annotations = pd.DataFrame(data['annotations'])
    
    images['county'] = images.get('county', pd.Series(["Unknown"]*len(images))).fillna("Unknown")
    images['land tenure'] = images.get('land tenure', pd.Series(["Unknown"]*len(images))).fillna("Unknown")
        
    images = images.rename(columns={"uuid": "image_uuid"})
    zebra_ids = get_zebra_mapping(db_path, annotations['uuid'])
    big_data = pd.merge(images, annotations, on="image_uuid", how='inner')
    big_data["zebra_id"] = big_data['uuid'].map(zebra_ids)
    
    mask = big_data["zebra_id"] != "Unmatched"
    big_data.loc[mask, "zebra_id"] = f"{dataset_name}_" + big_data.loc[mask, "zebra_id"]
    big_data["dataset"] = dataset_name
    return big_data

# --- Data Loading ---
all_dfs = []
for name, paths in DATASETS.items():
    try:
        all_dfs.append(get_database(name, paths["json"], paths["db"]))
    except: pass

df = pd.concat(all_dfs, ignore_index=True)
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
map_df = df[(df['gps_lat'] != -1) & (df['gps_lat'].notnull())].copy()
map_df = map_df.sort_values(by=['zebra_id', 'timestamp'])

# ==========================================
# MISSING VARIABLES RESTORED HERE
# ==========================================
dataset_choices = ["All"] + list(DATASETS.keys())
initial_matched_zebras = sorted([z for z in map_df['zebra_id'].unique() if z != "Unmatched"])
initial_zebra_choices = ["All"] + initial_matched_zebras + ["Unmatched"]
# ==========================================

# --- Map Logic ---
def create_map(selected_dataset, selected_zebra, show_heatmap):
    plot_df = map_df.copy()
    if selected_dataset != "All": plot_df = plot_df[plot_df['dataset'] == selected_dataset]
    if selected_zebra != "All": plot_df = plot_df[plot_df['zebra_id'] == selected_zebra]

    fig = go.Figure()
    if show_heatmap:
        for ds, group in plot_df.groupby('dataset'):
            fig.add_trace(go.Densitymapbox(lat=group['gps_lat'], lon=group['gps_lon'], z=np.ones(len(group)),
                                          radius=15, colorscale=DATASET_HEATMAP_SCALES.get(ds, 'Inferno'), opacity=0.6, showscale=False))
    else:
        for (ds, zid), group in plot_df.groupby(['dataset', 'zebra_id']):
            color = DATASET_COLORS.get(ds, 'gray')
            marker = dict(size=7, opacity=0.7, color=color) if zid == "Unmatched" else dict(size=10, color=color)
            
            hover_texts = [f"<b>Dataset:</b> {row['dataset']}<br><b>ID:</b> {zid}<br><b>County:</b> {row.get('county', 'Unknown')}<br><b>Land Tenure:</b> {row.get('land tenure', 'Unknown')}<br><b>Date:</b> {row['date']}" for _, row in group.iterrows()]
            
            fig.add_trace(go.Scattermapbox(mode='markers', lon=group['gps_lon'], lat=group['gps_lat'], 
                                          text=hover_texts, hoverinfo='text',
                                          name=f"{zid} ({ds})", marker=marker))

    fig.update_layout(mapbox=dict(style="open-street-map", zoom=7, center=dict(lat=map_df['gps_lat'].mean(), lon=map_df['gps_lon'].mean())),
                      margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False, height=500)
    return fig

# --- Histogram Logic (Dataset Separated) ---
def create_histogram(selected_dataset):
    plot_df = map_df.copy()
    if selected_dataset != "All": plot_df = plot_df[plot_df['dataset'] == selected_dataset]
    
    # Calculate counts
    matched = plot_df[plot_df['zebra_id'] != "Unmatched"].groupby(['dataset', 'county'])['zebra_id'].nunique()
    unmatched = plot_df[plot_df['zebra_id'] == "Unmatched"].groupby(['dataset', 'county']).size()
    total = (matched.add(unmatched, fill_value=0)).reset_index(name='count')
    total = total[total['county'] != "Unknown"]

    fig = go.Figure()
    for ds in total['dataset'].unique():
        ds_data = total[total['dataset'] == ds]
        fig.add_trace(go.Bar(x=ds_data['county'], y=ds_data['count'], name=ds, 
                            marker_color=DATASET_COLORS.get(ds), text=ds_data['count'], textposition='auto'))

    fig.update_layout(title="Zebras per County", barmode='group', template="plotly_white", height=400)
    return fig

# --- FACETED Land Tenure Logic ---
def create_lt_histogram(selected_dataset):
    plot_df = map_df.copy()
    if selected_dataset != "All": plot_df = plot_df[plot_df['dataset'] == selected_dataset]
    
    # Calculate unique IDs per dataset/county/land-tenure
    matched = plot_df[plot_df['zebra_id'] != "Unmatched"].groupby(['dataset', 'county', 'land tenure'])['zebra_id'].nunique()
    unmatched = plot_df[plot_df['zebra_id'] == "Unmatched"].groupby(['dataset', 'county', 'land tenure']).size()
    total = (matched.add(unmatched, fill_value=0)).reset_index(name='count')
    total = total[(total['county'] != "Unknown") & (total['land tenure'] != "Unknown")]

    counties = sorted(total['county'].unique())
    if not counties: return go.Figure()

    # Create subplots: one row per county
    fig = make_subplots(rows=len(counties), cols=1, subplot_titles=counties, vertical_spacing=0.05)

    for i, county in enumerate(counties):
        county_data = total[total['county'] == county]
        for ds in ["GGR2018", "GGR2024"]:
            ds_data = county_data[county_data['dataset'] == ds]
            if not ds_data.empty:
                fig.add_trace(
                    go.Bar(x=ds_data['land tenure'], y=ds_data['count'], name=ds,
                           marker_color=DATASET_COLORS.get(ds), showlegend=(i == 0)),
                    row=i+1, col=1
                )

    fig.update_layout(height=400 * len(counties), title_text="Land Tenure breakdown per County", 
                      barmode='group', template="plotly_white", showlegend=True)
    fig.update_annotations(font_size=14)
    return fig

# --- Gradio Interface ---
with gr.Blocks() as app:
    gr.Markdown("# 🦓 Grevy's Zebra Spacial Bias Analysis")
    with gr.Row():
        dataset_dropdown = gr.Dropdown(choices=dataset_choices, value="All", label="Dataset Filter")
        zebra_dropdown = gr.Dropdown(choices=initial_zebra_choices, value="All", label="Zebra ID Filter")
        heatmap_toggle = gr.Checkbox(label="Show Heatmap", value=False)
    
    map_output = gr.Plot()
    
    with gr.Tabs():
        with gr.TabItem("County Distribution"):
            histogram_output = gr.Plot()
        with gr.TabItem("Land Tenure by County"):
            # Subplots can be tall, so we display this in its own tab
            lt_histogram_output = gr.Plot()

    dataset_dropdown.change(create_map, [dataset_dropdown, zebra_dropdown, heatmap_toggle], map_output)
    dataset_dropdown.change(create_histogram, dataset_dropdown, histogram_output)
    dataset_dropdown.change(create_lt_histogram, dataset_dropdown, lt_histogram_output)
    
    app.load(create_map, [dataset_dropdown, zebra_dropdown, heatmap_toggle], map_output)
    app.load(create_histogram, dataset_dropdown, histogram_output)
    app.load(create_lt_histogram, dataset_dropdown, lt_histogram_output)

if __name__ == "__main__":
    app.launch(share=True)