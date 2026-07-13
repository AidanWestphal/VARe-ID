import json
import os
import yaml
import math

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.workflow_funcs import build_config, decode_config

config = build_config(load_config('config.yaml'))
    
print("PIPELINE STATS:")
    
    
print("     Import:")
with open(config['image_out_path']) as f:
    image_data = json.load(f)['images']
    
print(f"        {len(image_data)} images imported")

with open(config['import_logs']) as f:
    content = f.read()
    
err_msg = "can't open file for writing: permission denied"
print(f"        {content.count(err_msg)} images failed to open due to permissions")
print(f"        {content.count('found QR code')} QR codes found")


print("\n     Detection:")
with open(config['dt_image_out_path']) as f:
    dt_data = json.load(f)['annotations']
    
print(f"        {len(dt_data)} bboxes found")


print("\n     Species Classifier:")
with open(config['si_out_path']) as f:
    data = json.load(f)
    si_data = data['annotations']
    species = data['categories']
    species = { x['id']: x['species'] for x in species}
    
species_tracker = {}
for i in range(len(si_data)):
    species_tracker[species[si_data[i]['category_id']]] = species_tracker.get(species[si_data[i]['category_id']], 0) + 1
    
species_tracker = dict(sorted(species_tracker.items(), key=lambda x: x[1], reverse=True))
    
for key in species_tracker.keys():
    print(f"        {key}: {species_tracker[key]}")


print("\n     Viewpoint Classifier:")
with open(config['vc_out_path']) as f:
    vc_data = json.load(f)['annotations']
    
viewpoint_tracker = {}
for i in range(len(vc_data)):
    viewpoint_tracker[vc_data[i]['viewpoint']] = viewpoint_tracker.get(vc_data[i]['viewpoint'], 0) + 1
    
viewpoint_tracker = dict(sorted(viewpoint_tracker.items(), key=lambda x: x[1], reverse=True))
    
accepted_viewpoint = 0
for key in viewpoint_tracker.keys():
    print(f"        {key}: {viewpoint_tracker[key]}")
    if 'right' in key:
        accepted_viewpoint += viewpoint_tracker[key]
print(f"\n        {accepted_viewpoint} viewpoints accepted")
    
    
print("\n     IA Classifier:")
with open(config['ia_out_path']) as f:
    ia_data = json.load(f)['annotations']
    
annot_regions = 0
non_annot_regions = 0

for i in range(len(ia_data)):
    if ia_data[i]['annotations_census']:
        annot_regions += 1
    else:
        non_annot_regions += 1
    
print(f"        {annot_regions} possible identifiable regions found")
print(f"        {non_annot_regions} regions filtered out")


print("\n     ID Region:")
with open(config['idr_out_path']) as f:
    idr_data = json.load(f)['annotations']
    
annot_regions = 0
non_annot_regions = 0

for i in range(len(idr_data)):
    if idr_data[i]['annotations_census']:
        annot_regions += 1
    else:
        non_annot_regions += 1
    
print(f"        {annot_regions} identifiable regions found")
print(f"        {non_annot_regions} regions filtered out")


print("\n     LCA:")
with open(config['lca_out_path']) as f:
    lca_data = json.load(f)['annotations']
    
lca_tracker = {}
for i in range(len(lca_data)):
    lca_tracker[lca_data[i]['cluster_id']] = lca_tracker.get(lca_data[i]['cluster_id'], 0) + 1
    
lca_tracker = dict(sorted(lca_tracker.items(), key=lambda x: x[1], reverse=True))
    
match_count = 0
for key in lca_tracker.keys():
    match_count += lca_tracker[key]*(lca_tracker[key]-1)//2
    if lca_tracker[key] == 1:
        break
        
print(f"        {len(lca_tracker)} individuals found")
print(f"        {match_count} matches found")

print(f"\n        Top individual sightings: ")
count = 0
for key in lca_tracker.keys():
    if count == 10:
        break
    print(f"           {key}: {lca_tracker[key]}")
    count += 1