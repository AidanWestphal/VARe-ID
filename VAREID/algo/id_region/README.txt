The purpose of id_region is to focus in on the specific region of an animal that is needed for identification using 
embeddings from miew id.The id_region model takes a crop of a target animal and generates a new tighter bounding box 
around the relevant identification region. The code, by default, is designed to use a finetuned version of YOLOvn8. 
This step will need a model (taken from id_region_model in the main config) that is taylored to the specific animal 
and region being identified. Non-maximum supression is used to remove overlapping bounding boxes.