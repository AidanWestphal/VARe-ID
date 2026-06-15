# IA Classifier (Identifiable annotation classifier)

The IA classifier takes in a target and returns a score representing it's identifiability as determined by a finetuned model (default ResNet50).

## How it works

1. Load crops of targets gathered from previous steps
2. Run each crop through finetuned model to gather identifiability score
3. Accpet or reject target based on threshold specified in the IA classifier configuration file
4. Filter images that were rejected by the previous step

## Output

- ia classifier
    - categories
        - id
        - species
    - images
        - uuid # corresponds to image_uuid in annotations 
        - timestamp
        - gps_lat
        - gps_lon
        - image_path
    - annotations
        - uuid
        - image_uuid
        - bbox #xywh
        - confidence
        - detection_class
        - tracking_id
        - category_id
        - viewpoint
        - CA_score
        - individual_id
        - annotations_census

- ia filtered classifier # same as ia classifier output but with annotations_census=false filtered out and column removed
    - categories
        - id
        - species
    - images
        - uuid # corresponds to image_uuid in annotations 
        - timestamp
        - gps_lat
        - gps_lon
        - image_path
    - annotations
        - uuid
        - image_uuid
        - bbox #xywh
        - confidence
        - detection_class
        - tracking_id
        - category_id
        - viewpoint
        - CA_score
        - individual_id