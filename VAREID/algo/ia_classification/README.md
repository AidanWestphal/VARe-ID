# IA Classifier (Identifiable annotation classifier)

The IA classifier takes in a target and returns a score representing it's identifiability as determined by a finetuned model (default ResNet50).

## How it works

1. Load crops of targets gathered from previous steps
2. Run each crop through finetuned model to gather identifiability score
3. Accpet or reject target based on threshold specified in the IA classifier configuration file
4. Filter images that were rejected by the previous step