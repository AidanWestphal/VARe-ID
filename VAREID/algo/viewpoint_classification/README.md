# Viewpoint Classification

The viewpoint classifier is used to give a description of how the target is pictured. Viewpoint classifer uses a finetuned ResNet50 to return a distribution of predictions across many possible different base viewpoints as specified in the viewpoint configuration file. It then accepts or rejects the viewpoints and combines them into one final overall viewpoint.

## How it works

1. Load crops of targets gathered from previous steps
2. Run each crop through finetuned model to gather scores for each base viewpoint
3. Accpet or reject base viewpoints based on threshold specified in the viewpoint configuration file
4. Combine base viewpoints into one viewpoint 