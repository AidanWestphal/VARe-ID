# ID Region

The purpose of id_region is to focus in on the specific region of an animal that is needed for identification using a finetuned model (YOLOv8 by defualt).

## How it works

1. Load crops of targets gathered from previous steps
2. Run each crop through finetuned model to focus in on relevant region
3. Filter regions using NMS, aspect ratio and clarity algorithms

## NMS (Non-Maximum Supression)

Regions that overlap more than the amount specified in the id region config file have NMS applied to them. For the case of zebra identification, by default, aspect ratio is used for the NMS metric.

## Clarity Score

Images that are not clear enough for identification must be filtered out. The default metric used to determine the clarity of an image is the sum of the edge density as determined by Canny edge detection and the normalized Laplacian variance.