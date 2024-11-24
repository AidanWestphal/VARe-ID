import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Load the CSV file
csv_path = '/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/test_dataset/test_dataset/ca_classifier/final_output_with_softmax_and_census.csv'
df = pd.read_csv(csv_path)

# Directory where images are stored
image_dir = '/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/test_dataset/images'


# Function to display image with bounding box and annotations
def display_images_with_info(row):
    # Construct the image path
    image_path = os.path.join(image_dir, row['image uuid'] + '.jpg')

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    # Convert color from BGR to RGB (matplotlib expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a plot for the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Extract bounding box information and draw it
    bbox = eval(row['bbox'])  # Convert string to tuple
    x, y, w, h = bbox
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))

    # Add descriptive information
    plt.title(f"Image UUID: {row['image uuid']}")
    plt.xlabel(f"""
    bbox pred score: {row['bbox pred score']}
    category id: {row['category id']}
    annot species: {row['annot species']}
    annot census: {row['annot census']}
    predicted_viewpoint: {row['predicted_viewpoint']}
    """, fontsize=10, loc='left')

    plt.axis('off')
    plt.show()


# Display each image with details
for _, row in df.iterrows():
    display_images_with_info(row)
