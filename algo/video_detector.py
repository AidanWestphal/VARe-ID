import cv2
import csv
import os
import re
import yaml
from ultralytics import YOLO

def extract_frames_from_video(video_path, output_dir, frame_rate=8, max_frames=2000):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened(): 
        print("Error opening video file")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the original frame rate of the video
    original_frame_rate = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_frame_rate
    print(f"Original frame rate: {original_frame_rate}")
    print(f"Total frames in the video: {total_frames}")
    print(f"Video duration: {duration} seconds")

    frame_interval = round(original_frame_rate / frame_rate)
    print(f"Frame interval for extraction: {frame_interval}")

    extracted_frames = 0
    current_frame = 0

    while video.isOpened() and extracted_frames < max_frames:
        ret, frame = video.read()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            filename = os.path.join(output_dir, f"DJI_0102_{extracted_frames}.jpg")
            cv2.imwrite(filename, frame)
            #print(f"Extracted frame {current_frame} as {filename}")
            extracted_frames += 1
        
        current_frame += 1

    video.release()
    cv2.destroyAllWindows()

    print(f"Total frames extracted: {extracted_frames}")
    
def detect_and_track_objects(frames_dir, model_path, output_csv, output_visual_dir):
    # Load the YOLO model
    model = YOLO(model_path)

    # Ensure the output directory exists
    os.makedirs(output_visual_dir, exist_ok=True)

    # Function to extract the last number from the filename
    def extract_last_number(filename):
        match = re.search(r'DJI_0102_(\d+)\.jpg$', filename)
        return int(match.group(1)) if match else None

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['frame_name', 'bounding_box', 'tracking_id', 'confidence', 'detection_class'])

        # Process each frame
        frame_names = sorted(
            [f for f in os.listdir(frames_dir) if f.startswith("DJI_0102_") and f.endswith(".jpg")],
            key=extract_last_number
        )

        for frame_name in frame_names:
            frame_path = os.path.join(frames_dir, frame_name)
            frame = cv2.imread(frame_path)

            # Run YOLO detection and tracking
            results = model.track(frame, imgsz=3000, persist=True)

            # Extract detections and tracking information
            for result in results:
                for box in result.boxes:
                    # Flatten the bbox array to ensure it's a list of four elements
                    bbox = box.xyxy.cpu().numpy().astype(int).flatten().tolist()
                    confidence = float(box.conf.cpu().numpy()) if box.conf is not None else 0.0
                    detection_class = int(box.cls.cpu().numpy()) if box.cls is not None else -1
                    tracking_id = int(box.id.cpu().numpy()) if box.id is not None else None
                    writer.writerow([frame_name, bbox, tracking_id, confidence, detection_class])

                    # Draw bounding boxes and tracking IDs on the frame
                    x1, y1, x2, y2 = bbox
                    label = f"ID {tracking_id}"
                    color = (0, 255, 0)  # Green color for bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                    )

            # Save the frame with visualizations
            output_frame_path = os.path.join(output_visual_dir, frame_name)
            cv2.imwrite(output_frame_path, frame)

    print(f'Results saved to {output_csv}')
    print(f'Visualizations saved to {output_visual_dir}')

def filter_animal_detections(input_file, output_file, animal_class_ids):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            detection_class = int(row['detection_class'])
            if detection_class in animal_class_ids:
                writer.writerow(row)

    print(f'Filtered Results saved to {output_file}')


if __name__ == '__main__':
    # Load the configuration file
    with open('detection_rounded.yaml', 'r') as file:
        config = yaml.safe_load(file)

    video_path = config['video_path']
    output_dir = config['output_dir']
    frame_rate = config['frame_rate']
    max_frames = config['max_frames']
    model_path = config['model_path']
    output_csv = config['output_csv']
    output_file = config['output_file']
    animal_class_ids = config['animal_class_ids']
    output_visual_dir = config['output_visual_dir']

    # Extract frames from video using your specified logic
    extract_frames_from_video(video_path, output_dir, frame_rate, max_frames)

    # Detect and track objects with visualization
    detect_and_track_objects(output_dir, model_path, output_csv, output_visual_dir)

    # Filter animal detections
    filter_animal_detections(output_csv, output_file, animal_class_ids)
