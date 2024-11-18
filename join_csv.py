# import os
# import pandas as pd
#
# def join_ground_truth_predicted(ground_truth_df, predicted_df):
#     # Ensure that the filenames from the ground truth are in the expected format
#     ground_truth_df['filename'] = ground_truth_df['filename'].apply(lambda x: x.split('.')[0])  # Remove .jpg
#
#     # Join on the modified filename and the image UUID
#     merged_df = ground_truth_df.merge(predicted_df, left_on='filename', right_on='image uuid', how='left')
#
#     return merged_df
#
# def save_annotations_to_csv(annotations_dict, file_path):
#     annotations = annotations_dict['annotations']
#     df = pd.DataFrame(annotations)
#     df.to_csv(file_path, index=False)
#
#
# if __name__ == '__main__':
#     yolo_df_path = "/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/test_dataset/annots/pred_annots_yolov10l.csv"
#     test_df_path = "/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/test_dataset/GZCD_Cleaned2.csv"
#
#     # Load CSV files into DataFrames
#     yolo_df = pd.read_csv(yolo_df_path)
#     test_df = pd.read_csv(test_df_path)
#
#     # Join the ground truth and predictions
#     ground_truth_df = join_ground_truth_predicted(test_df, yolo_df)
#
#     # Save joined annotations if needed
#     joined_annotations_path = "/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/test_dataset/annots/joined_annotations.csv"
#     save_annotations_to_csv({'annotations': ground_truth_df.to_dict('records')}, joined_annotations_path)
#
#     print(f"Joined annotations saved to: {joined_annotations_path}")
#


import os
import pandas as pd
import ast

def join_and_save_annotations(ground_truth_path, predicted_path, save_path):
    # Load CSV files into DataFrames
    ground_truth_df = pd.read_csv(ground_truth_path)
    predicted_df = pd.read_csv(predicted_path)

    # Ensure that the filenames from the ground truth are in the expected format
    ground_truth_df['filename'] = ground_truth_df['filename'].apply(lambda x: x.split('.')[0])  # Remove .jpg if present

    # Convert 'bbox' and 'bbox_y' columns from string to list
    if 'bbox' in ground_truth_df.columns:
        ground_truth_df['bbox'] = ground_truth_df['bbox'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else x)
    if 'bbox_y' in predicted_df.columns:
        predicted_df['bbox_y'] = predicted_df['bbox_y'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else x)

    # Join on the modified filename and the image UUID
    merged_df = ground_truth_df.merge(predicted_df, left_on='filename', right_on='image uuid', how='left')

    # Save joined annotations to CSV
    merged_df.to_csv(save_path, index=False)
    print(f"Joined annotations saved to: {save_path}")

if __name__ == '__main__':
    yolo_df_path = "/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/test_dataset/annots/pred_annots_yolov10l.csv"
    test_df_path = "/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/test_dataset/GZCD_Cleaned2.csv"
    joined_annotations_path = "/Users/jaeseok/Developer/GGR/Imageomics-Species-ReID/test_dataset/annots/joined_annotations.csv"

    # Perform the join and save to CSV
    join_and_save_annotations(test_df_path, yolo_df_path, joined_annotations_path)


