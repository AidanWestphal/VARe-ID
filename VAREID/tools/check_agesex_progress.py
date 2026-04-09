import sqlite3
import pandas as pd
import os
import yaml


# Update this if your database is named differently in the yaml

with open('age_sex_labeler.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

DB_PATH = config_data['db_path']

def check_progress(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Could not find database at '{db_path}'.")
        print("Make sure you are running this script in the same directory as the database.")
        return

    try:
        conn = sqlite3.connect(db_path)
        
        # Load tables into pandas for easy querying
        df_labels = pd.read_sql_query("SELECT * FROM cluster_labels", conn)
        df_annots = pd.read_sql_query("SELECT * FROM annotation_status", conn)
        
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")
        return

    if df_labels.empty:
        print("Database is empty. No clusters to track.")
        return

    # Calculate metrics
    total_clusters = len(df_labels)
    labeled_clusters = len(df_labels[df_labels['status'] == 'labeled'])
    pending_clusters = len(df_labels[df_labels['status'] == 'pending'])
    
    progress_pct = (labeled_clusters / total_clusters) * 100 if total_clusters > 0 else 0
    
    total_annots = len(df_annots)
    removed_annots = len(df_annots[df_annots['keep'] == 0])

    # Print Summary Report
    print("\n" + "="*50)
    print("ZEBRA LABELING AUDIT")
    print("="*50)
    print(f" Total Clusters:   {total_clusters}")
    print(f" Labeled:          {labeled_clusters}")
    print(f" Pending:          {pending_clusters}")
    print(f" Progress:         {progress_pct:.1f}%")
    print("-" * 50)
    print(f" Total Bounding Boxes: {total_annots}")
    print(f" Boxes Removed:        {removed_annots}")
    print("="*50)
    
    # Show recently labeled
    print("\n Last 5 Labeled Clusters:")
    labeled_df = df_labels[df_labels['status'] == 'labeled'].tail(5)
    if not labeled_df.empty:
        print(labeled_df[['cluster_id', 'age', 'sex']].to_string(index=False))
    else:
        print("  None labeled yet.")
        
    # Show what's up next
    print("\n Next 5 Pending Clusters:")
    pending_df = df_labels[df_labels['status'] == 'pending'].head(5)
    if not pending_df.empty:
        print(pending_df[['cluster_id']].to_string(index=False))
    else:
        print("  No pending clusters left! 🎉")
    print("\n")

if __name__ == "__main__":
    check_progress(DB_PATH)