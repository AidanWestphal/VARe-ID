import json
import pandas as pd

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['annotations'])

def verify_dataset_splits(train_path, val_path, test_path):
    print("Loading splits...")
    train_df = load_annotations(train_path)
    val_df = load_annotations(val_path)
    test_df = load_annotations(test_path)
    
    splits = {'Train': train_df, 'Valid': val_df, 'Test': test_df}
    
    # --- 1 & 2. Age and Sex Numbers Per Split ---
    for name, df in splits.items():
        print(f"\n{'='*15} {name.upper()} SPLIT {'='*15}")
        print(f"Total Annotations: {len(df)}")
        print(f"Total Unique IDs: {df['cluster_id'].nunique()}")
        
        print("\n-- AGE STATS --")
        age_stats = df.groupby('age').agg(
            Annotations=('uuid', 'count'),  # Assuming 'id' is the annotation ID
            Unique_IDs=('cluster_id', 'nunique')
        )
        print(age_stats)
        
        print("\n-- SEX STATS --")
        sex_stats = df.groupby('sex').agg(
            Annotations=('uuid', 'count'),
            Unique_IDs=('cluster_id', 'nunique')
        )
        print(sex_stats)

    # --- 3. Dataset Totals ---
    print(f"\n{'='*15} FULL DATASET TOTALS {'='*15}")
    full_df = pd.concat(splits.values(), ignore_index=True)
    print(f"Total Annotations: {len(full_df)}")
    print(f"Total Unique IDs: {full_df['cluster_id'].nunique()}")

    # --- 4. Leakage / Overlap Check ---
    print(f"\n{'='*15} LEAKAGE CHECK (CLUSTER IDs) {'='*15}")
    train_ids = set(train_df['cluster_id'].dropna().unique())
    val_ids = set(val_df['cluster_id'].dropna().unique())
    test_ids = set(test_df['cluster_id'].dropna().unique())
    
    train_val_leak = train_ids.intersection(val_ids)
    train_test_leak = train_ids.intersection(test_ids)
    val_test_leak = val_ids.intersection(test_ids)
    
    print(f"Train & Valid Overlap: {len(train_val_leak)} IDs")
    print(f"Train & Test Overlap:  {len(train_test_leak)} IDs")
    print(f"Valid & Test Overlap:  {len(val_test_leak)} IDs")
    
    if len(train_val_leak) > 0 or len(train_test_leak) > 0 or len(val_test_leak) > 0:
        print("\nWARNING: DATA LEAKAGE DETECTED BETWEEN SPLITS!")
    else:
        print("\nSUCCESS: Strict ID isolation confirmed. Zero leakage.")

if __name__ == "__main__":
    verify_dataset_splits(
        '/fs/ess/PAS2136/ggr_data/wbia/GGR2016/agesex_train_split.json', 
        '/fs/ess/PAS2136/ggr_data/wbia/GGR2016/agesex_val_split.json', 
        '/fs/ess/PAS2136/ggr_data/wbia/GGR2016/agesex_test_split.json'
    )