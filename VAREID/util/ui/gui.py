import gradio as gr
import argparse
from VAREID.util.ui.db_scripts import (
    update_status, get_next_pair_atomic, release_pair, 
    start_heartbeat_system, init_db, get_instance_stats,
    update_heartbeat, INSTANCE_IDENTIFIER
)
import os
import threading
import time
import atexit
import json
from PIL import Image

os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/gradio_cache")

# Global variables
current_pair = {"id": None, "image1": None, "image2": None, "bbox1": None, "bbox2": None}
history_stack = []
heartbeat_timer = None


def crop_image_with_bbox(image_path, bbox_json):
    """Crop image according to bounding box if provided"""
    if not image_path or image_path == "NO_IMAGE":
        return None
    
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        return None
    
    try:
        # Load the image
        img = Image.open(image_path)
        
        # If no bbox, return original image path
        if not bbox_json:
            return image_path
        
        # Parse bbox
        bbox = json.loads(bbox_json)
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img.width, int(x2))
        y2 = min(img.height, int(y2))
        
        # Crop the image
        cropped = img.crop((x1, y1, x2, y2))
        
        # Save to temp file
        temp_path = os.path.join(os.path.expanduser("~/gradio_cache"), f"cropped_{os.path.basename(image_path)}")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        cropped.save(temp_path)
        
        return temp_path
        
    except Exception as e:
        print(f"Error cropping image: {e}")
        # If cropping fails, return original image
        return image_path


def start_pair_heartbeat(pair_id):
    """Start sending heartbeats for a pair"""
    global heartbeat_timer
    
    def send_heartbeat():
        if current_pair.get("id") == pair_id:
            update_heartbeat(pair_id, db_path)
            # Schedule next heartbeat
            global heartbeat_timer
            heartbeat_timer = threading.Timer(30.0, send_heartbeat)
            heartbeat_timer.daemon = True
            heartbeat_timer.start()
    
    # Cancel any existing timer
    if heartbeat_timer:
        heartbeat_timer.cancel()
    
    # Start new heartbeat
    send_heartbeat()


def stop_pair_heartbeat():
    """Stop sending heartbeats"""
    global heartbeat_timer
    if heartbeat_timer:
        heartbeat_timer.cancel()
        heartbeat_timer = None


def fetch_pair():
    """Atomically fetch and reserve a pair from the database"""
    result = get_next_pair_atomic(db_path=db_path)
    if result:
        pair_id, img1_path, img2_path, bbox1, bbox2, cluster1, cluster2 = result
        return {
            "id": pair_id,
            "image1": img1_path,
            "image2": img2_path,
            "bbox1": bbox1,
            "bbox2": bbox2,
            "cluster1": cluster1,
            "cluster2": cluster2
        }
    return None


def clear_images():
    """Return empty images to clear the display"""
    return None, None, "Loading new images...", "", "", True


def load_next_pair():
    """Load the next image pair"""
    global current_pair
    
    # Stop heartbeat for current pair
    stop_pair_heartbeat()
    
    # Add current pair to history if valid (but don't release it yet)
    if current_pair["id"] is not None:
        history_stack.append(current_pair.copy())
    
    # Fetch a new pair
    pair_data = fetch_pair()
    if pair_data:
        current_pair = pair_data
        # Start heartbeat for new pair
        start_pair_heartbeat(current_pair["id"])
        
        # Crop images based on bounding boxes
        cropped_img1 = crop_image_with_bbox(current_pair["image1"], current_pair["bbox1"])
        cropped_img2 = crop_image_with_bbox(current_pair["image2"], current_pair["bbox2"])
        
        # Get instance stats for status message
        stats = get_instance_stats(db_path)
        status_msg = (f"Loaded pair {current_pair['id']} | "
                     f"Available: {stats['awaiting']} | "
                     f"Active instances: {stats['active_instances']} | "
                     f"Instance: {INSTANCE_IDENTIFIER}")
        return cropped_img1, cropped_img2, status_msg, f"**Cluster ID: {current_pair['cluster1']}**", f"**Cluster ID: {current_pair['cluster2']}**", False

    else:
        current_pair = {"id": None, "image1": None, "image2": None, "bbox1": None, "bbox2": None, "cluster1": None, "cluster2": None}
        stats = get_instance_stats(db_path)
        status_msg = (f"No pairs available | "
                     f"Awaiting: {stats['awaiting']} | "
                     f"In progress: {stats['in_progress']} | "
                     f"Checked: {stats['checked']}")
        
        return None, None, status_msg, None, None, True


def refresh_and_load():
    """Simple function: check for pairs and load if available"""
    global current_pair
    stats = get_instance_stats(db_path)

    if current_pair.get("id") is None and stats['awaiting'] > 0:
        return load_next_pair()

    if current_pair.get("id"):
        cropped_img1 = crop_image_with_bbox(current_pair["image1"], current_pair["bbox1"])
        cropped_img2 = crop_image_with_bbox(current_pair["image2"], current_pair["bbox2"])
        cluster1 = f"**Cluster ID: {current_pair['cluster1']}**"
        cluster2 = f"**Cluster ID: {current_pair['cluster2']}**"
        status_msg = (f"Working on pair {current_pair['id']} | "
                      f"Available: {stats['awaiting']} | "
                      f"Active instances: {stats['active_instances']} | "
                      f"Instance: {INSTANCE_IDENTIFIER}")
        return cropped_img1, cropped_img2, status_msg, cluster1, cluster2, False
    else:
        status_msg = (f"No active pair | "
                      f"Available: {stats['awaiting']} | "
                      f"In progress: {stats['in_progress']} | "
                      f"Checked: {stats['checked']} | "
                      f"Instance: {INSTANCE_IDENTIFIER}")
        return None, None, status_msg, "", "", stats['awaiting'] == 0


def timer_check():
    """Timer only calls refresh when specific conditions met"""
    # Only refresh if: no pair loaded AND pairs might be available
    if current_pair.get("id") is None:
        stats = get_instance_stats(db_path)
        if stats['awaiting'] > 0:
            return refresh_and_load()
    
    # Otherwise do nothing
    return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()


def submit_decision(label):
    """Submit user decision and trigger the two-step update"""
    global current_pair
    
    # Stop heartbeat
    stop_pair_heartbeat()
    
    # Store current pair ID before updating
    current_id = current_pair.get("id")
    
    if current_id is not None:
        # Submit decision in background
        def update_in_background():
            success = update_status(current_id, label, db_path=db_path)
            if not success:
                print(f"Warning: Failed to update pair {current_id} - may have been taken by another instance")
        
        threading.Thread(target=update_in_background).start()
    
    # First step: Clear images
    return clear_images()


def load_after_decision():
    """Second step: Load new images after clearing"""
    return load_next_pair()


def go_back_clear():
    """First step of going back: clear images and release current pair"""
    global current_pair
    
    # Stop heartbeat
    stop_pair_heartbeat()
    
    # Release current pair back to awaiting status
    if current_pair.get("id") is not None:
        def release_in_background():
            success = release_pair(current_pair["id"], db_path=db_path)
            if not success:
                print(f"Warning: Could not release pair {current_pair['id']} - may have been taken by another instance")
        threading.Thread(target=release_in_background).start()
    
    return clear_images()


def go_back_load():
    """Second step of going back: try to load previous pair"""
    global current_pair
    
    if history_stack:
        # Try to get the previous pair again
        previous = history_stack.pop()
        
        # Try to re-acquire the previous pair atomically
        # First check if it's still available
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM image_verification WHERE id=?", (previous["id"],))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] == 'awaiting':
            # Try to get it through normal atomic method
            # Set it back to awaiting first, then try to get it
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE image_verification 
                SET status='awaiting' 
                WHERE id=? AND status='awaiting'
            """, (previous["id"],))
            conn.commit()
            conn.close()
            
            # Now try to get it
            result = get_next_pair_atomic(db_path)
            if result and result[0] == previous["id"]:
                current_pair = previous
                start_pair_heartbeat(current_pair["id"])
                
                # Crop images based on bounding boxes
                cropped_img1 = crop_image_with_bbox(current_pair["image1"], current_pair["bbox1"])
                cropped_img2 = crop_image_with_bbox(current_pair["image2"], current_pair["bbox2"])
                
                status_msg = f"Returned to previous pair {current_pair['id']}"
                return cropped_img1, cropped_img2, status_msg, f"**Cluster ID: {current_pair['cluster1']}**", f"**Cluster ID: {current_pair['cluster2']}**", False
        
        # If we couldn't get the previous pair, get a new one
        status_msg = "Previous pair no longer available, loading new pair..."
        next_result = load_next_pair()
        return next_result[0], next_result[1], status_msg, next_result[3], next_result[4], next_result[5]
    else:
        # No history available
        stats = get_instance_stats(db_path)
        status_msg = f"No history to go back to | Available: {stats['awaiting']}"
        if current_pair["id"] is not None:
            start_pair_heartbeat(current_pair["id"])  # Restart heartbeat
            
            # Crop images based on bounding boxes
            cropped_img1 = crop_image_with_bbox(current_pair["image1"], current_pair["bbox1"])
            cropped_img2 = crop_image_with_bbox(current_pair["image2"], current_pair["bbox2"])
            
            return cropped_img1, cropped_img2, status_msg, f"**Cluster ID: {current_pair['cluster1']}**", f"**Cluster ID: {current_pair['cluster2']}**", False
        else:
            return None, None, status_msg, "", "", True


def cleanup_on_exit():
    """Clean up any active pairs when the app shuts down"""
    global current_pair
    stop_pair_heartbeat()
    if current_pair.get("id") is not None:
        release_pair(current_pair["id"], db_path)


# Register cleanup function
atexit.register(cleanup_on_exit)

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## ID Verification Interface")
    all_done_flag = gr.State(value=False)
    
    # Status message
    status = gr.Textbox(label="Status", value="Loading...", interactive=False)
    
    with gr.Row():
        with gr.Column():
            cluster1_label = gr.Markdown("**Cluster ID: ?**", elem_id="cluster1")
            img1 = gr.Image(label="Image 1", type="filepath", show_label=False)
        with gr.Column():
            cluster2_label = gr.Markdown("**Cluster ID: ?**", elem_id="cluster2")
            img2 = gr.Image(label="Image 2", type="filepath", show_label=False)

        
    with gr.Row():
        btn_back = gr.Button("⬅ Back")
        btn_yes = gr.Button("Same")
        btn_no = gr.Button("Different")
        btn_cant_tell = gr.Button("Can't tell")
        btn_refresh_status = gr.Button("🔄 Refresh Status")
    
    # Simple timer - only refreshes when no pair loaded and pairs available
    timer = gr.Timer(value=1, active=True)
    timer.tick(
        fn=timer_check,
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    )
    
    # Two-step update process for decisions
    btn_yes.click(
        lambda *args: submit_decision("correct"), 
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    ).then(
        lambda *args: load_after_decision(),
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    )
    
    btn_no.click(
        lambda *args: submit_decision("incorrect"), 
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    ).then(
        lambda *args: load_after_decision(),
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    )
    
    btn_cant_tell.click(
        lambda *args: submit_decision("cant_tell"), 
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    ).then(
        lambda *args: load_after_decision(),
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    )
    
    # Two-step update process for back button
    btn_back.click(
        lambda *args: go_back_clear(),
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    ).then(
        lambda *args: go_back_load(),
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    )
    
    # Refresh status button - calls same function as timer
    btn_refresh_status.click(
        lambda *args: refresh_and_load(),
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    )
    
    # Load initial pair on startup
    demo.load(
        lambda *args: load_next_pair(),
        outputs=[img1, img2, status, cluster1_label, cluster2_label, all_done_flag]
    )

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--allowed_dir', required=True, help='The directory that the GUI is allowed to access')
    args = parser.parse_args()
    print("GUI: Loading args...")

    db_path = args.db

    # Initialize database and start heartbeat system
    init_db(db_path)
    start_heartbeat_system(db_path)
    
    print(f"Starting instance {INSTANCE_IDENTIFIER}")
    
    # Launch the interface with allowed paths
    demo.launch(
        server_port=7861,
        server_name="0.0.0.0",  # Allow external connections
        share=False,
        allowed_paths=[args.allowed_dir]  # Add data directory to allowed paths
    )