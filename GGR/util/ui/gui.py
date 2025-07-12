import gradio as gr
import argparse
from db_scripts import (
    update_status, get_next_pair_atomic, release_pair, 
    start_heartbeat_system, init_db, get_instance_stats,
    update_heartbeat, INSTANCE_IDENTIFIER
)
import os
import threading
import time
import atexit

os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/gradio_cache")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--db', required=True, help='Path to SQLite database')
args = parser.parse_args()

db_path = args.db

# Initialize database and start heartbeat system
init_db(db_path)
start_heartbeat_system(db_path)

# Global variables
current_pair = {"id": None, "image1": None, "image2": None}
history_stack = []
heartbeat_timer = None

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
        pair_id, img1_path, img2_path = result
        return {
            "id": pair_id,
            "image1": img1_path,
            "image2": img2_path
        }
    return None

def clear_images():
    """Return empty images to clear the display"""
    return None, None, "Loading new images..."

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
        
        # Get instance stats for status message
        stats = get_instance_stats(db_path)
        status_msg = (f"Loaded pair {current_pair['id']} | "
                     f"Available: {stats['awaiting']} | "
                     f"Active instances: {stats['active_instances']} | "
                     f"Instance: {INSTANCE_IDENTIFIER}")
        return current_pair["image1"], current_pair["image2"], status_msg
    else:
        current_pair = {"id": None, "image1": None, "image2": None}
        stats = get_instance_stats(db_path)
        status_msg = (f"No pairs available | "
                     f"Awaiting: {stats['awaiting']} | "
                     f"In progress: {stats['in_progress']} | "
                     f"Checked: {stats['checked']}")
        return None, None, status_msg

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
                status_msg = f"Returned to previous pair {current_pair['id']}"
                return current_pair["image1"], current_pair["image2"], status_msg
        
        # If we couldn't get the previous pair, get a new one
        status_msg = "Previous pair no longer available, loading new pair..."
        next_result = load_next_pair()
        return next_result[0], next_result[1], status_msg
    else:
        # No history available
        stats = get_instance_stats(db_path)
        status_msg = f"No history to go back to | Available: {stats['awaiting']}"
        if current_pair["id"] is not None:
            start_pair_heartbeat(current_pair["id"])  # Restart heartbeat
            return current_pair["image1"], current_pair["image2"], status_msg
        else:
            return None, None, status_msg

def get_status_update():
    """Get current status for display"""
    stats = get_instance_stats(db_path)
    if current_pair.get("id"):
        return (f"Working on pair {current_pair['id']} | "
                f"Available: {stats['awaiting']} | "
                f"Active instances: {stats['active_instances']} | "
                f"Instance: {INSTANCE_IDENTIFIER}")
    else:
        return (f"No active pair | "
                f"Available: {stats['awaiting']} | "
                f"In progress: {stats['in_progress']} | "
                f"Instance: {INSTANCE_IDENTIFIER}")

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
    
    # Status message
    status = gr.Textbox(label="Status", value="Loading...", interactive=False)
    
    with gr.Row():
        img1 = gr.Image(label="Image 1", type="filepath", show_label=False)
        img2 = gr.Image(label="Image 2", type="filepath", show_label=False)
    
    with gr.Row():
        btn_back = gr.Button("⬅ Back")
        btn_yes = gr.Button("Same")
        btn_no = gr.Button("Different")
        btn_cant_tell = gr.Button("Can't tell")
        btn_refresh_status = gr.Button("🔄 Refresh Status")
    
    # Keyboard shortcuts
    gr.HTML("""
    <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'b' || e.key === 'B') {
                document.querySelector('button:nth-of-type(1)').click(); // Back
            } else if (e.key === 's' || e.key === 'S' || e.key === 'y' || e.key === 'Y') {
                document.querySelector('button:nth-of-type(2)').click(); // Same/Yes
            } else if (e.key === 'd' || e.key === 'D' || e.key === 'n' || e.key === 'N') {
                document.querySelector('button:nth-of-type(3)').click(); // Different/No
            } else if (e.key === 'c' || e.key === 'C') {
                document.querySelector('button:nth-of-type(4)').click(); // Can't tell
            } else if (e.key === 'r' || e.key === 'R') {
                document.querySelector('button:nth-of-type(5)').click(); // Refresh
            }
        });
    </script>
    """)
    
    # Two-step update process for decisions
    btn_yes.click(
        lambda: submit_decision("correct"), 
        outputs=[img1, img2, status]
    ).then(
        load_after_decision,
        outputs=[img1, img2, status]
    )
    
    btn_no.click(
        lambda: submit_decision("incorrect"), 
        outputs=[img1, img2, status]
    ).then(
        load_after_decision,
        outputs=[img1, img2, status]
    )
    
    btn_cant_tell.click(
        lambda: submit_decision("cant_tell"), 
        outputs=[img1, img2, status]
    ).then(
        load_after_decision,
        outputs=[img1, img2, status]
    )
    
    # Two-step update process for back button
    btn_back.click(
        go_back_clear,
        outputs=[img1, img2, status]
    ).then(
        go_back_load,
        outputs=[img1, img2, status]
    )
    
    # Refresh status button
    btn_refresh_status.click(
        get_status_update,
        outputs=[status]
    )
    
    # Load initial pair on startup
    demo.load(
        load_next_pair,
        outputs=[img1, img2, status]
    )

if __name__ == "__main__":
    # Get the base data directory
    data_dir = "/fs/ess/PAS2136/ggr_data/image_data/GGR2020_subset"
    
    print(f"Starting instance {INSTANCE_IDENTIFIER}")
    
    # Launch the interface with allowed paths
    demo.launch(
        server_port=7861,
        server_name="0.0.0.0",  # Allow external connections
        share=False,
        allowed_paths=[data_dir]  # Add data directory to allowed paths
    )