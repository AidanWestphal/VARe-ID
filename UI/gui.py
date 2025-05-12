import gradio as gr
import argparse
from db_scripts import update_status, get_next_pair
import os
import threading
import time

os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/gradio_cache")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--db', required=True, help='Path to SQLite database')
args = parser.parse_args()

db_path = args.db

# Global variables
current_pair = {"id": None, "image1": None, "image2": None}
history_stack = []

def fetch_pair():
    """Fetch a pair from the database"""
    result = get_next_pair(db_path=db_path)
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
    """Load the next image pair after clearing"""
    global current_pair
    
    # Add current pair to history if valid
    if current_pair["id"] is not None:
        history_stack.append(current_pair.copy())
    
    # Fetch a new pair
    pair_data = fetch_pair()
    if pair_data:
        current_pair = pair_data
        status_msg = f"Loaded pair {current_pair['id']}"
        return current_pair["image1"], current_pair["image2"], status_msg
    else:
        status_msg = "No pairs available"
        return None, None, status_msg

def submit_decision(label):
    """Submit user decision and trigger the two-step update"""
    global current_pair
    
    # Store current pair ID before updating
    current_id = current_pair.get("id")
    
    if current_id is not None:
        # Submit decision in background
        def update_in_background():
            update_status(current_id, label, db_path=db_path)
        
        threading.Thread(target=update_in_background).start()
    
    # First step: Clear images
    return clear_images()

def load_after_decision():
    """Second step: Load new images after clearing"""
    return load_next_pair()

def go_back_clear():
    """First step of going back: clear images"""
    return clear_images()

def go_back_load():
    """Second step of going back: load previous pair"""
    global current_pair
    
    if history_stack:
        # Pop previous from history
        previous = history_stack.pop()
        current_pair = previous
        
        status_msg = f"Returned to previous pair {current_pair['id']}"
        return current_pair["image1"], current_pair["image2"], status_msg
    else:
        status_msg = "No history to go back to"
        if current_pair["id"] is not None:
            return current_pair["image1"], current_pair["image2"], status_msg
        else:
            return None, None, status_msg

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
    
    # Hidden button for second step of update
    trigger_load = gr.Button("Load Images", visible=False)
    
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
    
    # Load initial pair on startup
    demo.load(
        load_next_pair,
        outputs=[img1, img2, status]
    )

if __name__ == "__main__":
    # Get the base data directory
    data_dir = "/fs/ess/PAS2136/ggr_data/image_data/GGR2020_subset"
    
    # Launch the interface with allowed paths
    demo.launch(
        server_port=7861,
        server_name="0.0.0.0",  # Allow external connections
        share=False,
        allowed_paths=[data_dir]  # Add data directory to allowed paths
    )