import gradio as gr
from PIL import Image
import argparse
from db_scripts import update_status, get_next_pair
import os
import threading

os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/gradio_cache")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--db', required=True, help='Path to SQLite database')
args = parser.parse_args()

db_path = args.db

# Global variables to store current and next pairs
current_pair = {"id": None, "image1": None, "image2": None}
next_pair = {"id": None, "image1": None, "image2": None, "img1_obj": None, "img2_obj": None}
preload_lock = threading.Lock()
history_stack = []

def fetch_pair():
    """Fetch a pair from the database"""
    result = get_next_pair(db_path=db_path)
    print(f"Fetched pair: {result}")
    if result:
        pair_id, img1_path, img2_path = result
        try:
            return {
                "id": pair_id,
                "image1": img1_path,
                "image2": img2_path,
                "img1_obj": Image.open(img1_path),
                "img2_obj": Image.open(img2_path)
            }
        except Exception as e:
            print(f"Error loading images: {e}")
            return None
    return None

def preload_next_pair():
    """Function to preload the next pair"""
    global next_pair
    with preload_lock:
        pair_data = fetch_pair()
        if pair_data:
            next_pair = pair_data
            print(f"Preloaded next pair: {next_pair['id']}")
        else:
            print("No next pair to preload")

def start_preload_thread():
    """Start a thread to preload the next image pair"""
    thread = threading.Thread(target=preload_next_pair)
    thread.daemon = True
    thread.start()

def load_next_pair():
    global current_pair, next_pair, history_stack

    with preload_lock:
        if next_pair["id"] is not None:
            if current_pair["id"] is not None:
                history_stack.append({
                    "id": current_pair["id"],
                    "image1": current_pair["image1"],
                    "image2": current_pair["image2"],
                    "img1_obj": Image.open(current_pair["image1"]),
                    "img2_obj": Image.open(current_pair["image2"])
                })
            current_pair = {
                "id": next_pair["id"],
                "image1": next_pair["image1"],
                "image2": next_pair["image2"]
            }
            img1_obj = next_pair["img1_obj"]
            img2_obj = next_pair["img2_obj"]
            next_pair = {"id": None, "image1": None, "image2": None, "img1_obj": None, "img2_obj": None}
            print(f"Using preloaded pair: {current_pair['id']}")
        else:
            pair_data = fetch_pair()
            if pair_data:
                if current_pair["id"] is not None:
                    history_stack.append({
                        "id": current_pair["id"],
                        "image1": current_pair["image1"],
                        "image2": current_pair["image2"],
                        "img1_obj": Image.open(current_pair["image1"]),
                        "img2_obj": Image.open(current_pair["image2"])
                    })

                current_pair = {
                    "id": pair_data["id"],
                    "image1": pair_data["image1"],
                    "image2": pair_data["image2"]
                }
                img1_obj = pair_data["img1_obj"]
                img2_obj = pair_data["img2_obj"]
                print(f"Fetched new pair: {current_pair['id']}")
            else:
                print("No pair available")
                return None, None

    start_preload_thread()
    return img1_obj, img2_obj


def submit_decision(label):
    """Submit user decision and load next pair"""
    if current_pair["id"] is not None:
        print(f"Submitting decision '{label}' for pair {current_pair['id']}")
        update_status(current_pair["id"], label, db_path=db_path)
    return load_next_pair()

def go_back():
    """Go back to the previous pair if available"""
    global current_pair, next_pair
    if history_stack:
        # Before going back, push current pair back into next_pair
        try:
            next_pair = {
                "id": current_pair["id"],
                "image1": current_pair["image1"],
                "image2": current_pair["image2"],
                "img1_obj": Image.open(current_pair["image1"]),
                "img2_obj": Image.open(current_pair["image2"])
            }
        except Exception as e:
            print(f"Error caching current pair for reuse: {e}")
            next_pair = {"id": None, "image1": None, "image2": None, "img1_obj": None, "img2_obj": None}

        # Pop previous from history and show it
        previous = history_stack.pop()
        current_pair = {
            "id": previous["id"],
            "image1": previous["image1"],
            "image2": previous["image2"]
        }
        print(f"Going back to pair: {current_pair['id']}")
        return previous["img1_obj"], previous["img2_obj"]
    else:
        print("No history to go back to")
        return None, None

with gr.Blocks() as demo:
    gr.Markdown("## ID Verification Interface")
    with gr.Row():
        img1 = gr.Image(type="pil")
        img2 = gr.Image(type="pil")
    with gr.Row():
        btn_back = gr.Button("⬅ Back")
        btn_yes = gr.Button("Same")
        btn_no = gr.Button("Different")
        btn_cant_tell = gr.Button("Can't tell")
        

    btn_yes.click(lambda: submit_decision("correct"), outputs=[img1, img2])
    btn_no.click(lambda: submit_decision("incorrect"), outputs=[img1, img2])
    btn_cant_tell.click(lambda: submit_decision("cant_tell"), outputs=[img1, img2])
    btn_back.click(go_back, outputs=[img1, img2])


    # Load initial pair when app starts
    demo.load(load_next_pair, outputs=[img1, img2])

if __name__ == "__main__":
    # Start preloading first pair before launching the interface
    start_preload_thread()
    demo.launch(server_port=7861)