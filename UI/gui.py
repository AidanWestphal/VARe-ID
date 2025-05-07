import gradio as gr
from PIL import Image
import argparse
from db_scripts import update_status, get_next_pair
import os
os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/gradio_cache")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--db', required=True, help='Path to SQLite database')
args = parser.parse_args()

db_path = args.db
current_pair = {"id": None, "image1": None, "image2": None}

def load_next_pair():
    result = get_next_pair(db_path=db_path)
    print(result)
    if result:
        current_pair["id"], img1_path, img2_path = result
        current_pair["image1"], current_pair["image2"] = img1_path, img2_path
        return Image.open(img1_path), Image.open(img2_path)
    return None, None

def submit_decision(label):
    if current_pair["id"] is not None:
        update_status(current_pair["id"], label, db_path=db_path)
    return load_next_pair()

with gr.Blocks() as demo:
    gr.Markdown("## Image Verification Interface")
    with gr.Row():
        img1 = gr.Image()
        img2 = gr.Image()
    with gr.Row():
        btn_yes = gr.Button("Yes")
        btn_no = gr.Button("No")
        btn_cant_tell = gr.Button("Can't tell")

    btn_yes.click(lambda: submit_decision("correct"), outputs=[img1, img2])
    btn_no.click(lambda: submit_decision("incorrect"), outputs=[img1, img2])
    btn_cant_tell.click(lambda: submit_decision("cant_tell"), outputs=[img1, img2])

    demo.load(load_next_pair, outputs=[img1, img2])

demo.launch(server_port=7860)
