import os

import pandas as pd
import streamlit as st
from PIL import Image

IMAGE_DIR = "test_dataset/images"
DETECTIONS_PATH = "test_dataset/annots/pred_annots_yolov10l.csv"
OUT_PATH = "annotations_archive_.csv"

# Load the CSV file
bbox_df = pd.read_csv(DETECTIONS_PATH)

labels = ["Left", "Right", "Up", "Down", "Front", "Back"]
ia_options = ["Select...", "True", "False"]
ca_options = ["Select...", "True", "False"]
species_options = [
    "Select...",
    "zebra_plains",
    "zebra_grevys",
    "wild_dog",
    "Other",
    "Garbage",
]


def annotate_image(row, df, output_row, key):
    """Image annotation block"""
    image_path = os.path.join(IMAGE_DIR, row["image uuid"]) + ".jpg"
    img = Image.open(image_path)
    bbox = list(
        map(
            lambda x: int(float(x)),
            row["bbox"].replace("[", "").replace("]", "").split(","),
        )
    )

    # Image display
    st.image(img.crop(bbox), use_column_width=True)

    # Format image size, aspect ratio
    st.markdown(
        """
        <style>
            img {
                max-width: 400px;
                max-height: 500px;
                height: auto;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    selected_labels = []
    st.write("Viewpoint (Select all that apply):")
    cols = st.columns(len(labels))

    loaded_checks = (
        [i for i in labels if i in output_row["viewpoint"]]
        if output_row is not None
        else None
    )

    for i, label in enumerate(labels):
        is_checked = False
        if loaded_checks and label in loaded_checks:
            is_checked = True

        if cols[i].checkbox(label, value=is_checked, key=key + label):
            selected_labels.append(label)

    # Custom CSS to reduce space between the label and the radio options
    st.markdown(
        """
        <style>
        .compact-radio .stRadio { margin-top: -15px; margin-bottom: -20px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.write("IA Annotation:")
    with st.container():
        st.markdown('<div class="compact-radio">', unsafe_allow_html=True)
        ia_selection = st.radio(
            "Select IA Annotation",
            ia_options,
            horizontal=True,
            key=key + "ia_radio",
            index=(
                0
                if output_row is None
                else (1 if output_row["ia_annotation"] is True else 2)
            ),
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("CA Annotation:")
    with st.container():
        st.markdown('<div class="compact-radio">', unsafe_allow_html=True)
        ca_selection = st.radio(
            "Select CA Annotation",
            ca_options,
            horizontal=True,
            key=key + "ca_radio",
            index=(
                0
                if output_row is None
                else (1 if output_row["ca_annotation"] is True else 2)
            ),
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("Species ID:")
    with st.container():
        st.markdown('<div class="compact-radio">', unsafe_allow_html=True)
        species_selection = st.radio(
            "Select Species ID",
            species_options,
            horizontal=True,
            key=key + "species_radio",
            index=(
                species_options.index(output_row["species_id"])
                if output_row is not None
                else 0
            ),
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Check if all required fields are filled
    is_viewpoint_selected = len(selected_labels) > 0
    is_ia_selected = ia_selection != "Select..."
    is_ca_selected = ca_selection != "Select..."
    is_species_selected = species_selection != "Select..."

    # Disable the save button if any of the annotations are not selected
    save_button_disabled = not (
        is_viewpoint_selected
        and is_ia_selected
        and is_ca_selected
        and is_species_selected
    )

    if st.button("Save", key=key + "button", disabled=save_button_disabled):
        filename = row["filename"] + ".jpg"
        bbox_str = row["bbox"]
        mask = (df["filename"] == filename) & (df["bbox"] == bbox_str)
        selected_labels_str = ", ".join(selected_labels)
        if mask.any():
            df.loc[mask, "viewpoint"] = selected_labels_str
            df.loc[mask, "ia_annotation"] = ia_selection == "True"
            df.loc[mask, "species_id"] = species_selection
            df.loc[mask, "ca_annotation"] = ca_selection == "True"
        else:
            df.loc[len(df)] = [
                filename,
                bbox_str,
                selected_labels_str,
                ia_selection,
                species_selection,
                ca_selection,
            ]
        st.success("Annotations saved.")
    return df


def save_annotations(df):
    """Read annotations to file"""
    df.to_csv(OUT_PATH, index=False)
    st.success("Annotations saved to CSV file.")


def main():
    """Streamlit app"""
    page_size = 10

    st.title("Image Annotation Tool")
    page = st.number_input(
        "Page number", min_value=1, max_value=len(bbox_df) // page_size + 1, step=1
    )
    start = (page - 1) * page_size
    end = start + page_size

    if "df" not in st.session_state:
        if os.path.exists(OUT_PATH):
            # Load pre-existing annotation file
            st.session_state.df = pd.read_csv(OUT_PATH)
        else:
            # Create new annotation file
            st.session_state.df = pd.DataFrame(
                columns=[
                    "filename",
                    "bbox",
                    "viewpoint",
                    "ia_annotation",
                    "species_id",
                    "ca_annotation",
                ]
            )

    # Render N annotation blocks
    for i, row in bbox_df.iloc[start:end].iterrows():
        output_row = (
            None if len(st.session_state.df) <= i else st.session_state.df.iloc[i]
        )

        st.session_state.df = annotate_image(
            row,
            st.session_state.df,
            output_row=output_row,
            key=str(page) + "_" + str(i),
        )

    st.dataframe(st.session_state.df)

    if st.button("Save Annotations"):
        save_annotations(st.session_state.df)


if __name__ == "__main__":
    main()
