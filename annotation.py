import streamlit as st
import os
import pandas as pd
from PIL import Image

# Directory where your images are stored
image_dir = '/Users/upadha2/Desktop/wild_dogs_3'

# Load the CSV file
bbox_df = pd.read_csv('/Users/upadha2/Desktop/annotation_tool/annot_output_file.csv')

labels = ['Left', 'Right', 'Up', 'Down', 'Front', 'Back']
ia_options = ['Select...', 'True', 'False']
species_options = ['Select...', 'zebra_plains', 'zebra_grevys', 'wild_dog', 'Other', 'Garbage']

def annotate_image(row, df, key):
    image_path = os.path.join(image_dir, row['filename'])
    img = Image.open(image_path)
    bbox = list(map(lambda x: int(float(x)), row['bbox'].replace('[', '').replace(']', '').split(',')))
    cropped_img = img.crop(bbox)
    resized_img = cropped_img.resize((224, 224))
    st.image(resized_img, use_column_width=True)

    selected_labels = []
    st.write("Viewpoint (Select all that apply):")
    cols = st.columns(len(labels))
    for i, label in enumerate(labels):
        if cols[i].checkbox(label, key=key+label):
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
        ia_selection = st.radio("Select IA Annotation", ia_options, horizontal=True, key=key+'ia_radio', label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("Species ID:")
    with st.container():
        st.markdown('<div class="compact-radio">', unsafe_allow_html=True)
        species_selection = st.radio("Select Species ID", species_options, horizontal=True, key=key+'species_radio', label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    # Check if all required fields are filled
    is_viewpoint_selected = len(selected_labels) > 0
    is_ia_selected = ia_selection != "Select..."
    is_species_selected = species_selection != "Select..."

    # Disable the save button if any of the annotations are not selected
    save_button_disabled = not (is_viewpoint_selected and is_ia_selected and is_species_selected)

    if st.button('Save', key=key+'button', disabled=save_button_disabled):
        filename = row['filename']
        bbox_str = row['bbox']
        mask = (df['filename'] == filename) & (df['bbox'] == bbox_str)
        selected_labels_str = ', '.join(selected_labels)
        if mask.any():
            df.loc[mask, 'viewpoint'] = selected_labels_str
            df.loc[mask, 'ia_annotation'] = ia_selection
            df.loc[mask, 'species_id'] = species_selection
        else:
            df.loc[len(df)] = [filename, bbox_str, selected_labels_str, ia_selection, species_selection]
        st.success('Annotations saved.')
    return df

def save_annotations(df):
    df.to_csv('annotations.csv', index=False)
    st.success('Annotations saved to CSV file.')

def main():
    st.title('Image Annotation Tool')
    page = st.number_input('Page number', min_value=1, max_value=len(bbox_df)//10+1, step=1)
    start = (page-1)*10
    end = start+10

    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=['filename', 'bbox', 'viewpoint', 'ia_annotation', 'species_id'])

    for i, row in bbox_df.iloc[start:end].iterrows():
        st.session_state.df = annotate_image(row, st.session_state.df, key=str(page)+'_'+str(i))

    st.dataframe(st.session_state.df)

    if st.button('Save Annotations'):
        save_annotations(st.session_state.df)

if __name__ == '__main__':
    main()


