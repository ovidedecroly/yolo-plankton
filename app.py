# Import required libraries
import PIL

import streamlit as st
from ultralytics import YOLO
import pandas as pd
import os
from io import StringIO

# Replace the relative path to your weight file
model_path = 'weights/best.pt'

# Initialize or load the log
log_file = 'detection_log.csv'
if os.path.exists(log_file):
    detection_log = pd.read_csv(log_file)
else:
    detection_log = pd.DataFrame(columns=['Image Name', 'Confidence'])

# Setting page layout
st.set_page_config(
    page_title="Object Detection",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
    
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Object Detection")
st.caption('Updload a photo with this :blue[hand signals]: :+1:, :hand:, :i_love_you_hand_sign:, and :spock-hand:.')
st.caption('Then click the :blue[Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Detecting objects when button is clicked
if st.sidebar.button('Detect Objects') and source_img:
    res = model.predict(uploaded_image, conf=confidence)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    
    with col2:
        st.image(res_plotted, caption='Detected Image', use_column_width=True)
        
        with st.expander("Detection Results"):
            for box in boxes:
                st.write(box.xywh)
    
    # Log the detection details
    image_name = source_img.name
    new_log = pd.DataFrame([[image_name, confidence]], columns=['Image Name', 'Confidence'])
    detection_log = pd.concat([detection_log, new_log], ignore_index=True)
    detection_log.to_csv(log_file, index=False)
    
    st.sidebar.write("Detection log updated.")
else:
    st.write("No image uploaded yet or Detect Objects button not clicked.")

# Display the log
if st.sidebar.checkbox('Show Detection Log'):
    st.sidebar.write(detection_log)
    
    # Create a CSV buffer
    csv = StringIO()
    detection_log.to_csv(csv, index=False)
    csv.seek(0)
    
    # Download button
    st.sidebar.download_button(
        label="Download Detection Log",
        data=csv.getvalue(),
        file_name='detection_log.csv',
        mime='text/csv',
    )