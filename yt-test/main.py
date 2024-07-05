import streamlit as st
from PIL import Image
import cv2
import easyocr
import numpy as np
import tempfile
import pdf2image

reader = easyocr.Reader(['en','ar'], gpu=False)
st.title("Ai task internship")

uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:


    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        # Convert PDF to images
        with tempfile.TemporaryDirectory() as path:
            images_from_path = pdf2image.convert_from_bytes(uploaded_file.read(), output_folder=path)
        image = images_from_path[0] 
    else:
        # Open the image
        image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded File', use_column_width=True)
    
        # using pillow to resize the image 
    img_resized = image.resize((800, 600), Image.Resampling.LANCZOS)
    
        # now convert to cv format
    img_cv2 = np.array(img_resized)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

      
    text_ = reader.readtext(img_cv2)

        # hay l2n sometimes the detector detects "?" and weird stuff
    high_conf_texts = [text for _, text, score in text_ if score >= 0.10 and len(text) > 1 and not text.startswith("?")]
    for text in high_conf_texts:
        st.write(text)

        # copy button
    if high_conf_texts:
        high_conf_text_content = '\n'.join(high_conf_texts)
        st.download_button('Copy Detected Text', high_conf_text_content)

        # displays the detected texts on the image
    for t in text_:
        bbox, text, score = t
        if score >= 0.10 and len(text) > 1 and not text.startswith("?"):
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple([int(i) for i in top_left])
            bottom_right = tuple([int(i) for i in bottom_right])
            
        # this here for the rectangle that shows up bl images
            cv2.rectangle(img_cv2, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img_cv2, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, use_column_width=True)
