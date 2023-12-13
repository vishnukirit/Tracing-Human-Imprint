import streamlit as st
from PIL import Image
import numpy as np
from predict import function1

st.set_page_config(page_title='MY APP', layout='wide')
st.title('TRACING THE HUMAN IMPRINT')

original_image = st.file_uploader('Select an image', type=['jpg', 'jpeg'])

if original_image is not None:
    img = Image.open(original_image)

    container = st.container()
    left_col, right_col = container.columns(2)
    left_col.image(img, caption='Original image', use_column_width=True)

    with st.spinner('Processing image...'):
        processed_img = function1(np.array(img))

    right_col.image(processed_img, caption='InceptionV3', use_column_width=True)
else:
    st.write('Please upload an image.')
