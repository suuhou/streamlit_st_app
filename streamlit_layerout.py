import io
import random
import streamlit as st
import numpy as np
import urllib
import requests
from PIL import Image
from mainstream import main, load_local_image

st.set_page_config(page_title='watercolor-style image',
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="auto", )

add_selectbox = st.sidebar.selectbox\
    ("you can upload your own image from here!",
    ("watch a demo", "transform my image"))



if add_selectbox == "watch a demo":
    f"# Hi! im waiting for you! have fun!"
    "---"

    r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns([1,1,1,1,1,1])
    image_list = load_local_image()

    with r1c1:
        st.image(image_list[0])
        st.image(image_list[1])
        st.image(image_list[2])
    with r1c2:
        st.image(image_list[3])
        st.image(image_list[4])
        st.image(image_list[5])
    with r1c3:
        st.image(image_list[6])
        st.image(image_list[7])
        st.image(image_list[8])
    with r1c4:
        st.image(image_list[9])
        st.image(image_list[10])
        st.image(image_list[11])
    with r1c5:
        st.image(image_list[12])
        st.image(image_list[13])
        st.image(image_list[14])
    with r1c6:
        st.image(image_list[15])
        st.image(image_list[16])
        st.image(image_list[17])

if add_selectbox == "transform my image":
    r1col1, r1col2, r1col3 = st.columns([6, 1, 6])
    r2col1, r2col2, r2col3 = st.columns([6, 1, 6])
    with r1col1:
        uploadFile = st.file_uploader(label='', type=['jpg', 'png'])
        if uploadFile is not None:
            with r1col3:
                st.code("watercolor-style image is here!")
                st.code("feel free to download!")
            with r2col1:
                image = Image.open(uploadFile).convert('RGB')
                st.image(np.array(image).astype(np.uint8))
                st.write("Image Uploaded Successfully")
            with r2col3:
                out = main(image)
                st.image(out)
                st.write("here you are!")
        else:
            st.write("Please upload a image in JPG/PNG Format.")

if __name__ == '__main__':
    #load_local_image(r'http://cz.coder17.com/suuhou/test_convnextv2_nopaper/images/0001_fake_B.png')
    print('ok')