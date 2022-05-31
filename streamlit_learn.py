import streamlit as st
import numpy as np
import urllib
from PIL import Image
from testtest import main, load_local_image

st.set_page_config(page_title='watercolor-style image',
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="auto", )

add_selectbox = st.sidebar.selectbox\
    ("you can upload your own image from here!",
    ("watch a demo", "transform my image"))

if add_selectbox == "watch a demo":
    f"# Hi! im waiting for you! have fun!"
    f"#### this is a demo of the watercolor-style image generation app"
    "---"
    r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns([1,1,1,1,1,1])
    with r1c1:
        #st.image(load_local_image(r'http://cz.coder17.com/suuhou/test_convnextv2_nopaper/images/0001_fake_B.png'))
        st.image("images/fake_0026.png")
        st.image("images/fake_0026.png")
        st.image("images/fake_0038.png")
    with r1c2:
        st.image("images/fake_0050.png")
        st.image("images/fake_0051.png")
        st.image("images/fake_0091.png")
    with r1c3:
        st.image("images/fake_0096.png")
        st.image("images/fake_0101.png")
        st.image("images/fake_0104.png")
    with r1c4:
        st.image("images/fake_0107.png")
        st.image("images/fake_0116.png")
        st.image("images/fake_0146.png")
    with r1c5:
        st.image("images/fake_0183.png")
        st.image("images/fake_0214.png")
        st.image("images/fake_0392.png")
    with r1c6:
        st.image("images/fake_0405.png")
        st.image("images/fake_0510.png")
        st.image("images/fake_0522.png")

if add_selectbox == "transform my image":
    r1col1, r1col2, r1col3 = st.columns([6, 1, 6])
    r2col1, r2col2, r2col3 = st.columns([6, 1, 6])
    with r1col1:
        uploadFile = st.file_uploader(label='', type=['jpg', 'png'])
        if uploadFile is not None:
            with r1col3:
                st.code("watercolor-style image is here!")
                st.code("Currently, this app only supports outputting an image with a size of 512")
            with r2col1:
                image = Image.open(uploadFile).convert('RGB')
                st.image(np.array(image).astype(np.uint8))
                st.write("Image Uploaded Successfully")
            with r2col3:
                out = main(image)
                st.image(np.array(out).astype(np.uint8))
                st.write("here you are!")
        else:
            st.write("Please upload a image in JPG/PNG Format.")