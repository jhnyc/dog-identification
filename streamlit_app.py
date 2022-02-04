import streamlit as st
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
import numpy as np
import pickle

# Pretrained model InceptionResNetV2 for feature extraction
inception_bottleneck = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Model for prediction
model = tf.keras.models.load_model('model.h5')

# Dog adjectives
with open('random_dog_adj.text') as file:
    lines = file.readlines()
    adj = [line.rstrip() for line in lines]

# Breed labels
with open('breed_label.pickle', 'rb') as f:
    breed_label = pickle.load(f)
breed_label[27] = 'Cardigan Welsh Corgi'
breed_label[86] = 'Pembroke Welsh Corgi'

# Resize & reshape image array, extract features and return prediction
def predict(image):
    img_array = np.array(image)
    image_array = cv2.resize(img_array, (299,299))/255
    image_array = np.reshape(image_array,(1,299,299,3))
    features = inception_bottleneck.predict(image_array)
    X = features.reshape((1,-1))
    prediction = model.predict(X)
    return breed_label[prediction.argmax()]

# Crop the image shown on Streamlit to square
def crop_center(pil_img):
    img_w, img_h = pil_img.size
    hw = min(pil_img.size)
    return pil_img.crop(((img_w - hw) // 2,
                         (img_h - hw) // 2,
                         (img_w + hw) // 2,
                         (img_h + hw) // 2))


# Streamlit App
st.title("Doggo Recognition")

# Customize font size with markdown
st.markdown("""
<style>
.big-font {
    font-size:60px !important;
}
.small-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    cropped_img = crop_center(image)
    st.image(cropped_img, caption='Doggo Image.', width=500) # use_column_width=True
    st.write("")
    st.write("......")
    label = " ".join([i.capitalize() for i in predict(cropped_img).split('_')])
    st.markdown(f'<p class="small-font">This is a {label}!</p>', unsafe_allow_html=True) 
