# coding=utf-8

### Script for CS329s ML Deployment Lec 
import os
import json
import requests
import SessionState
import gettext
import streamlit.components.v1 as components
import streamlit as st
import tensorflow as tf
from utils import predict_off, load_result, load_and_prep_image, classes_and_models, update_logger, predict_json
from inter import inter

#online = True
online = False

#HtmlFile = open("test.html", 'r', encoding='utf-8')
#source_code = HtmlFile.read()

_ = inter("pt-br") #internationalization

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join("files", "custom-background-remover-a499821ff491.json") # change for your GCP key
PROJECT = "custom-background-remover" # change for your GCP project
REGION = "us-east1" # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
emojis = " 🙌😍📸" # win+"."" shortcut icon
st.title(_("Welcome to AI Sweet Background Remover!"))
message = _("Remove background from your photos")
st.header(message + emojis) 


@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def remove_background(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """

    pimage = load_and_prep_image(image)
    if(online):
        #pimage = tf.cast(tf.expand_dims(pimage, axis=0), tf.int16) #it expects a map
        preds = predict_json(project=PROJECT,
                            region=REGION,
                            model=model,
                            instance=pimage)
    else:
        preds = predict_off(pimage)
    image = load_result(preds, image)
    return image


CLASSES = classes_and_models["model_1"]["classes"]
MODEL = classes_and_models["model_1"]["model_name"]

# File uploader allows user to add their own image

uploaded_file = st.file_uploader(label=_("Upload an image to be processed:"),
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)

#ad?



# Create logic for app flow
if not uploaded_file:
    st.warning(_("Please upload an image."))
    #components.html(source_code, height=600)
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    #st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button(_("Send"))


# Did the user press the Send button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image = remove_background(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    st.image(session_state.image, use_column_width=True)
  



# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()

