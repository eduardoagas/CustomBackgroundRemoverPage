# coding=utf-8

### Script for CS329s ML Deployment Lec 
import os
import json
import requests
import SessionState
import gettext
import streamlit as st
import tensorflow as tf
from utils import load_result, load_and_prep_image, classes_and_models, update_logger, predict_json
from inter import inter

_ = inter("pt-br") #internationalization

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "daniels-dl-playground-4edbcb2e6e37.json" # change for your GCP key
PROJECT = "daniels-dl-playground" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)

### Streamlit code (works as a straigtht-forward script) ###
emojis = " 🙌😍📸" # win+"."" shortcut icon
st.title(_("Welcome to Sweet Background Remover!"))
message = _("Remove background from your photos")
st.header(message + emojis) 


@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """

    pimage, ishape = load_and_prep_image(image)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=pimage)
    image = load_result(preds, ishape)
    return image

choose_model = "Model 1 (10 food classes)"

CLASSES = classes_and_models["model_1"]["classes"]
MODEL = classes_and_models["model_1"]["model_name"]

# File uploader allows user to add their own image

uploaded_file = st.file_uploader(label=_("Upload an image to be processed:"),
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    #session_state.uploaded_image = uploaded_file.read()
    #st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Send")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf:.3f}")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=session_state.image,
                            model_used=MODEL,
                            pred_class=session_state.pred_class,
                            pred_conf=session_state.pred_conf,
                            correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct label be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=session_state.image,
                                model_used=MODEL,
                                pred_class=session_state.pred_class,
                                pred_conf=session_state.pred_conf,
                                correct=False,
                                user_label=session_state.correct_class))

# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()