# Utils for preprocessing data etc 
import os
import tensorflow as tf
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import cv2
import numpy as np

""" Global parameters """
H = 512
W = 512

base_classes = ['chicken_curry',
 'chicken_wings',
 'fried_rice',
 'grilled_salmon',
 'hamburger',
 'ice_cream',
 'pizza',
 'ramen',
 'steak',
 'sushi']

classes_and_models = {
    "model_1": {
        "classes": base_classes,
        "model_name": "efficientnet_model_1_10_classes" # change to be your model name
    },
    "model_2": {
        "classes": sorted(base_classes + ["donut"]),
        "model_name": "efficientnet_model_2_11_classes"
    },
    "model_3": {
        "classes": sorted(base_classes + ["donut", "not_food"]),
        "model_name": "efficientnet_model_3_12_classes"
    }
}

def predict_off(image):
    model_path = os.path.join("files", "modelrefpmaredone3.h5")
    model = tf.keras.models.load_model(model_path)
    return model.predict(image, verbose=0)

def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to Tensors.
        version (str): version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the 
            model.
    """
    # Create the ML Engine service object
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = os.path.join("files", "modelrefpmaredone3.h5")
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    instances_list = instances.numpy().tolist() # turn input into list (ML Engine wants JSON)
    
    input_data_json = {"signature_name": "serving_default",
                       "instances": instances_list} 

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    
    # # ALT: Create model api
    # model_api = api_endpoint + model_path + ":predict"
    # headers = {"Authorization": "Bearer " + token}
    # response = requests.post(model_api, json=input_data_json, headers=headers)

    if "error" in response:
        raise RuntimeError(response["error"])

    return response["predictions"]

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename): 
    file_bytes = np.asarray(bytearray(filename), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    x = cv2.resize(image, (W, H))
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    return x

def load_result(pred, filename):
    file_bytes = np.asarray(bytearray(filename), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_h, image_w,_ = image.shape
    y0 = pred[0][0]
    y0 = cv2.resize(y0, (image_w, image_h))
    y0 = np.expand_dims(y0, axis=-1)
    result = image*y0
    is_sucess, result = cv2.imencode(".jpg", result)
    print(is_sucess)
    return result.tobytes()

def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
    """
    Function for tracking feedback given in app, updates and reutrns 
    logger dictionary.
    """
    logger = {
        "image": image,
        "model_used": model_used,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "correct": correct,
        "user_label": user_label
    }   
    return logger
