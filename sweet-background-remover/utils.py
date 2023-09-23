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
        "model_name": "modelrefpmaredone3" # change to be your model name
    },
}

def predict_off(pimage):
    model_path = os.path.join("files", "modelrefpmaredone3.h5")
    model = tf.keras.models.load_model(model_path)
    return model.predict(pimage, verbose=0)

def predict_json(project, region, model, instance, version=None):
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
    model_path = "projects/{}/models/{}".format(project, model)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    ml_resource = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    instance_list = instance.tolist() # turn input into list (ML Engine wants JSON)
    
    input_data_json = {"signature_name": "serving_default",
                       "instances": instance_list} 

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
    y0 = pred[0][0]* 255
    y0 = cv2.resize(y0, (image_w, image_h))
    y0 = np.expand_dims(y0, axis=-1)
    #result = image*y0

    #todo esse rolê pra transformar em grayscale img
    #_,mask = cv2.imencode(".png", y0)
    #file_bytes = np.asarray(bytearray(mask.tobytes()), dtype=np.uint8)
    #mask = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    #cv2.imwrite("img.png", image)
    #cv2.imwrite("mask.png", mask)
    result = np.dstack((image, y0))
    #cv2.imwrite("result.png", result)
    is_sucess, result = cv2.imencode(".png", result)
    #result = remove_black_background(result.tobytes())
    #print(is_sucess)
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


"""""
def remove_black_background(src):
    file_bytes = np.asarray(bytearray(src), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    tmp = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    cv2.imwrite("test.png", dst)
    _,result = cv2.imencode(".png", dst)
    return result

"""""