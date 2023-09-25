import os 
import sys
import mlflow
import cv2
import tensorflow as tf
from tensorflow import keras
from common.common_img_functions import resize_image
from common.label_names import label30 
import numpy as np 

SAVED_MODEL_ID = "model_registry"
IMG_SIZE = 224
class_names = [ name.split('-')[1] for name in label30 ]


if __name__ == "__main__" : 
    print("[NN Classification] Check given parameters ............. ")
    
    if len(sys.argv) != 2 :
        print("[NN Classification][Error User Param] Please provide image path as first argument")
        sys.exit()
    
    img_path = sys.argv[1]
    if not  os.path.exists(img_path):
        print(f"[NN Classification][Error User Param] Given image path does not exist, given :  {img_path}")
        sys.exit()
        
    ## Load image 
    print(f"[NN Classification] Load {img_path} image")
    img = cv2.imread(img_path) 

    ## Resizing image and add unique batch dimension 
    resized_img = resize_image(img, (224, 224), from_path=False)
    print(f"[NN Classification] Resize image input from {img.shape} to {resized_img.shape}")
    batched_img = tf.expand_dims(resized_img, axis=0)

    ## Model inference
    model_path = os.path.join(os.getcwd(), SAVED_MODEL_ID)
    model =  mlflow.keras.load_model(model_path)
    
    # Prediction output 
    predictions = model.predict(batched_img)
    prediction = predictions[0]
    
    print(f"[NN Classification][Result] The networks predict that this image is a {class_names[np.argmax(prediction)]} with a probability of {round(100*prediction[np.argmax(prediction)],2)}%")
