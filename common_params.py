import os 

data_dir = "C:/dev/image_classification/data/"

img_dir = os.path.join(data_dir , "Images\\")

annotation_dir = os.path.join(data_dir, "Annotation\\")

npy_dir  = os.path.join(data_dir, "npy\\")
model_dir = os.path.join(data_dir,"model\\")

MAX_NUMBER_CLASS = 30 

DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.1
IMG_SIZE = 224
input_shape = (IMG_SIZE, IMG_SIZE, 3) 
image_size = (IMG_SIZE, IMG_SIZE)

NB_AUGMENTATION = 10