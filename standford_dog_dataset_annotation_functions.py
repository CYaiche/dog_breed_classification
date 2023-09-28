import random as rand
import pandas as pd 
import numpy as np 
import cv2
import os 

from common_params import img_dir
from common_xml import xml_to_dic
from common_img_functions import square_img, resize_image, crop_image

    
def get_random_image_path_from_breed(data, breed): 
    nb_img_for_breed = int(data[data["dog_breed"] == breed]["image_count"].values[0])
    n = rand.randint(0, nb_img_for_breed)
    folder = data[data["dog_breed"] == breed]["folder_name"].values[0]
    breedDir  = os.path.join(img_dir + folder)
    return os.path.join(breedDir , os.listdir(breedDir)[n]) 

def get_imgs_info(data):
    breed_nb = data.shape[0]
    dog_img_size = pd.DataFrame()
    for i in range(breed_nb) : 
        
        dog_folder = os.path.join(img_dir, data.loc[i]["folder_name"])
        dog_breed = data.loc[i]["dog_breed"]
        for img in os.scandir(dog_folder) : 
            img_path = os.path.join(dog_folder, img.name)
            img_read = cv2.imread(img_path)
            height, width, channels = img_read.shape
            dog_img_size = dog_img_size.append({'breed' : dog_breed, 'height' : height, 'width': width,'channels' : channels}, ignore_index=True)
    return  dog_img_size
# Annotation 

def get_size_from_bndbox(bndbox_coordinate):
    return (
        int(bndbox_coordinate["xmax"]) - int(bndbox_coordinate["xmin"])
    ) * int(bndbox_coordinate["ymax"]) - int(bndbox_coordinate["ymin"])
    
    
def get_dog_picture_box(img_dic):
    cor = np.array([])
    objs = img_dic['annotation']['object']
    if isinstance(objs, list) : 
        for dog_info in objs: 
            cor =  np.append(cor, dog_info['bndbox'])
    else : 
        cor =  [ objs['bndbox'] ]
    return cor

def extract_and_resize_pipeline(img_path, annotation_path, target_size, margin=None):
    
    dic    = xml_to_dic(annotation_path)

    cor    = get_dog_picture_box(dic)

    img_source     = cv2.imread(img_path) 
    images = []
    
    for c in cor :

        img = crop_image(img_source, c, from_path=False, margin=margin)

        img = square_img(img, from_path=False)

        img = resize_image(img, target_size, from_path=False)

        images.append(img)

    return images, len(images)