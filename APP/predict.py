import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import keras
import segmentation_models as sm
import os
from dotenv import load_dotenv

def onehot_to_rgb(onehot):
    color_dict = { 
        0: [0, 255, 255], 1: [255, 255, 0], 
        2: [255, 0, 255],  3: [0, 255, 0], 
        4: [0, 0, 255], 5: [255, 255, 255],  6: [0, 0, 0]}
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

def predict_image(model, img):
    w,h = img.shape[:2]
    w = w // 256 +1
    h = h // 256 +1
    padding_shape = (w*256,h*256 ,3)
    padded_img= np.zeros(padding_shape)
    mask_shape = (w*256,h*256,7)

    padded_img[0:img.shape[0], 0:img.shape[1], :] = img
    padded_img = padded_img/255
    mask_padded = np.zeros(mask_shape)

    for i in range(0,mask_shape[0], 256):
        for j in range(0, mask_shape[1], 256):
            patch = padded_img[i:i+256,j:j+256,:]
            predicted  = model.predict(np.expand_dims(patch,axis=0))
            mask_padded[i:i+256,j:j+256,:] =predicted 
    return mask_padded[0:img.shape[0], 0:img.shape[1], :]

def function1(img):
    model = keras.models.load_model(os.getenv("model_path"),
                                    custom_objects={'focal_loss_plus_jaccard_loss': sm.losses.categorical_focal_jaccard_loss,
                                                    'iou_score': sm.metrics.IOUScore, 
                                                    'threshold': 0.5, 'f1-score':sm.metrics.FScore})
    res = predict_image(model, img)
    return onehot_to_rgb(res)
