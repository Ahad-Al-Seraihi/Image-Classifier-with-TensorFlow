# Importing Libraries
import argparse

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from PIL import Image

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Defining an arg parser containing 4 parse arguments 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Predict a flower class')
    
    parser.add_argument('-i','--input_image', default = './test_images/hard-leaved_pocket_orchid.jpg', action = 'store', help = 'image path', type = str)
    parser.add_argument('-m','--model', default = './1626224003.h5', action = 'store')
    parser.add_argument('-k','--top_k', default = 5, action = 'store', help = 'Top k flower classes', type = int)
    parser.add_argument('-n','--category_names', default = './label_map.json', action = 'store', type = str)
    
    args = parser.parse_args()
    image_path = args.input_image
    saved_keras_model = args.model
    top_k = args.top_k
    flower_classes = args.category_names
    print(args)

# Loading the saved model
def load_model(saved_keras_model):
    loaded_keras_model = tf.keras.models.load_model(saved_keras_model, custom_objects={'KerasLayer':hub.KerasLayer})
    return loaded_keras_model

# Loading a JSON file to display class names instead of integers
def load_class_names(flower_classes):
    with open(flower_classes, 'r') as f:
        class_names = json.load(f)
    
# Image processing and normalization
batch_size = 32
image_size = 224

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

# Defining the predict function 
def predict (image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    
    prediction = loaded_keras_model.predict(np.expand_dims(processed_test_image, axis=0))
    
    top_k_prediction_values, top_k_prediction_indices = tf.math.top_k(prediction, top_k)
    
    top_k_prediction_values = top_k_prediction_values.numpy()
    top_k_prediction_indices = top_k_prediction_indices.numpy()
    
    return top_k_prediction_values, top_k_prediction_indices
    
    # Prediction 
    top_k_prediction_values, top_k_prediction_indices = predict (image_path, loaded_keras_model, 5)
    
    print ('Top Probabilities:', top_k_prediction_values[0])
    
    top_flower_classes = [class_names[str(value+1)] for value in top_k_prediction_indices[0]]
    print ('Top Classes:', top_flower_classes)
