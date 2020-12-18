import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import base64

sys.path.append(os.path.abspath('./model'))


# initialize model
def init_model():
    loaded_model = tf.keras.models.load_model('model/classifier-model.h5')
    print("loaded model successfully")
    return loaded_model


# decode from base64 and save image
def process_image(img):
    with open('img_output.png', 'wb') as output:
        output.write(base64.b64decode(str(img, 'utf-8')))


# make a prediction based on image
def predict(model):
    # read in image
    img = cv2.imread('img_output.png')
    # resize image to 28x28
    img = cv2.resize(img, (28, 28))
    # change to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # invert image, mnist encodes white w/ 0, and black w/ 255 (while normally its opposite)
    img = cv2.bitwise_not(img)
    # mean normalization
    img = (img / 255) - 0.5
    # flatten the image
    img = img.reshape((-1, 784))
    # predict on image
    prediction = model.predict(img)
    # predicted label
    output = np.argmax(prediction[0])
    return output
