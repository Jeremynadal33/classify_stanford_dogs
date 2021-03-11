import sys
import os
import numpy as np
import tensorflow as tf


def main():
    print('r')

def predict_new(model, img_path, labels, input_shape = (442, 386)):
    if type(model)== str : model = tf.keras.models.load_model(model)
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.smart_resize(img,input_shape)
    img = tf.expand_dims(img,0)
    pred = model.predict(img)[0]
    print('Predicted', labels[np.argmax(pred)].split('-')[-1], ': ', pred[np.argmax(pred)])

    #img = CLAHE_rgb(img)

if __name__ == "__main__":
    assert len(sys.argv) > 2, 'Please enter model path as arg 1 and path to img as arg 2 and label_path as arg 3 (if not in models/labels.txt)'

    img_path = sys.argv[1]
    model_path = sys.argv[2]

    label_path = 'models/labels.txt'
    if os.path.exists(label_path) :
        labels = open(label_path, 'r').read().split('\n')[:-1]
        assert len(labels)==120, 'Careful, there should be 120 labels'
    elif len(sys.argv)>3:
        assert os.path.exists(sys.argv[3]), 'path given as arg 3 is doesnot exist'
        labels = open(sys.argv[3], 'r').read().split('\n')[:-1]
    else :
        print('Please, either put a file labels.txt with labels in the models dir or enter the path as third arg')

    assert os.path.exists(img_path), 'img_path doesnot exists'
    assert os.path.exists(model_path), 'model_path doesnot exists'

    predict_new(model_path, img_path, labels)
