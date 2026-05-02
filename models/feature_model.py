import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import vgg16

def build_feature_model():
    input_shape = (150, 150, 3)

    vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    output = vgg.layers[-1].output
    output = tf.keras.layers.Flatten()(output)
    basemodel = Model(vgg.input, output)

    for layer in basemodel.layers:
        layer.trainable = False

    model = Sequential()
    model.add(basemodel)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    return model