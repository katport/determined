import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.datasets
import tensorflow.keras.utils
import tensorflow.keras.backend
import numpy

from typing import List

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

from determined import keras

from determined.experimental import Determined

class LambdaTrial(keras.TFKerasTrial):
    def __init__(self, context: keras.TFKerasTrialContext) -> None:
        self.context = context

    def build_model(self) -> Model:
        train = True
        if train:
            input_layer = tensorflow.keras.layers.Input(shape=(784), name="input_layer")

            dense_layer_1 = tensorflow.keras.layers.Dense(units=500, name="dense_layer_1")(input_layer)
            activ_layer_1 = tensorflow.keras.layers.ReLU(name="relu_layer_1")(dense_layer_1)

            dense_layer_2 = tensorflow.keras.layers.Dense(units=250, name="dense_layer_2")(activ_layer_1)
            activ_layer_2 = tensorflow.keras.layers.ReLU(name="relu_layer_2")(dense_layer_2)

            dense_layer_3 = tensorflow.keras.layers.Dense(units=20, name="dense_layer_3")(activ_layer_2)

            before_lambda_model = tensorflow.keras.models.Model(input_layer, dense_layer_3, name="before_lambda_model")

            def custom_layer(tensor):
                return tensor + 2

            lambda_layer = tensorflow.keras.layers.Lambda(custom_layer, name="lambda_layer")(dense_layer_3)
            after_lambda_model = tensorflow.keras.models.Model(input_layer, lambda_layer, name="after_lambda_model")

            activ_layer_3 = tensorflow.keras.layers.ReLU(name="relu_layer_3")(lambda_layer)

            dense_layer_4 = tensorflow.keras.layers.Dense(units=10, name="dense_layer_4")(activ_layer_3)
            output_layer = tensorflow.keras.layers.Softmax(name="output_layer")(dense_layer_4)

            model = tensorflow.keras.models.Model(input_layer, output_layer, name="model")
        
        else:
            checkpoint = Determined().get_experiment(276).top_checkpoint()
            print ('checkpoint: ', checkpoint)
            model = checkpoint.load()   
            print ('model: ', model)

        model = self.context.wrap_model(model)
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005), loss="categorical_crossentropy")
        return model

    def build_training_data_loader(self) -> keras.InputData:

        (x_train, y_train), (_, _) = tensorflow.keras.datasets.mnist.load_data()

        x_train = x_train.astype(numpy.float64) / 255.0
        x_train = x_train.reshape((x_train.shape[0], numpy.prod(x_train.shape[1:])))
        y_train = tensorflow.keras.utils.to_categorical(y_train)
        print ('length of train: ', len(y_train))

        return x_train, y_train

    def build_validation_data_loader(self) -> keras.InputData:
        (_,_), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

        x_test = x_test.astype(numpy.float64) / 255.0
        x_test = x_test.reshape((x_test.shape[0], numpy.prod(x_test.shape[1:])))
        y_test = tensorflow.keras.utils.to_categorical(y_test)

        return x_test, y_test