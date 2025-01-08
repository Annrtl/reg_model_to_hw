import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import callbacks
import tensorflow as tf
import tflite_runtime.interpreter as tflite

keras.utils.set_random_seed(55)

FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)
ROOT_PATH = os.path.dirname(DIR_PATH)

def train_model(model, x_train, y_train, x_valid, y_valid):
    early_stopping = callbacks.EarlyStopping(
        min_delta=25,
        patience=200,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        batch_size=512,
        epochs=8172,
        callbacks=[early_stopping],
        verbose=1
    )

    history_df = pd.DataFrame(history.history)
    history = history_df.loc[:, ['loss', 'val_loss']]

    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Train loss')
    ax.plot(history['val_loss'], label='Valid loss')
    ax.legend()
    ax.grid()
    ax.set_title('MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.yscale('log')
    plt.show()

    model.save_weights(os.path.join(ROOT_PATH, 'model/model.h5'))

    # serialize model to JSON
    json_model = json.loads(model.to_json())
    json_model['weights'] = model.get_weights()

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    json_str = json.dumps(json_model, cls=NumpyEncoder)

    with open(os.path.join(ROOT_PATH, 'model/model.json'), 'w') as json_file:
        json_file.write(json_str)

    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #tflite_model = converter.convert()
    #tflite_model.save_weights(os.path.join(ROOT_PATH, 'model/model_lite.h5'))

if __name__ == '__main__':
    train_csv = pd.read_csv(os.path.join(ROOT_PATH, 'data/house_prices/train.csv'))
    target = 'SalePrice'

    train_csv = train_csv.select_dtypes(exclude=['object'])
    train_csv = train_csv.dropna()

    train_corr_matrix = train_csv.corr()

    for c in train_csv.columns:
        if abs(train_corr_matrix[target][c]) < 0.1:
            train_csv = train_csv.drop(c, axis=1)

    train = train_csv.sample(frac=0.8)
    valid = train_csv.drop(train.index)

    x_train = train.copy()
    y_train = x_train.pop(target)
    x_valid = valid.copy()
    y_valid = x_valid.pop(target)

    input_shape = x_train.shape[1]
    input_shape = [input_shape]

    print(f"Input shape is {input_shape}")

    keras_const_pos = keras.constraints.MinMaxNorm(min_value=-7, max_value=7, rate=1.0, axis=0)
    activation = "relu"
    model = keras.Sequential([
        keras.layers.Dense(32, activation=activation, kernel_constraint=keras_const_pos, input_shape=input_shape),
        keras.layers.Dense(32, activation=activation, kernel_constraint=keras_const_pos),
        keras.layers.Dense(32, activation=activation, kernel_constraint=keras_const_pos),
    ])

    model.compile(
        optimizer='adam',
        loss='mae',
    )

    model.summary()

    do_load = False

    if do_load and os.path.isfile(os.path.join(ROOT_PATH, 'model/model.h5')):
        model.load_weights(os.path.join(ROOT_PATH, 'model/model.h5'))
    else:
        train_model(model, x_train, y_train, x_valid, y_valid)