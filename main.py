import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

keras.utils.set_random_seed(55)

def train_model(model, x_train, y_train, x_valid, y_valid):
    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        batch_size=512,
        epochs=64,
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
    # plt.show()

    model.save_weights("model/model.h5")

    # serialize model to JSON
    json_model = json.loads(model.to_json())
    json_model['weights'] = model.get_weights()

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    json_str = json.dumps(json_model, cls=NumpyEncoder)

    with open("model/model.json", "w") as json_file:
        json_file.write(json_str)

#train_csv = pd.read_csv('phone_price/train.csv')
#target = 'clock_speed'
train_csv = pd.read_csv('data/housing.csv')
target = 'MedHouseVal'

train = train_csv.sample(frac=0.8)
valid = train_csv.drop(train.index)

x_train = train.copy()
y_train = x_train.pop(target)
x_train.pop('id')

x_valid = valid.copy()
y_valid = x_valid.pop(target)
x_valid.pop('id')

input_shape = x_train.shape[1]
input_shape = [input_shape]

print(f"Input shape is {input_shape}")

model = keras.Sequential([
    keras.layers.Dense(1, input_shape=input_shape),
])

model.compile(
    optimizer='adam',
    loss='mae',
)

if os.path.isfile("model/model.h5"):
    model.load_weights("model/model.h5")
else:
    train_model(model, x_train, y_train, x_valid, y_valid)

test = train_csv.sample(n=10)
for index, rows in test.iterrows():
    row = rows.copy()
    row = row.to_frame()
    row = row.transpose()
    expected = row.pop(target)
    row.pop('id')
    res = model.predict(row, verbose=0)
    argv = row.values.tolist()[0]
    argv = list(map(str, argv))
    argv = " ".join(argv)
    cpp_predict = os.popen(f"./build/nn {argv}").read()
    print(f"Predicted: {res[0][0]}, cpp predicted: {cpp_predict} and Expected: {expected.iloc[0]}")