import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(num_users, num_items, embedding_size=50):
    user_input = layers.Input(shape=(1,), name="user")
    item_input = layers.Input(shape=(1,), name="item")

    user_embedding = layers.Embedding(num_users, embedding_size)(user_input)
    item_embedding = layers.Embedding(num_items, embedding_size)(item_input)

    user_vec = layers.Flatten()(user_embedding)
    item_vec = layers.Flatten()(item_embedding)

    concat = layers.Concatenate()([user_vec, item_vec])
    dense1 = layers.Dense(128, activation="relu")(concat)
    dense2 = layers.Dense(64, activation="relu")(dense1)
    output = layers.Dense(1, activation="linear")(dense2)

    model = keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    return model
