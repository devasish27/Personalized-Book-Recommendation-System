# src/train.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
# IMPORTANT: when you run `python src/train.py` from project root,
# the script's directory is 'src' so imports of sibling modules work.
from data_loader import load_data, preprocess_data
from model import build_model

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')

    print("Project root:", project_root)
    print("Data dir:", data_dir)

    # load & preprocess
    df, users, books = load_data(data_dir=data_dir)
    # you can lower min_user_ratings/min_item_ratings to keep more data (e.g., 3 or 1)
    X_train, X_test, y_train, y_test, num_users, num_items, df_processed, user2idx, item2idx = preprocess_data(
        df, min_user_ratings=5, min_item_ratings=5, test_size=0.2
    )

    print("After preprocessing -> users:", num_users, " items:", num_items)
    print("Train size:", X_train.shape[0], " | Test size:", X_test.shape[0])

    # build model
    model = build_model(num_users, num_items, embedding_size=50)
    model.summary()

    # train (start with fewer epochs to test)
    history = model.fit(
        [X_train[:, 0], X_train[:, 1]],
        y_train,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
        epochs=10,
        batch_size=256,
        verbose=1
    )

    model.compile(optimizer="adam",
              loss=MeanSquaredError(),
              metrics=[MeanSquaredError()])

    # save model to project root
    out_path = os.path.join(project_root, "recommender_model.h5")
    model.save(out_path)
    print("Saved model to:", out_path)
