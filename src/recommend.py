# src/recommend.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.data_loader import load_data, preprocess_data
import random

def recommend_books(user_id, model_path="recommender_model.h5", top_n=10):
    """
    Generate top-N book recommendations for a given user_id.
    If user_id is not found, fall back to a random active user.
    """
    df, users, books = load_data()
    X_train, X_test, y_train, y_test, num_users, num_items, df_proc, user2idx, item2idx = preprocess_data(df)

    # Load model
    model = load_model(model_path, compile=False)


    # ✅ Handle missing users
    if user_id not in user2idx:
        print(f"⚠️ User {user_id} not found. Showing recommendations for a random active user instead.")
        user_id = random.choice(list(user2idx.keys()))

    user_idx = user2idx[user_id]

    # Candidate books (not rated by this user)
    rated_books = set(df_proc[df_proc["User-ID"] == user_idx]["ISBN"].values)
    candidates = [b for b in item2idx.values() if b not in rated_books]

    if not candidates:
        return pd.DataFrame(columns=["Book-Title", "Book-Author", "Publisher", "Image-URL-M"])

    # Predict ratings
    user_array = np.full(len(candidates), user_idx)
    preds = model.predict([user_array, np.array(candidates)], verbose=0).flatten()

    # Select top-N
    top_idx = preds.argsort()[-top_n:][::-1]
    top_books = [list(item2idx.keys())[list(item2idx.values()).index(candidates[i])] for i in top_idx]

    valid_cols = [c for c in ["Book-Title", "Book-Author", "Publisher", "Image-URL-M"] if c in books.columns]

    return books[books["ISBN"].isin(top_books)][valid_cols].drop_duplicates()

