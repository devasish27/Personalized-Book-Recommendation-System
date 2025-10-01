# src/evaluate.py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from data_loader import load_data, preprocess_data
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# Load trained model
model = load_model("../recommender_model.h5", compile=False)



# =========================
# 📌 Metrics
# =========================
def precision_at_k(y_true, y_pred, k=10, threshold=5):
    """
    Precision@K: fraction of recommended items in top-K that are relevant
    """
    if len(y_true) < k:
        return None
    top_k_idx = np.argsort(y_pred)[-k:]
    relevant_idx = np.where(y_true >= threshold)[0]
    if len(relevant_idx) == 0:
        return None
    prec = len(set(top_k_idx) & set(relevant_idx)) / k
    return prec

def recall_at_k(y_true, y_pred, k=10, threshold=5):
    """
    Recall@K: fraction of relevant items that are in the top-K
    """
    if len(y_true) < k:
        return None
    top_k_idx = np.argsort(y_pred)[-k:]
    relevant_idx = np.where(y_true >= threshold)[0]
    if len(relevant_idx) == 0:
        return None
    rec = len(set(top_k_idx) & set(relevant_idx)) / len(relevant_idx)
    return rec

def ndcg_at_k(y_true, y_pred, k=10):
    """
    Normalized Discounted Cumulative Gain at K
    """
    if len(y_true) < k:
        return None
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.take(y_true, order[:k])

    dcg = np.sum((2**y_true_sorted - 1) / np.log2(np.arange(2, k+2)))
    ideal_order = np.argsort(y_true)[::-1]
    ideal_sorted = np.take(y_true, ideal_order[:k])
    idcg = np.sum((2**ideal_sorted - 1) / np.log2(np.arange(2, k+2)))

    return dcg / idcg if idcg > 0 else None

# =========================
# 📌 Evaluation Script
# =========================
if __name__ == "__main__":
    # Load dataset
    df, users, books = load_data()
    X_train, X_test, y_train, y_test, num_users, num_items, df_proc, user2idx, item2idx = preprocess_data(df)


    # Predict ratings
    y_pred = model.predict([X_test[:,0], X_test[:,1]]).flatten()

    # ---- RMSE ----
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ Test RMSE: {rmse:.4f}")

    # ---- Ranking Metrics (sampled over users) ----
    users_test = np.unique(X_test[:,0])
    precisions, recalls, ndcgs = [], [], []

    for u in users_test[:500]:  # evaluate on first 500 users for speed
        mask = (X_test[:,0] == u)
        y_true_u = y_test[mask]
        y_pred_u = y_pred[mask]

        if len(y_true_u) < 10:  # skip users with very few ratings
            continue

        p = precision_at_k(y_true_u, y_pred_u, k=10)
        r = recall_at_k(y_true_u, y_pred_u, k=10)
        n = ndcg_at_k(y_true_u, y_pred_u, k=10)

        if p is not None: precisions.append(p)
        if r is not None: recalls.append(r)
        if n is not None: ndcgs.append(n)

    if precisions:
        print(f"📌 Precision@10: {np.mean(precisions):.4f}")
    if recalls:
        print(f"📌 Recall@10: {np.mean(recalls):.4f}")
    if ndcgs:
        print(f"📌 NDCG@10: {np.mean(ndcgs):.4f}")
