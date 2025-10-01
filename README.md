
# 📚 Personalized Book Recommendation System  

An AI-powered book recommendation engine built with **TensorFlow/Keras** on the popular **Book-Crossing dataset**.  
The system learns user preferences using **neural collaborative filtering (matrix factorization with embeddings)** and provides **personalized book suggestions** via an API and Streamlit app.  

---

## ✨ Features  
- **Dataset**: Book-Crossing (68k users, 149k books, 383k ratings).  
- **Model**: Embedding-based collaborative filtering using Keras.  
- **Evaluation Metrics**:  
  - ✅ RMSE ~1.79  
  - ✅ Precision@10: 0.98  
  - ✅ Recall@10: 0.69  
  - ✅ NDCG@10: 0.71  
- **Apps**:  
  - 🔹 REST API with FastAPI (`app/api.py`)  
  - 🔹 Interactive Streamlit app (`app/streamlit_app.py`) with book covers, author, publisher  

---

## 📊 Project Structure  
```

Recommender-System/
│
├── app/
│   ├── api.py              # FastAPI for recommendations
│   ├── streamlit_app.py    # Streamlit UI for recommendations
│
├── src/
│   ├── train.py            # Train the recommender model
│   ├── evaluate.py         # Evaluate model performance
│   ├── recommend.py        # Core recommendation logic
│   ├── data_loader.py      # Load & preprocess dataset
│   └── model.py            # Model architecture
│
├── data/
│   ├── BX-Users.csv
│   ├── BX-Books.csv
│   └── BX-Book-Ratings.csv
│
├── recommender_model.h5    # Saved trained model
├── requirements.txt
└── README.md

````

---

## 🚀 Installation  

1. Clone the repository  
   ```bash
   git clone https://github.com/<your-username>/Recommender-System.git
   cd Recommender-System
   ```

2. Create & activate virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```


## ⚡ Usage

### 1️⃣ Train the model

```bash
python src/train.py
```

### 2️⃣ Evaluate the model

```bash
python src/evaluate.py
```

### 3️⃣ Run the API (FastAPI)

```bash
uvicorn app.api:app --reload
```

### 4️⃣ Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## 🎨 Streamlit UI

* Enter a **User ID** and number of recommendations.
* Get a grid of book covers, clickable titles, authors, and publishers.
* If a user ID isn’t found, the system falls back to a random active user.

---

## ✅ Applications

* Personalized book suggestions (like Goodreads/Amazon).
* Extendable to **movies, music, or e-commerce recommendation systems**.
* Great **portfolio project** for AI/ML engineers and data scientists.

---

## 🛠 Tech Stack

* **Python**
* **TensorFlow/Keras** – model training
* **Pandas & NumPy** – data preprocessing
* **FastAPI** – backend API
* **Streamlit** – interactive frontend

---

## 📌 Future Improvements

* Add hybrid recommendations (content + collaborative).
* Integrate Goodreads/Google Books API for real book links.
* Deploy with Docker + cloud hosting.

---

## 📜 License

MIT License – free to use and modify.

---
