# src/data_loader.py
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def find_csv_by_keywords(data_dir, keywords):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    files_lower = {os.path.basename(f).lower(): f for f in files}
    # 1) exact keyword in filename (best)
    for kw in keywords:
        for name, path in files_lower.items():
            if kw in name:
                return path
    # 2) fallback: pick first file that starts with 'bx-' and contains kw
    for name, path in files_lower.items():
        if name.startswith("bx-"):
            for kw in keywords:
                if kw in name:
                    return path
    # 3) ultimate fallback: return any csv if exists
    return files[0] if files else None

def read_csv_flexible(path):
    # Try common separators — return dataframe when we get >1 column
    for sep in [';', '\t', ',']:
        try:
            df = pd.read_csv(path, sep=sep, encoding='latin-1', on_bad_lines="skip", low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    # last resort (let pandas guess)
    return pd.read_csv(path, encoding='latin-1', on_bad_lines="skip", low_memory=False)

def _find_col(cols, candidates):
    cols_norm = [c.lower().replace(' ', '').replace('-', '') for c in cols]
    for cand in candidates:
        cand_norm = cand.lower().replace(' ', '').replace('-', '')
        for i, c in enumerate(cols_norm):
            if cand_norm == c or cand_norm in c or c in cand_norm:
                return cols[i]
    return None

def load_data(data_dir=None, users_path=None, books_path=None, ratings_path=None):
    """
    Returns: df (merged ratings+books), users_df, books_df
    The function is robust: will try to auto-detect filenames and separators.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data')

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    # try to detect files if explicit paths not given
    if users_path is None:
        users_path = find_csv_by_keywords(data_dir, ['user', 'users'])
    if books_path is None:
        books_path = find_csv_by_keywords(data_dir, ['book', 'books'])
    if ratings_path is None:
        ratings_path = find_csv_by_keywords(data_dir, ['rating', 'ratings', 'book-rating', 'bookratings'])

    if not users_path or not books_path or not ratings_path:
        raise FileNotFoundError(f"Couldn't find Users/Books/Ratings csv in {data_dir}. Files found: {os.listdir(data_dir)}")

    print("Detected files:")
    print(" USERS :", users_path)
    print(" BOOKS :", books_path)
    print(" RATINGS:", ratings_path)

    users = read_csv_flexible(users_path)
    books = read_csv_flexible(books_path)
    ratings = read_csv_flexible(ratings_path)

    print("\nUsers columns:", users.columns.tolist())
    print("Books columns:", books.columns.tolist())
    print("Ratings columns:", ratings.columns.tolist())
    print("----")

    # try to locate expected columns (flexible names)
    user_id_col = _find_col(users.columns, ['User-ID', 'UserID', 'userid', 'user-id', 'user'])
    location_col = _find_col(users.columns, ['Location', 'location'])
    age_col = _find_col(users.columns, ['Age', 'age'])

    isbn_books_col = _find_col(books.columns, ['ISBN', 'isbn'])
    title_col = _find_col(books.columns, ['Book-Title', 'Title', 'booktitle', 'book-title'])
    author_col = _find_col(books.columns, ['Book-Author','Author','bookauthor'])
    yop_col = _find_col(books.columns, ['Year-Of-Publication','Year','yearofpublication'])

    user_id_r_col = _find_col(ratings.columns, ['User-ID','UserID','userid','user'])
    isbn_r_col = _find_col(ratings.columns, ['ISBN','isbn'])
    rating_col = _find_col(ratings.columns, ['Book-Rating','Rating','bookrating','book-rating','rating'])

    if not user_id_col or not isbn_books_col or not isbn_r_col or not rating_col:
        raise ValueError("Could not detect required columns. See printed column lists above and rename or provide explicit paths.")

    # select and rename to standardized column names
    users_sel = {}
    if user_id_col: users_sel[user_id_col] = 'User-ID'
    if location_col: users_sel[location_col] = 'Location'
    if age_col: users_sel[age_col] = 'Age'
    users = users[list(users_sel.keys())].rename(columns=users_sel)

    books_sel = {isbn_books_col: 'ISBN'}
    if title_col: books_sel[title_col] = 'Book-Title'
    if author_col: books_sel[author_col] = 'Book-Author'
    if yop_col: books_sel[yop_col] = 'Year-Of-Publication'
    books = books[[k for k in books_sel.keys() if k in books.columns]].rename(columns=books_sel)

    ratings_sel = {user_id_r_col: 'User-ID', isbn_r_col: 'ISBN', rating_col: 'Book-Rating'}
    ratings = ratings[[k for k in ratings_sel.keys() if k in ratings.columns]].rename(columns=ratings_sel)

    # ensure numeric rating
    try:
        ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')
    except Exception:
        pass
    ratings = ratings.dropna(subset=['Book-Rating'])
    # optional: keep only explicit positive ratings
    ratings = ratings[ratings['Book-Rating'] > 0]

    # merge ratings with books (inner join ensures only books with metadata kept)
    df = ratings.merge(books, on='ISBN', how='inner')

    print(f"After merge: {len(df)} rating rows, unique users: {df['User-ID'].nunique()}, unique books: {df['ISBN'].nunique()}")
    return df.reset_index(drop=True), users.reset_index(drop=True), books.reset_index(drop=True)


def preprocess_data(df, min_user_ratings=5, min_item_ratings=5, test_size=0.2, random_state=42):
    """
    Filters out very inactive users/items (optional) and encodes user/item -> integer indices.
    Returns: X_train, X_test, y_train, y_test, num_users, num_items, df_processed, user2idx, item2idx
    """
    # filter users/items with few ratings to reduce sparsity (you can set min_* to 1 to keep all)
    user_counts = df['User-ID'].value_counts()
    good_users = user_counts[user_counts >= min_user_ratings].index
    item_counts = df['ISBN'].value_counts()
    good_items = item_counts[item_counts >= min_item_ratings].index

    df_f = df[df['User-ID'].isin(good_users) & df['ISBN'].isin(good_items)].copy()
    df_f = df_f.reset_index(drop=True)
    if df_f.empty:
        raise ValueError("After filtering by min_user_ratings/min_item_ratings the dataset is empty. Lower the thresholds.")

    # encodings
    user_ids = df_f['User-ID'].unique().tolist()
    item_ids = df_f['ISBN'].unique().tolist()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {b: i for i, b in enumerate(item_ids)}

    df_f['user'] = df_f['User-ID'].map(user2idx)
    df_f['item'] = df_f['ISBN'].map(item2idx)

    X = df_f[['user', 'item']].values
    y = df_f['Book-Rating'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, len(user2idx), len(item2idx), df_f, user2idx, item2idx
