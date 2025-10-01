# app/streamlit_app.py
import sys, os
import streamlit as st

# ensure src/ is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommend import recommend_books

st.set_page_config(page_title="📚 Book Recommender", layout="wide")

st.title("📚 Personalized Book Recommender System")
st.markdown("Enter a **User ID** to get recommendations (if not found, random user will be used).")

user_id = st.text_input("User ID", "276729")  # one of the known active IDs
top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations"):
    user_id_clean = str(user_id).strip()
    if not user_id_clean.isdigit():
        st.error("Please enter a valid integer User ID.")
    else:
        user_id_int = int(user_id_clean)
        recs = recommend_books(user_id_int, top_n=top_n)

        if recs.empty:
            st.warning("No recommendations available.")
        else:
            st.success(f"Top {top_n} books (User {user_id_int} or fallback):")
            cols = st.columns(3)

            for i, row in recs.iterrows():
                col = cols[i % 3]
                with col:
                    # Container "card"
                    st.markdown(
                        """
                        <div style="border:1px solid #ddd; border-radius:10px; padding:10px; text-align:center; margin-bottom:20px; background-color:#fafafa;">
                        """,
                        unsafe_allow_html=True,
                    )

                    # ✅ Image
                    img_url = row.get("Image-URL-M", "")
                    if isinstance(img_url, str) and img_url.startswith("http"):
                        st.image(img_url, width=120)
                    else:
                        st.image("https://via.placeholder.com/120x160?text=No+Cover", width=120)

                    # ✅ Title (clickable Google link)
                    book_title = row.get("Book-Title", "Unknown Title")
                    google_link = f"https://www.google.com/search?q={book_title.replace(' ', '+')}"
                    st.markdown(f"<a href='{google_link}' target='_blank'><b>{book_title}</b></a>", unsafe_allow_html=True)

                    # ✅ Author + Publisher
                    st.markdown(f"<p style='font-size:13px; color:gray;'>✍️ {row.get('Book-Author', 'Unknown')}</p>", unsafe_allow_html=True)
                    if "Publisher" in row:
                        st.markdown(f"<p style='font-size:12px; color:#555;'>🏢 {row['Publisher']}</p>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
