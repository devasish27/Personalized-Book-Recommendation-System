# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.recommend import recommend_books

app = FastAPI(title="📚 Book Recommendation API")

class UserRequest(BaseModel):
    user_id: int
    top_n: int = 10

@app.post("/recommend")
def get_recommendations(req: UserRequest):
    recs = recommend_books(req.user_id, top_n=req.top_n)
    return {"user_id": req.user_id, "recommendations": recs.to_dict(orient="records")}
