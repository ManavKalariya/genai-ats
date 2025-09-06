from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once (MiniLM is small & fast)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding for input text"""
    return model.encode([text])[0]  # returns a 384-dim vector

def get_similarity(resume_text: str, jd_text: str) -> float:
    """Compute cosine similarity between resume & JD embeddings"""
    resume_emb = get_embedding(resume_text)
    jd_emb = get_embedding(jd_text)
    similarity = cosine_similarity([resume_emb], [jd_emb])[0][0]
    return round(similarity * 100, 2)  # percentage score
