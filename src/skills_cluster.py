# skills_cluster.py
"""
Embedding-based deduplication & clustering for extracted raw candidate skill phrases.
Input: list[str] candidate phrases (raw from KeyBERT / noun-chunks)
Output: clusters -> representative phrase per cluster and embedding-based canonical mapping
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from rapidfuzz import process, fuzz
from typing import List, Tuple, Dict
import json
import os

# Load canonical vocab from JSON instead of hardcoding
def load_canonical_vocab(path="canonical_vocab.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Canonical vocab file not found: {path}")
    with open(path, "r") as f:
        vocab = json.load(f)
    return [v.lower().strip() for v in vocab]

# Load a small, fast embedding model
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Optional: small canonical mapping you already have in skills.py (extendable)
# Example canonical vocab (lowercase)
CANONICAL_VOCAB = [
    "python", "sql", "tensorflow", "pytorch", "scikit-learn", "pandas",
    "numpy", "docker", "kubernetes", "aws", "gcp", "azure", "airflow",
    "ml", "machine learning", "nlp", "data preprocessing", "model deployment",
    "fastapi", "streamlit", "keras", "spark", "hadoop", "ci/cd"
]

def _embed_phrases(phrases: List[str]) -> np.ndarray:
    if not phrases:
        return np.zeros((0, _EMBED_MODEL.get_sentence_embedding_dimension()))
    # model.encode supports batches; return NxD
    return _EMBED_MODEL.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)

def cluster_candidates_by_embedding(candidates: List[str],
                                    distance_threshold: float = 0.35,
                                    min_cluster_size: int = 1
                                   ) -> Tuple[Dict[int, List[int]], np.ndarray]:
    """
    Cluster candidate phrases using Agglomerative Clustering on cosine distances.
    - distance_threshold: linkage threshold in cosine distance (0..2). Lower = more clusters.
      Typical: 0.25..0.45 depending on noise.
    Returns:
      clusters: dict cluster_id -> list of candidate indices
      embeddings: numpy array of candidate embeddings
    """
    if not candidates:
        return {}, np.zeros((0, _EMBED_MODEL.get_sentence_embedding_dimension()))

    emb = _embed_phrases(candidates)  # normalized embeddings

    # Compute cosine distance matrix
    # AgglomerativeClustering with affinity='precomputed' expects a distance matrix
    #dist = cosine_distances(emb)  # NxN

    # Agglomerative clustering with single linkage on distance matrix
    # choose n_clusters=None and distance_threshold so we form clusters by threshold
    clustering = AgglomerativeClustering(n_clusters=None,
                                         metric="cosine",
                                         linkage="average",
                                         distance_threshold=distance_threshold)
    labels = clustering.fit_predict(emb)  # cluster labels for each candidate

    clusters = {}
    for idx, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(idx)

    # Optionally remove tiny clusters? keep all and let caller filter by size
    return clusters, emb

def cluster_representatives(candidates: List[str],
                            clusters: Dict[int, List[int]],
                            embeddings: np.ndarray
                           ) -> Dict[int, Dict]:
    """
    For each cluster, pick representative phrase (closest to cluster centroid).
    Returns mapping cluster_id -> {"rep": phrase, "indices": [...], "centroid": vec}
    """
    reps = {}
    for cid, idxs in clusters.items():
        cluster_embs = embeddings[idxs]
        centroid = cluster_embs.mean(axis=0)
        # pick candidate closest to centroid (cosine distance)
        dists = cosine_distances(cluster_embs, centroid.reshape(1, -1)).flatten()
        best_pos = int(np.argmin(dists))
        rep_idx = idxs[best_pos]
        reps[cid] = {
            "rep": candidates[rep_idx],
            "indices": idxs,
            "centroid": centroid
        }
    return reps

# fuzzy map to canonical vocab (use rapidfuzz)
def map_rep_to_canonical(rep_phrase: str, canonical_vocab: List[str]=None, cutoff: int = 78):
    if canonical_vocab is None:
        canonical_vocab = CANONICAL_VOCAB
    best = process.extractOne(rep_phrase, canonical_vocab, scorer=fuzz.token_sort_ratio)
    if not best:
        return None, 0
    cand, score, _ = best
    if score >= cutoff:
        return cand, score
    return None, score

def dedupe_and_canonicalize(candidates: List[str],
                            distance_threshold: float = 0.35,
                            canonical_vocab: List[str] = None,
                            fuzzy_cutoff: int = 78
                           ) -> Tuple[List[str], Dict]:
    """
    Main helper: cluster candidates, pick reps, map reps to canonicals.
    Returns:
      final_tokens: list[str] canonical or representative tokens (one per cluster)
      debug: mapping with cluster details (rep, mapped_canonical, members)
    """
    if canonical_vocab is None:
        canonical_vocab = load_canonical_vocab()
    # normalize input lower-case & strip
    cand_norm = [c.lower().strip() for c in candidates if c and c.strip()]
    if not cand_norm:
        return [], {}

    clusters, emb = cluster_candidates_by_embedding(cand_norm, distance_threshold=distance_threshold)
    reps = cluster_representatives(cand_norm, clusters, emb)

    final_tokens = []
    debug = {}
    for cid, info in reps.items():
        rep = info["rep"]
        mapped, score = map_rep_to_canonical(rep, canonical_vocab, cutoff=fuzzy_cutoff)
        token = mapped if mapped else rep  # prefer canonical mapping
        final_tokens.append(token)
        debug[cid] = {
            "rep": rep,
            "mapped": mapped,
            "map_score": score,
            "members": [cand_norm[i] for i in info["indices"]]
        }

    # dedupe final tokens preserving order
    seen = set()
    final_ordered = []
    for t in final_tokens:
        if t not in seen:
            seen.add(t)
            final_ordered.append(t)
    return final_ordered, debug

# ---------------- Example quick test function (can be used in scripts) -----------
def quick_cluster_demo():
    candidates = [
        "tensorflow pytorch experience data preprocessing model deployment cloud",
        "tf pytorch experience",
        "machine learning frameworks tensorflow",
        "docker and kubernetes for deployment",
        "deployment with docker",
        "python sql",
        "sql database",
        "nlp transformers",
        "natural language processing",
        "pandas and numpy"
    ]
    final, debug = dedupe_and_canonicalize(candidates, distance_threshold=0.5)
    print("Final tokens:", final)
    for cid, info in debug.items():
        print(f"[{cid}] rep: {info['rep']}, mapped:{info['mapped']}, members:{info['members']}")
