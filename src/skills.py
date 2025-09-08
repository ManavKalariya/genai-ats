# skills.py  — improved skill extraction + robust comparison
import re
import spacy
from keybert import KeyBERT
from rapidfuzz import process, fuzz
from skills_cluster import dedupe_and_canonicalize
import json
import os

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

def load_group_requirements(path="group_requirements.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Group requirements file not found: {path}")
    with open(path, "r") as f:
        groups = json.load(f)
    # normalize everything to lowercase
    return {k.lower(): [v.lower() for v in vals] for k, vals in groups.items()}

GROUP_REQUIREMENTS = load_group_requirements()

# --- SMALL canonical skill synonyms mapping (extend as needed) ---
# keys are synonyms/variants -> value is canonical skill token
SKILL_SYNONYMS = {
    # python & data
    "python": "python", "py": "python",
    "sql": "sql",

    # ML frameworks
    "tensorflow": "tensorflow", "tf": "tensorflow",
    "pytorch": "pytorch", "torch": "pytorch",
    "keras": "keras",
    "scikit-learn": "scikit-learn", "sklearn": "scikit-learn",

    # Data libs
    "pandas": "pandas", "numpy": "numpy",

    # infra / deployment
    "docker": "docker", "kubernetes": "kubernetes", "k8s": "kubernetes",
    "aws": "aws", "amazon web services": "aws",
    "gcp": "gcp", "google cloud": "gcp", "azure": "azure",

    # orchestration / ci
    "airflow": "airflow", "kafka": "kafka",
    "ci/cd": "ci/cd", "github actions": "ci/cd",

    # web frameworks / tools
    "fastapi": "fastapi", "streamlit": "streamlit",

    # concepts
    "machine learning": "machine learning", "ml": "machine learning",
    "nlp": "nlp", "natural language processing": "nlp",
    "data preprocessing": "data preprocessing", "data cleaning": "data preprocessing",
    "model deployment": "model deployment",
    "feature engineering": "feature engineering",
    "spark": "spark", "hadoop": "hadoop",
    "etl": "etl",
    "lstm": "lstm", "cnn": "cnn",
}

# canonical vocab (unique values)
CANONICAL_SKILLS = set(SKILL_SYNONYMS.values())

# group heuristics: JD phrases describing a class of skills -> acceptable set
# GROUP_REQUIREMENTS = {
#     "ml frameworks": {"tensorflow", "pytorch", "scikit-learn", "keras"},
#     "machine learning": {"machine learning", "tensorflow", "pytorch", "scikit-learn", "keras"},
#     "data preprocessing": {"data preprocessing", "pandas", "spark"},
#     "model deployment": {"docker", "kubernetes", "aws", "gcp", "ci/cd", "airflow"},
#     "cloud": {"aws", "gcp", "azure"},
#     "nlp": {"nlp", "spacy", "transformers"},
# }

# stopwords / junk tokens to remove from candidate phrases
JUNK_PATTERNS = [
    r"\bemail\b", r"\bphone\b", r"\blinkedin\b", r"www\.", r"\.com\b",
    r"\bresume\b", r"\bcv\b"
]

# ---------------------- utilities ---------------------------------
def _clean_text_for_candidates(text: str) -> str:
    # remove emails, phones, urls, weird chars (but keep words)
    text = re.sub(r"\S+@\S+", " ", text)            # emails
    text = re.sub(r"\+?\d[\d\-\s]{6,}\d", " ", text)  # phone-like
    text = re.sub(r"http\S+", " ", text)            # urls
    text = re.sub(r"[^A-Za-z0-9\s\-/+\.()]", " ", text)  # keep slashes, dashes
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def _get_person_names(text: str):
    doc = nlp(text)
    return {ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"}

def _is_junk_phrase(phrase: str) -> bool:
    if len(phrase.strip()) == 0:
        return True
    if len(phrase) < 2:
        return True
    # too long (>3 words) or too short characters-only
    if len(phrase.split()) > 3:
        return True
    for p in JUNK_PATTERNS:
        if re.search(p, phrase):
            return True
    # numbers-only or single punctuation
    if re.fullmatch(r"[\d\W]+", phrase):
        return True
    return False

def _split_candidate(phrase: str):
    # split on common separators and filler words to get atomic tokens
    # e.g., "tensorflow pytorch experience data preprocessing model deployment cloud"
    # -> ["tensorflow", "pytorch", "data preprocessing", "model deployment", "cloud"]
    separators = r",|/|;|\band\b|\bwith\b|\busing\b|\bexperience\b|\bexperience with\b|\bfor\b|\bin\b"
    parts = re.split(separators, phrase)
    parts = [p.strip() for p in parts if p and len(p.strip())>0]
    # further split long items into words if necessary
    tokens = []
    for p in parts:
        words = p.split()
        if len(words) <= 3:
            tokens.append(p)
        else:
            # break into 1-3 word sliding windows to try to extract tech tokens
            for i in range(len(words)):
                for j in range(i+1, min(i+4, len(words)+1)):
                    candidate = " ".join(words[i:j]).strip()
                    if len(candidate.split()) <= 3:
                        tokens.append(candidate)
    # dedup
    uniq = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)
    return uniq

def _fuzzy_map_to_canonical(phrase: str, score_cutoff: int = 78):
    """
    Try to map a phrase to a canonical skill via fuzzy match on SKILL_SYNONYMS keys.
    Returns canonical skill or None.
    """
    choices = list(SKILL_SYNONYMS.keys())
    best = process.extractOne(phrase, choices, scorer=fuzz.token_sort_ratio)
    if not best:
        return None
    candidate_key, score, _ = best
    if score >= score_cutoff:
        return SKILL_SYNONYMS[candidate_key]
    return None

# --------------------- main extraction -------------------------
def extract_skills(text: str, top_n: int = 20, use_keybert: bool = True):
    """
    Improved skill extraction:
    - removes PII (person names, emails, phones)
    - uses spaCy noun-chunks + KeyBERT candidate phrases
    - splits compound phrases into atomic tokens (<=3 words)
    - normalizes and fuzzy-maps to canonical skills when possible
    - returns list of canonical skills and a list of raw_candidates for debugging
    """
    text_clean = _clean_text_for_candidates(text)
    person_names = _get_person_names(text)
    candidates = []

    # 1) spaCy noun chunks (quick candidates)
    doc = nlp(text_clean)
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks]
    candidates.extend(noun_chunks)

    # 2) KeyBERT candidates
    if use_keybert:
        try:
            kws = kw_model.extract_keywords(text_clean, keyphrase_ngram_range=(1, 3),
                                            stop_words="english", top_n=top_n)
            keybert_phrases = [kw[0] for kw in kws]
            candidates.extend(keybert_phrases)
        except Exception:
            # fallback: ignore KeyBERT failures
            pass

    # 3) Clean, split and filter candidates
    processed = []
    for c in candidates:
        c = c.lower().strip()
        if not c:
            continue
        # split compound candidates into smaller tokens
        pieces = _split_candidate(c)
        for p in pieces:
            p = p.strip().lower()
            # remove person names / PII
            if any(name in p for name in person_names):
                continue
            if _is_junk_phrase(p):
                continue
            processed.append(p)

    # 4) Map each processed candidate to a canonical skill if possible (fuzzy)
    canonical_found = []
    raw_candidates = []
    for p in processed:
        raw_candidates.append(p)
        mapped = _fuzzy_map_to_canonical(p)
        if mapped:
            canonical_found.append(mapped)
        else:
            # keep p if it looks like a tech token (alphanumeric + <=3 words)
            if 0 < len(p.split()) <= 3:
                canonical_found.append(p)

    final_tokens, debug = dedupe_and_canonicalize(raw_candidates,
                                              distance_threshold=0.6,
                                              canonical_vocab=None,  # uses built-in CANONICAL_VOCAB
                                              fuzzy_cutoff=78)

    # 5) deduplicate & prioritize canonical vocabulary
    # prefer canonical names from CANONICAL_SKILLS
    # final = []
    # for s in canonical_found:
    #     s = s.lower().strip()
    #     if s in final:
    #         continue
    #     # prefer canonical normalized token (map alias->canonical)
    #     if s in SKILL_SYNONYMS:
    #         s = SKILL_SYNONYMS[s]
    #     final.append(s)

    # final filtering: remove junk again, and prune phrases longer than 3 words
    final_tokens = [f for f in final_tokens if not _is_junk_phrase(f) and len(f.split()) <= 3]

    return final_tokens, raw_candidates

# --------------------- robust comparison -------------------------
def compare_skills(resume_skills, jd_skills, debug=False):
    """
    resume_skills, jd_skills: lists returned by extract_skills (canonical tokens)
    returns: missing_skills list + matched_map {jd_skill: matched_resume_skill or None}
    Logic:
      - For each jd_skill phrase, check if any resume canonical skill satisfies it.
      - Uses group heuristics: e.g., 'ml frameworks' in JD is satisfied if resume has tensorflow/pytorch/scikit-learn/...
      - Uses direct equality/fuzzy canonical mapping if needed.
    """
    R = set([r.lower() for r in resume_skills])
    missing = []
    matched_map = {}

    # normalize jd tokens: ensure they are atomized (split compounds)
    jd_atomic = []
    for jd in jd_skills:
        # split jd similar to extract logic to get atomic units
        jd_parts = _split_candidate(jd.lower())
        jd_atomic.extend(jd_parts)

    # dedup preserve order
    seen = set()
    jd_atomic = [x for x in jd_atomic if not (x in seen or seen.add(x))]

    for jd in jd_atomic:
        jd_norm = jd.strip().lower()
        satisfied = False
        match = None

        # 1) direct set intersection with resume skills
        for r in R:
            if r == jd_norm:
                satisfied = True
                match = r
                break

        if not satisfied:
            # 2) fuzzy compare to resume skills
            best = None
            try:
                best = process.extractOne(jd_norm, list(R), scorer=fuzz.token_sort_ratio)
            except Exception:
                best = None
            if best:
                cand, score, _ = best
                if score >= 78:
                    satisfied = True
                    match = cand

        if not satisfied:
            # 3) group heuristics: if jd contains group keywords, check resume for any group members
            for group_key, group_set in GROUP_REQUIREMENTS.items():
                if group_key in jd_norm or any(token in jd_norm for token in group_key.split()):
                    # if resume has any of group_set -> satisfied
                    if R.intersection(group_set):
                        satisfied = True
                        match = list(R.intersection(group_set))[0]
                        break

        if not satisfied:
            missing.append(jd_norm)
            matched_map[jd_norm] = None
        else:
            matched_map[jd_norm] = match

        if debug:
            print(f"JD token: '{jd_norm}' -> satisfied: {satisfied}, match: {match}")

    return missing, matched_map
