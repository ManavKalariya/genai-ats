import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (run once)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Lowercase
    text = text.lower()
    
    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(filtered_tokens)
