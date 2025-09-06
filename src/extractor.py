import pdfplumber
import docx

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF resume"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX resume"""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()
