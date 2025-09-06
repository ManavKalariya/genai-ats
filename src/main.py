from extractor import extract_text_from_pdf, extract_text_from_docx
from cleaner import clean_text

resume_text = extract_text_from_pdf("../data/resumes/sample_resume.pdf")
jd_text = open("../data/jobs/sample_jd.txt").read()

print("=== Raw Resume ===")
print(resume_text)  # First 400 chars

print("\n=== Cleaned Resume ===")
print(clean_text(resume_text))

print("\n=== Job Description ===")
print(jd_text)
