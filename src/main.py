from extractor import extract_text_from_pdf
from cleaner import clean_text
from embedder import get_similarity
from visualizer import plot_match_score

if __name__ == "__main__":
    # Load resume (PDF) and JD
    resume_text = extract_text_from_pdf("../data/resumes/sample_resume.pdf")
    jd_text = open("../data/jobs/sample_jd.txt").read()

    # Clean text
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # Get similarity score
    match_score = get_similarity(resume_clean, jd_clean)
    print(f"\nResume–JD Match Score: {match_score}%")

    # Visualize
    plot_match_score(match_score)
