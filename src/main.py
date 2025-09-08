from extractor import extract_text_from_pdf
from cleaner import clean_text
from embedder import get_similarity
from visualizer import plot_match_score
from skills import extract_skills, compare_skills   # NEW

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

     # === Milestone 3: Skill Extraction ===
    resume_skills, resume_raw = extract_skills(resume_clean, top_n=10)
    jd_skills, jd_raw = extract_skills(jd_clean, top_n=10)

    print("Resume raw candidates:", resume_raw)
    print("\nExtracted Resume Skills:", resume_skills)
    print("\nJD raw candidates:", jd_raw)
    print("\nExtracted JD Skills:", jd_skills)

    missing_skills, matched_map = compare_skills(resume_skills, jd_skills)
    print("\nMissing Skills (final):", missing_skills)
    print("\nMatched map:", matched_map)

    # Save results
    with open("../outputs/missing_skills.txt", "w", encoding="utf-8") as f:
        f.write("Resume Skills:\n" + ", ".join(resume_skills) + "\n\n")
        f.write("JD Skills:\n" + ", ".join(jd_skills) + "\n\n")
        f.write("Missing Skills:\n" + ", ".join(missing_skills))
