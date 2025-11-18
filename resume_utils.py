import os
import re
import pdfplumber
import docx2txt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

COMMON_SKILLS = [
    "python","java","c++","c","c#","sql","aws","azure","gcp","pandas","numpy",
    "docker","kubernetes","react","node.js","javascript","html","css",
    "machine learning","deep learning","nlp","computer vision","tableau",
    "excel","linux"
]

def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text += txt + "\n"
    except:
        pass
    return text

def extract_text_from_docx(path: str) -> str:
    try:
        return docx2txt.process(path)
    except:
        return ""

def load_resume_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    if ext == ".txt":
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    return ""

def extract_skills(text: str) -> List[str]:
    text_clean = text.lower()
    found = []
    for skill in COMMON_SKILLS:
        if skill.lower() in text_clean:
            found.append(skill)
    return sorted(set(found))

def check_contact(text):
    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone = re.search(r"(\+?\d[\d \-\(\)]{7,}\d)", text)
    return bool(email), bool(phone)

def ats_score(path: str, text: str):
    score = 0
    details = {}

    if any(path.endswith(ext) for ext in [".pdf", ".docx", ".txt"]):
        score += 10

    email_ok, phone_ok = check_contact(text)
    if email_ok and phone_ok:
        score += 20
    details["email"] = email_ok
    details["phone"] = phone_ok

    headings = ["education","experience","projects","skills"]
    count = sum(1 for h in headings if h in text.lower())
    score += min(20, count * 5)
    details["section_score"] = count

    skills = extract_skills(text)
    skill_coverage = len(skills) / len(COMMON_SKILLS)
    score += int(skill_coverage * 30)
    details["skills_found"] = skills

    score = min(100, score)
    details["final"] = score
    return score, details

COMPANY_DB = [
    {"name":"DataCorp","skills":["python","sql","aws","pandas"],"tier":"A"},
    {"name":"AI StartUp","skills":["python","deep learning","nlp"],"tier":"A"},
    {"name":"CloudOps","skills":["aws","docker","kubernetes"],"tier":"A"},
    {"name":"Webify","skills":["javascript","react","node.js"],"tier":"B"},
]

def recommend_companies(skills):
    names = [c["name"] for c in COMPANY_DB]
    docs = [" ".join(c["skills"]) for c in COMPANY_DB]
    user_doc = " ".join(skills)
    vect = TfidfVectorizer().fit(docs + [user_doc])
    sims = cosine_similarity(vect.transform([user_doc]), vect.transform(docs))[0]
    scores = sorted(zip(names, sims), key=lambda x: -x[1])
    return scores[:5]

def analyze_resume(path, trained_model=None):
    text = load_resume_text(path)
    skills = extract_skills(text)
    ats, details = ats_score(path, text)
    recs = recommend_companies(skills)
    tier, conf = (None, None)
    if trained_model:
        tier, conf = trained_model["predict"](skills)
    return {
        "skills": skills,
        "ats": ats,
        "ats_details": details,
        "company_recommendations": recs,
        "placement_prediction": {"tier": tier, "confidence": conf}
    }