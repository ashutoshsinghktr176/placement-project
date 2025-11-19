from fastapi import FastAPI
from pydantic import BaseModel

from placement_model import load_model, get_prediction
from resume_utils import extract_skills


# ---------- Load model at startup ----------

try:
    trained_model = load_model()
except FileNotFoundError:
    # This will make the Render deploy fail clearly if model.pkl is missing
    raise RuntimeError("Application failed to start because model.pkl file was not found.")
except Exception as e:
    # Any other error while loading the model
    raise RuntimeError(f"Application failed to start because the model could not be loaded: {e}")


# ---------- FastAPI app ----------

app = FastAPI(title="Resume Analyzer & Placement Predictor")


# ---------- Request models ----------

class ResumeInput(BaseModel):
    text: str  # Full resume text


class SkillsInput(BaseModel):
    skills: list[str]  # List of skills like ["python", "aws", "docker"]


# ---------- Routes ----------

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Placement predictor is running"}


@app.post("/analyze_resume")
def analyze_resume(resume: ResumeInput):
    """
    Analyze a resume:
    1. Extract skills from the raw text.
    2. Use the trained model to predict placement category/label.
    """
    # 1. Extract skills from resume text
    skills = extract_skills(resume.text)

    # 2. Get prediction from the loaded model
    prediction = get_prediction(trained_model, skills)

    return {
        "status": "ok",
        "mode": "resume_analyzer",
        "skills": skills,
        "prediction": prediction,
    }


@app.post("/predict_from_skills")
def predict_from_skills(data: SkillsInput):
    """
    Predict placement category directly from a list of skills.
    Useful if the user already knows their skills.
    """
    prediction = get_prediction(trained_model, data.skills)

    return {
        "status": "ok",
        "mode": "skills_only",
        "skills": data.skills,
        "prediction": prediction,
    }


# ---------- Local run (optional, not used by Render) ----------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
