from fastapi import FastAPI
from placement_model import load_model, get_prediction
 # Assuming placement_model.py is in the same folder

# Global variable to hold the loaded model
# The app will crash if the model fails to load, preventing a bad deploy.
try:
    trained_model = load_model()
except FileNotFoundError as e:
    # This will cause the Render deployment to stop, highlighting the missing file
    raise RuntimeError("Application failed to start because model.pkl is missing.") from e

# Initialize the FastAPI app
app = FastAPI(
    title="Resume Analyzer & Placement Predictor",
    version="1.0.0"
)

# --- Define Data Schema (Pydantic model) ---
from pydantic import BaseModel
class PredictionRequest(BaseModel):
    cgpa: float
    iq: float
    projects: int

# --- Define Endpoints ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Placement predictor is running"}

@app.post("/predict")
def predict_placement(request: PredictionRequest):
    """
    Predicts placement based on input features.
    """
    result = get_prediction(
        trained_model, 
        request.cgpa, 
        request.iq, 
        request.projects
    )
    return {"placement_status": result}

# ---