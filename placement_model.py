import pickle
import os

# Function to safely load the pre-trained model
def load_model():
    model_path = "model.pkl"
    
    # On Render, the file MUST exist, otherwise it's a critical crash
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Deployment failed: Required model file '{model_path}' not found. "
            "Ensure you commit model.pkl to your repository."
        )

    # Load the pre-trained model object
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model

# Function to use the loaded model (example)
def get_prediction(model, cgpa, iq, projects):
    # Ensure features are provided in the correct 2D format for the model
    features = [[cgpa, iq, projects]]
    prediction = model.predict(features)
    
    # Convert prediction to a readable format
    result = "Placed" if prediction[0] == 1 else "Not Placed"
    return result