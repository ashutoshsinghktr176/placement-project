import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# --- 1. Train the Model ---
def train_model():
    print("Preparing data and training model...")
    
    # NOTE: REPLACE THIS DUMMY DATA with the actual loading and preprocessing 
    # of your Resume Analyzer dataset (e.g., loading from a CSV, feature scaling, etc.)
    
    # Dummy Data for demonstration: [CGPA, IQ, Projects]
    X = [[7.5, 100, 1], [8.2, 120, 2], [6.0, 90, 0], [9.0, 130, 3], [5.5, 95, 1], [8.8, 125, 2]]
    y = [0, 1, 0, 1, 0, 1]              # Target (0=Not Placed, 1=Placed)

    # Initialize and train the classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # This correctly returns the savable model object itself.
    return model

# --- 2. Save the Model ---
if __name__ == "__main__":
    trained_model = train_model()
    
    # Saving the model object using pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)

    print("✅ Success! 'model.pkl' has been generated.")

    # Verification: Try loading it back to confirm
    with open('model.pkl', 'rb') as f:
        test_model = pickle.load(f)
    print(f"Model successfully saved and test loaded: {type(test_model)}")
    print("---")