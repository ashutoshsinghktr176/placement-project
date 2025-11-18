# Placement Prediction & Resume Analyzer (Render Deployment)

## Features
- Resume upload (PDF/DOCX)
- Extracts skills
- ATS scoring
- Company recommendations
- Placement tier prediction (A/B/C)
- FastAPI backend
- Deployable on Render

## API Endpoints
### Upload Resume
POST /analyze  
form-data → file: your_resume.pdf

### Live Test
GET /

## Deployment
1. Push code to GitHub
2. Go to https://dashboard.render.com
3. New → Web Service
4. Build Command:
   pip install -r requirements.txt
5. Start Command:
   uvicorn main:app --host 0.0.0.0 --port 10000
6. Deploy 🎉