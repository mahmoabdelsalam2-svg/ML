import os
import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# --- ADD THIS DEBUG BLOCK ---
print("--- Checking for GOOGLE_API_KEY ---", file=sys.stderr)
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    print(f"SUCCESS: API Key found starting with '{api_key[:4]}...'", file=sys.stderr)
else:
    print("!!! FAILURE: GOOGLE_API_KEY environment variable not found by the script.", file=sys.stderr)
print("---------------------------------", file=sys.stderr)
# --- END OF DEBUG BLOCK ---

from .tabular_router import job_registry # Import to access tabular results

router = APIRouter()

# Configure the Gemini API
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except (KeyError, Exception) as e:
    print(f"!!! CRITICAL ERROR: Could not configure Google AI. The API key might be missing or invalid. Error: {e}", file=sys.stderr)
    model = None # Set model to None if configuration fails

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat_with_bot(request: ChatRequest):
    """Handles conversational chat using Gemini."""
    if not model:
        raise HTTPException(status_code=500, detail="Gemini AI model is not configured. Check the server logs for API key errors.")

    prompt = f"""
    You are a helpful assistant for an AutoML platform.
    A user sent the following message: '{request.message}'.
    Provide a concise and helpful response related to machine learning or the platform's capabilities.
    """
    try:
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interpret-results/{job_id}")
async def interpret_ml_results(job_id: str):
    """Uses Gemini to explain tabular model results in simple terms."""
    if not model:
        raise HTTPException(status_code=500, detail="Gemini AI model is not configured. Check the server logs for API key errors.")

    if job_id not in job_registry or job_registry[job_id].get("status") != "complete":
        raise HTTPException(status_code=404, detail="Results not available for this job ID.")

    results = job_registry[job_id]["results"]
    prompt_context = results.get("best_model", results) # Handle both AutoML and single model results

    prompt = f"""
    As a data science expert explaining results to a non-technical manager,
    please interpret the following machine learning model output.

    - Model Used: {prompt_context.get('model_name')}
    - Problem Type: {prompt_context.get('problem_type')}
    - Key Metrics: {prompt_context.get('metrics')}

    Explain what these metrics mean in simple terms and summarize the model's performance.
    Keep the explanation brief (2-3 sentences).
    """
    try:
        response = model.generate_content(prompt)
        return {"interpretation": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))