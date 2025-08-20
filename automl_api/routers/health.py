from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")
async def ping():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"status": "ok", "message": "pong"}