"""FastAPI application entry point - delegates to app.main."""

from app.main import app

__all__ = ["app"]


@app.get("/")
async def root():
    return {
        "version": "2.3.3",
        "service": "Speech Services API",
        "endpoints": {
            "stt": "/ws/stt",
            "voice_auth": {
                "enroll": "POST /voice/users/{user_id}/enroll",
                "identify": "POST /voice/identify",
                "verify": "POST /voice/users/{user_id}/verify",
            },
            "health": "/health",
            "docs": "/docs"
        },
        "features": ["streaming", "noise-filtering", "vietnamese-optimized", "voice-authentication"],
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

