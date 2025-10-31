"""FastAPI application entry point - delegates to app.main."""

from app.main import app

__all__ = ["app"]


@app.get("/")
async def root():
    return {
        "version": "2.3.2",
        "service": "Speech Services API",
        "endpoints": {
            "stt": "/ws/stt",
            "voice_auth": {
                "create_or_get": "POST /voice/profile",
                "create_profile": "POST /voice/profile/create",
                "enroll": "POST /voice/enroll",
                "identify": "POST /voice/identify",
                "verify": "POST /voice/verify",
                "profile_info": "GET /voice/profile/{user_id}",
                "delete_profile": "DELETE /voice/profile/{user_id}",
                "list_profiles": "GET /voice/profiles",
            },
        },
        "features": ["streaming", "noise-filtering", "vietnamese-optimized", "voice-authentication"],
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
