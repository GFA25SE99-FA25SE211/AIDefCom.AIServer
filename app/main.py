"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Config
from api.routers import speech_router, voice_router, question_router


# Create FastAPI app with enhanced Swagger documentation
app = FastAPI(
    title=Config.APP_TITLE,
    description=Config.APP_DESCRIPTION,
    version=Config.APP_VERSION,
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc",  # ReDoc at /redoc
    openapi_url="/openapi.json",  # OpenAPI schema
    openapi_tags=[
        {
            "name": "voice-auth",
            "description": "🔐 Voice authentication endpoints: enroll, verify, identify speakers",
        },
        {
            "name": "Questions",
            "description": "❓ Question duplicate detection and management",
        },
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=Config.CORS_ALLOW_CREDENTIALS,
    allow_methods=Config.CORS_ALLOW_METHODS,
    allow_headers=Config.CORS_ALLOW_HEADERS,
)


@app.on_event("startup")
async def startup_event():
    """Preload models and services on startup."""
    print("🚀 Preloading models and services...")
    from api.dependencies import get_speech_service, get_voice_service
    # Force initialize services and load models
    _ = get_speech_service()
    _ = get_voice_service()
    print("✅ Models loaded successfully")


# Include routers
app.include_router(speech_router.router, include_in_schema=False)  # Hide from Swagger (WebSocket)
app.include_router(voice_router.router)  # Show in Swagger
app.include_router(question_router.router)  # Show in Swagger


@app.get("/", include_in_schema=False)  # Hide root endpoint from Swagger
async def root():
    """Root endpoint."""
    return {
        "service": Config.APP_TITLE,
        "version": Config.APP_VERSION,
        "status": "running",
    }


@app.get("/health", include_in_schema=False)  # Hide health check from Swagger
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
