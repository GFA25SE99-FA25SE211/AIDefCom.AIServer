"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Config
from api.routers import speech_router, voice_router


# Create FastAPI app
app = FastAPI(
    title=Config.APP_TITLE,
    description=Config.APP_DESCRIPTION,
    version=Config.APP_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=Config.CORS_ALLOW_CREDENTIALS,
    allow_methods=Config.CORS_ALLOW_METHODS,
    allow_headers=Config.CORS_ALLOW_HEADERS,
)

# Include routers
app.include_router(speech_router.router)
app.include_router(voice_router.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": Config.APP_TITLE,
        "version": Config.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
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
