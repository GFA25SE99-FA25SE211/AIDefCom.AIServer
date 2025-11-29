"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import Config
from api.routers import speech_router, voice_router, question_router
from core.health import get_comprehensive_health
from core.metrics import websocket_connections
from core.health import get_comprehensive_health
from core.metrics import websocket_connections

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

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

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    from api.dependencies import get_speech_service, get_voice_service, get_question_service
    
    # Force initialize services and load models
    _ = get_speech_service()
    _ = get_voice_service()
    
    # Preload QuestionService semantic model (avoid timeout on first use)
    question_service = get_question_service()
    _ = question_service.semantic_model  # Trigger lazy load
    print("✅ Models loaded successfully (including semantic model)")
    
    # Background workers removed per configuration
    print("ℹ️ Background workers disabled")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Background workers removed per configuration
    print("ℹ️ No background workers to stop")


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
    """Comprehensive health check endpoint."""
    from api.dependencies import (
        get_redis_service,
        get_sql_server_repository,
        get_azure_blob_repository,
        get_azure_speech_repository
    )
    
    try:
        redis_service = get_redis_service()
        sql_repo = get_sql_server_repository()
        blob_repo = get_azure_blob_repository()
        speech_repo = get_azure_speech_repository()
        
        health_status = await get_comprehensive_health(
            redis_service=redis_service,
            sql_repo=sql_repo,
            blob_repo=blob_repo,
            speech_repo=speech_repo
        )
        
        # Return 503 if unhealthy (for load balancer/k8s)
        status_code = 200 if health_status["status"] == "healthy" else 503
        
        return Response(
            content=str(health_status),
            status_code=status_code,
            media_type="application/json"
        )
    except Exception as e:
        return Response(
            content=str({"status": "unhealthy", "error": str(e)}),
            status_code=503,
            media_type="application/json"
        )


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
