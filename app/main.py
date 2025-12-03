"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import gc
import warnings
from contextlib import asynccontextmanager

# Suppress torchcodec warnings before any pyannote imports
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*libtorchcodec.*", category=UserWarning)

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

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global warmup state for ACA health probes
_warmup_complete = asyncio.Event()
_warmup_status = {"stage": "pending", "progress": 0, "error": None}


async def background_warmup():
    """Background task to load models gradually without blocking startup.
    
    This prevents ACA from killing the container during heavy model loading.
    Health check returns 200 immediately, readiness returns 200 after warmup.
    
    IMPORTANT: All sync blocking I/O (model loading, blob/sql access) runs in 
    thread pool to avoid blocking the event loop.
    """
    global _warmup_status
    import time
    import concurrent.futures
    
    loop = asyncio.get_running_loop()
    
    # Thread pool for blocking I/O operations
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="warmup_")
    
    def _run_sync(fn):
        """Run sync function in thread pool with timeout."""
        return loop.run_in_executor(executor, fn)
    
    try:
        # Stage 1: Wait for app to be responsive (health check can pass)
        _warmup_status = {"stage": "waiting", "progress": 10, "error": None}
        await asyncio.sleep(0.5)  # Minimal wait
        
        # Stage 1.5: Pre-connect Redis (avoid delay on first request)
        _warmup_status = {"stage": "connecting_redis", "progress": 15, "error": None}
        print("🔌 [Warmup] Connecting to Redis...")
        try:
            from api.dependencies import get_redis_service
            redis_svc = get_redis_service()
            await asyncio.wait_for(redis_svc._ensure_connection(), timeout=10.0)
            if redis_svc.client:
                print("✅ [Warmup] Redis connected")
            else:
                print("⚠️ [Warmup] Redis not available, continuing without cache")
        except Exception as e:
            print(f"⚠️ [Warmup] Redis warmup skipped: {e}")
        
        # Stage 2: Load embedding model (Pyannote/WeSpeaker) - IN THREAD POOL
        _warmup_status = {"stage": "loading_voice_model", "progress": 20, "error": None}
        print("🔄 [Warmup] Loading voice model...")
        start = time.time()
        
        def _load_voice_model():
            from api.dependencies import get_embedding_model_repository
            return get_embedding_model_repository()
        
        await asyncio.wait_for(_run_sync(_load_voice_model), timeout=120.0)
        gc.collect()
        
        print(f"✅ [Warmup] Voice model loaded in {time.time() - start:.1f}s")
        
        # Stage 3: Load voice service (uses model from stage 2) - IN THREAD POOL
        _warmup_status = {"stage": "loading_voice_service", "progress": 50, "error": None}
        print("🔄 [Warmup] Initializing voice service...")
        
        def _load_voice_service():
            from api.dependencies import get_voice_service
            return get_voice_service()
        
        voice_service = await asyncio.wait_for(_run_sync(_load_voice_service), timeout=30.0)
        print("✅ [Warmup] Voice service initialized")
        
        # Stage 4: Preload voice profiles - IN THREAD POOL (has blocking blob I/O)
        _warmup_status = {"stage": "preloading_profiles", "progress": 60, "error": None}
        print("🔄 [Warmup] Preloading voice profiles...")
        try:
            def _preload_profiles():
                return voice_service.preload_enrolled_profiles()
            
            profile_count = await asyncio.wait_for(_run_sync(_preload_profiles), timeout=60.0)
            print(f"✅ [Warmup] Preloaded {profile_count} voice profiles")
        except asyncio.TimeoutError:
            print("⚠️ [Warmup] Profile preload timeout (60s), skipping")
        except Exception as e:
            print(f"⚠️ [Warmup] Profile preload skipped: {e}")
        
        gc.collect()
        
        # Stage 5: Load semantic model for question service (~500MB) - IN THREAD POOL
        _warmup_status = {"stage": "loading_semantic_model", "progress": 75, "error": None}
        print("🔄 [Warmup] Loading semantic model...")
        start = time.time()
        
        def _load_semantic_model():
            from api.dependencies import get_question_service
            question_service = get_question_service()
            _ = question_service.semantic_model  # Trigger lazy load
            return question_service
        
        await asyncio.wait_for(_run_sync(_load_semantic_model), timeout=120.0)
        
        gc.collect()
        print(f"✅ [Warmup] Semantic model loaded in {time.time() - start:.1f}s")
        
        # Stage 6: Initialize speech service - IN THREAD POOL
        _warmup_status = {"stage": "loading_speech_service", "progress": 90, "error": None}
        print("🔄 [Warmup] Initializing speech service...")
        
        def _load_speech_service():
            from api.dependencies import get_speech_service
            return get_speech_service()
        
        await asyncio.wait_for(_run_sync(_load_speech_service), timeout=30.0)
        print("✅ [Warmup] Speech service initialized")
        
        # Done!
        _warmup_status = {"stage": "complete", "progress": 100, "error": None}
        _warmup_complete.set()
        print("🎉 [Warmup] All services ready!")
        
    except asyncio.TimeoutError as e:
        _warmup_status = {"stage": "timeout", "progress": _warmup_status.get("progress", 0), "error": str(e)}
        print(f"⚠️ [Warmup] Timeout during warmup, continuing with partial initialization")
        _warmup_complete.set()
    except Exception as e:
        _warmup_status = {"stage": "error", "progress": 0, "error": str(e)}
        print(f"❌ [Warmup] Failed: {e}")
        import traceback
        traceback.print_exc()
        # Still mark complete so requests don't hang forever
        _warmup_complete.set()
    finally:
        executor.shutdown(wait=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager with non-blocking warmup for ACA compatibility."""
    print("🚀 Starting AIDefCom AI Service...")
    
    # Start background warmup (non-blocking - allows health check to pass immediately)
    warmup_task = asyncio.create_task(background_warmup())
    
    yield
    
    # Cleanup
    warmup_task.cancel()
    print("👋 Shutting down...")


# Create FastAPI app with lifespan manager
app = FastAPI(
    title=Config.APP_TITLE,
    description=Config.APP_DESCRIPTION,
    version=Config.APP_VERSION,
    lifespan=lifespan,
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

# Include routers
app.include_router(speech_router.router, include_in_schema=False)  # Hide from Swagger (WebSocket)
app.include_router(voice_router.router)  # Show in Swagger
app.include_router(question_router.router)  # Show in Swagger


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - always fast, shows warmup status."""
    return {
        "service": Config.APP_TITLE,
        "version": Config.APP_VERSION,
        "status": "running",
        "warmup": _warmup_status,
    }


@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check - returns 200 immediately for ACA liveness probes.
    
    This endpoint should always return 200 quickly so ACA doesn't kill the container.
    Use /ready for readiness checks that require warmup completion.
    """
    import json
    
    # Basic health: app is running (don't block on warmup)
    basic_health = {
        "status": "healthy",
        "warmup": _warmup_status,
    }
    
    # If warmup complete, do comprehensive health check
    if _warmup_complete.is_set():
        try:
            from api.dependencies import (
                get_redis_service,
                get_sql_server_repository,
                get_azure_blob_repository,
                get_azure_speech_repository
            )
            
            health_status = await get_comprehensive_health(
                redis_service=get_redis_service(),
                sql_repo=get_sql_server_repository(),
                blob_repo=get_azure_blob_repository(),
                speech_repo=get_azure_speech_repository()
            )
            health_status["warmup"] = _warmup_status
            
            status_code = 200 if health_status["status"] == "healthy" else 503
            return Response(
                content=json.dumps(health_status),
                status_code=status_code,
                media_type="application/json"
            )
        except Exception as e:
            basic_health["error"] = str(e)
    
    return Response(
        content=json.dumps(basic_health),
        status_code=200,  # Always 200 for liveness
        media_type="application/json"
    )


@app.get("/ready", include_in_schema=False)
async def readiness_check():
    """Readiness probe - returns 200 only when fully warmed up.
    
    Use this endpoint for ACA readiness probes to ensure traffic is only
    routed to the container after all models are loaded.
    """
    import json
    
    if _warmup_complete.is_set() and _warmup_status.get("stage") == "complete":
        return Response(
            content=json.dumps({"ready": True, "warmup": _warmup_status}),
            status_code=200,
            media_type="application/json"
        )
    
    return Response(
        content=json.dumps({"ready": False, "warmup": _warmup_status}),
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
