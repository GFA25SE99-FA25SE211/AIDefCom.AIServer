"""Health check utilities for dependency verification."""

import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


async def check_redis_health(redis_service) -> Dict[str, Any]:
    """Check Redis connectivity and basic operations."""
    try:
        # Test connection with ping
        if hasattr(redis_service, 'client') and redis_service.client:
            await redis_service.client.ping()
            
            # Test basic operations
            test_key = "health:check:test"
            await redis_service.set(test_key, "ok", expire=5)
            value = await redis_service.get(test_key)
            await redis_service.delete(test_key)
            
            if value == "ok":
                return {
                    "status": "healthy",
                    "message": "Redis operations working",
                    "latency_ms": 0  # Could add timing if needed
                }
        
        return {
            "status": "degraded",
            "message": "Redis client not initialized"
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


async def check_sql_health(sql_repo) -> Dict[str, Any]:
    """Check SQL Server connectivity."""
    try:
        # Simple query to test connection
        connection = sql_repo._get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result and result[0] == 1:
            return {
                "status": "healthy",
                "message": "SQL Server connection working"
            }
        
        return {
            "status": "degraded",
            "message": "Unexpected query result"
        }
    except Exception as e:
        logger.error(f"SQL health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


async def check_blob_health(blob_repo) -> Dict[str, Any]:
    """Check Azure Blob Storage connectivity."""
    try:
        # Test container accessibility
        container_client = blob_repo.blob_service_client.get_container_client(
            blob_repo.container_name
        )
        
        # Check if container exists
        exists = container_client.exists()
        
        if exists:
            return {
                "status": "healthy",
                "message": f"Blob container '{blob_repo.container_name}' accessible"
            }
        
        return {
            "status": "unhealthy",
            "message": f"Container '{blob_repo.container_name}' not found"
        }
    except Exception as e:
        logger.error(f"Blob health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


async def check_speech_api_health(speech_repo) -> Dict[str, Any]:
    """Check Azure Speech Service configuration."""
    try:
        # Verify configuration
        if not speech_repo.speech_key or not speech_repo.region:
            return {
                "status": "unhealthy",
                "message": "Speech API credentials not configured"
            }
        
        # Could add actual API call test here if needed
        # For now, just verify config is present
        return {
            "status": "healthy",
            "message": "Speech API configured",
            "region": speech_repo.region
        }
    except Exception as e:
        logger.error(f"Speech API health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


async def get_comprehensive_health(
    redis_service=None,
    sql_repo=None,
    blob_repo=None,
    speech_repo=None
) -> Dict[str, Any]:
    """Get comprehensive health check for all dependencies."""
    
    checks = {}
    
    if redis_service:
        checks["redis"] = await check_redis_health(redis_service)
    
    if sql_repo:
        checks["sql_server"] = await check_sql_health(sql_repo)
    
    if blob_repo:
        checks["azure_blob"] = await check_blob_health(blob_repo)
    
    if speech_repo:
        checks["azure_speech"] = await check_speech_api_health(speech_repo)
    
    # Determine overall status
    all_healthy = all(
        check.get("status") == "healthy" 
        for check in checks.values()
    )
    
    any_unhealthy = any(
        check.get("status") == "unhealthy"
        for check in checks.values()
    )
    
    if all_healthy:
        overall_status = "healthy"
    elif any_unhealthy:
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checks": checks
    }
