"""Shared thread/process executors for CPU-bound operations.

This module provides centralized executor management for offloading
CPU-intensive tasks from the async event loop.

Usage:
    from core.executors import get_cpu_executor, run_cpu_bound

    # Option 1: Direct executor access
    executor = get_cpu_executor()
    result = await loop.run_in_executor(executor, heavy_function, arg1, arg2)

    # Option 2: Convenience wrapper
    result = await run_cpu_bound(heavy_function, arg1, arg2)
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

# Type hints for generic functions
P = ParamSpec('P')
T = TypeVar('T')

# Global executors (lazy initialized)
_CPU_EXECUTOR: ThreadPoolExecutor | None = None
_VOICE_EXECUTOR: ThreadPoolExecutor | None = None
_IO_EXECUTOR: ThreadPoolExecutor | None = None


def _get_optimal_workers(task_type: str = "cpu") -> int:
    """Calculate optimal worker count based on task type and CPU cores."""
    cpu_count = os.cpu_count() or 2
    
    if task_type == "cpu":
        # CPU-bound: use more workers to handle concurrent requests
        return min(8, max(4, cpu_count))
    elif task_type == "voice":
        # Voice identification: increase to handle multiple concurrent identifications
        return min(8, max(4, cpu_count))
    elif task_type == "io":
        # I/O-bound: can use more workers
        return min(16, max(8, cpu_count * 2))
    else:
        return max(4, cpu_count)


def get_cpu_executor() -> ThreadPoolExecutor:
    """Get or create shared CPU-bound executor.
    
    Used for: fuzzy matching, text processing, numpy operations, etc.
    """
    global _CPU_EXECUTOR
    if _CPU_EXECUTOR is None:
        max_workers = _get_optimal_workers("cpu")
        _CPU_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cpu_bound_"
        )
        logger.info(f"Created CPU executor with {max_workers} workers")
    return _CPU_EXECUTOR


def get_voice_executor() -> ThreadPoolExecutor:
    """Get or create dedicated executor for voice identification.
    
    Voice ID is very CPU-intensive (embedding extraction, similarity).
    Separate executor prevents voice tasks from blocking other CPU tasks.
    """
    global _VOICE_EXECUTOR
    if _VOICE_EXECUTOR is None:
        max_workers = _get_optimal_workers("voice")
        _VOICE_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="voice_id_"
        )
        logger.info(f"Created voice identification executor with {max_workers} workers")
    return _VOICE_EXECUTOR


def get_io_executor() -> ThreadPoolExecutor:
    """Get or create executor for blocking I/O operations.
    
    Used for: file I/O, synchronous HTTP calls, etc.
    """
    global _IO_EXECUTOR
    if _IO_EXECUTOR is None:
        max_workers = _get_optimal_workers("io")
        _IO_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="io_bound_"
        )
        logger.info(f"Created I/O executor with {max_workers} workers")
    return _IO_EXECUTOR


async def run_cpu_bound(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a CPU-bound function in the thread pool executor.
    
    This prevents blocking the async event loop with CPU-intensive operations.
    
    Args:
        func: The CPU-bound function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The function's return value
        
    Example:
        # Instead of blocking:
        result = heavy_computation(data)
        
        # Use:
        result = await run_cpu_bound(heavy_computation, data)
    """
    loop = asyncio.get_running_loop()
    executor = get_cpu_executor()
    
    if kwargs:
        # functools.partial handles kwargs
        func_with_kwargs = partial(func, *args, **kwargs)
        future = loop.run_in_executor(executor, func_with_kwargs)
    else:
        future = loop.run_in_executor(executor, func, *args)
    
    # Add timeout to prevent hanging
    try:
        return await asyncio.wait_for(future, timeout=30.0)
    except asyncio.TimeoutError:
        logger.error(f"CPU-bound task {func.__name__} timed out after 30s")
        raise


async def run_voice_bound(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a voice identification function in dedicated executor.
    
    Args:
        func: The voice processing function
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        The function's return value
    """
    loop = asyncio.get_running_loop()
    executor = get_voice_executor()
    
    if kwargs:
        func_with_kwargs = partial(func, *args, **kwargs)
        future = loop.run_in_executor(executor, func_with_kwargs)
    else:
        future = loop.run_in_executor(executor, func, *args)
    
    # Add timeout to prevent hanging
    try:
        return await asyncio.wait_for(future, timeout=15.0)
    except asyncio.TimeoutError:
        logger.error(f"Voice-bound task {func.__name__} timed out after 15s")
        raise


async def run_io_bound(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a blocking I/O function in the I/O executor.
    
    Args:
        func: The I/O-bound function
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        The function's return value
    """
    loop = asyncio.get_running_loop()
    executor = get_io_executor()
    
    if kwargs:
        func_with_kwargs = partial(func, *args, **kwargs)
        future = loop.run_in_executor(executor, func_with_kwargs)
    else:
        future = loop.run_in_executor(executor, func, *args)
    
    # Add timeout to prevent hanging
    try:
        return await asyncio.wait_for(future, timeout=30.0)
    except asyncio.TimeoutError:
        logger.error(f"I/O-bound task {func.__name__} timed out after 30s")
        raise


def shutdown_executors() -> None:
    """Shutdown all executors gracefully.
    
    Call this during application shutdown to clean up resources.
    """
    global _CPU_EXECUTOR, _VOICE_EXECUTOR, _IO_EXECUTOR
    
    for name, executor in [
        ("CPU", _CPU_EXECUTOR),
        ("Voice", _VOICE_EXECUTOR),
        ("I/O", _IO_EXECUTOR),
    ]:
        if executor is not None:
            logger.info(f"Shutting down {name} executor...")
            executor.shutdown(wait=True, cancel_futures=False)
    
    _CPU_EXECUTOR = None
    _VOICE_EXECUTOR = None
    _IO_EXECUTOR = None
    
    logger.info("All executors shut down")
