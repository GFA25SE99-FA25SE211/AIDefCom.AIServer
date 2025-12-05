"""Streaming upload utilities for memory-efficient file handling.

Provides utilities for handling large audio file uploads without loading 
the entire file into memory, preventing OOM errors with large files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Optional

import aiofiles
from aiofiles.tempfile import NamedTemporaryFile

if TYPE_CHECKING:
    from fastapi import UploadFile

logger = logging.getLogger(__name__)

# Default chunk size for streaming (64KB - balanced for network I/O)
DEFAULT_CHUNK_SIZE = 64 * 1024  # 64KB

# Threshold for using disk-based temp file vs in-memory
# Files smaller than this stay in memory (SpooledTemporaryFile behavior)
MEMORY_THRESHOLD = 1024 * 1024  # 1MB


class StreamingUploadError(Exception):
    """Raised when streaming upload operations fail."""
    pass


@asynccontextmanager
async def stream_upload_to_temp_file(
    upload_file: "UploadFile",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_size: Optional[int] = None,
    suffix: str = ".wav",
    delete_on_exit: bool = True,
) -> AsyncIterator[Path]:
    """Stream uploaded file to a temporary file, yielding the path.
    
    This prevents loading large audio files entirely into memory,
    which can cause OOM errors with files > 10MB.
    
    Usage:
        async with stream_upload_to_temp_file(audio_file) as temp_path:
            # Process audio from temp_path
            audio_data = temp_path.read_bytes()  # or stream from file
            
    Args:
        upload_file: FastAPI UploadFile object
        chunk_size: Size of chunks to read/write (default 64KB)
        max_size: Maximum allowed file size in bytes (None = no limit)
        suffix: File suffix for temp file (default .wav)
        delete_on_exit: Whether to delete temp file on exit (default True)
        
    Yields:
        Path to temporary file containing uploaded data
        
    Raises:
        StreamingUploadError: If upload exceeds max_size or I/O fails
    """
    temp_path: Optional[Path] = None
    total_bytes = 0
    
    try:
        # Create temp file in system temp directory
        # Using delete=False so we control cleanup
        fd, temp_path_str = tempfile.mkstemp(suffix=suffix)
        os.close(fd)  # Close the file descriptor, we'll use aiofiles
        temp_path = Path(temp_path_str)
        
        logger.debug(
            "Streaming upload started | filename=%s | temp_path=%s | max_size=%s",
            upload_file.filename, temp_path, max_size
        )
        
        # Stream chunks from upload to temp file
        async with aiofiles.open(temp_path, 'wb') as temp_file:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                    
                total_bytes += len(chunk)
                
                # Check size limit
                if max_size and total_bytes > max_size:
                    raise StreamingUploadError(
                        f"Upload exceeds maximum size of {max_size} bytes "
                        f"(received {total_bytes} bytes)"
                    )
                
                await temp_file.write(chunk)
        
        logger.info(
            "Streaming upload complete | filename=%s | size=%d bytes | temp_path=%s",
            upload_file.filename, total_bytes, temp_path
        )
        
        yield temp_path
        
    except StreamingUploadError:
        raise
    except Exception as e:
        logger.error(
            "Streaming upload failed | filename=%s | error=%s",
            upload_file.filename, str(e)
        )
        raise StreamingUploadError(f"Failed to stream upload: {e}") from e
        
    finally:
        # Cleanup temp file
        if delete_on_exit and temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug("Temp file cleaned up | path=%s", temp_path)
            except Exception as e:
                logger.warning(
                    "Failed to cleanup temp file | path=%s | error=%s",
                    temp_path, str(e)
                )


async def stream_upload_to_bytes(
    upload_file: "UploadFile",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_size: Optional[int] = None,
) -> bytes:
    """Stream uploaded file to bytes buffer with size limit check.
    
    For smaller files where you still want byte content but with
    streaming validation of size limit.
    
    Args:
        upload_file: FastAPI UploadFile object
        chunk_size: Size of chunks to read (default 64KB)
        max_size: Maximum allowed file size in bytes (None = no limit)
        
    Returns:
        Complete file content as bytes
        
    Raises:
        StreamingUploadError: If upload exceeds max_size
    """
    chunks = []
    total_bytes = 0
    
    try:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
                
            total_bytes += len(chunk)
            
            if max_size and total_bytes > max_size:
                raise StreamingUploadError(
                    f"Upload exceeds maximum size of {max_size} bytes "
                    f"(received {total_bytes} bytes)"
                )
            
            chunks.append(chunk)
        
        logger.debug(
            "Streamed upload to bytes | filename=%s | size=%d bytes",
            upload_file.filename, total_bytes
        )
        
        return b''.join(chunks)
        
    except StreamingUploadError:
        raise
    except Exception as e:
        logger.error(
            "Failed to stream upload to bytes | filename=%s | error=%s",
            upload_file.filename, str(e)
        )
        raise StreamingUploadError(f"Failed to stream upload: {e}") from e


class ChunkedAudioReader:
    """Memory-efficient audio file reader that yields chunks.
    
    Useful for processing large audio files without loading entirely into memory.
    
    Usage:
        reader = ChunkedAudioReader(Path("large_audio.wav"))
        async for chunk in reader:
            process_chunk(chunk)
    """
    
    def __init__(
        self, 
        file_path: Path, 
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> None:
        """Initialize chunked reader.
        
        Args:
            file_path: Path to audio file
            chunk_size: Size of chunks to yield (default 64KB)
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.total_bytes_read = 0
    
    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Async iterator that yields file chunks."""
        async with aiofiles.open(self.file_path, 'rb') as f:
            while True:
                chunk = await f.read(self.chunk_size)
                if not chunk:
                    break
                self.total_bytes_read += len(chunk)
                yield chunk
    
    def read_sync(self) -> bytes:
        """Read entire file synchronously (for backward compatibility).
        
        WARNING: This loads entire file into memory. Use async iterator
        for large files.
        """
        return self.file_path.read_bytes()
