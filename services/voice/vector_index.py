"""Vector Index using FAISS for fast similarity search.

Replaces O(N) linear search with O(log N) approximate nearest neighbor search.
Supports both FAISS (recommended) and fallback to numpy for systems without FAISS.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS
_FAISS_AVAILABLE = False
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    pass  # Will use numpy fallback


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    index: int
    distance: float
    similarity: float  # Cosine similarity (1 - distance for L2, or direct for IP)
    metadata: Optional[Dict[str, Any]] = None


class VectorIndex(ABC):
    """Abstract base class for vector indices."""
    
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: Optional[List[str]] = None) -> None:
        """Add vectors to the index."""
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search for k nearest neighbors."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Clear the index."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Return number of vectors in index."""
        pass


class FAISSIndex(VectorIndex):
    """FAISS-based vector index for fast similarity search.
    
    Uses Inner Product (IP) similarity which is equivalent to cosine similarity
    when vectors are L2-normalized.
    
    Complexity:
    - Add: O(N) 
    - Search: O(log N) for IVF, O(N) for flat (but SIMD optimized)
    
    For small datasets (< 10K vectors), flat index is faster.
    For larger datasets, consider IVF or HNSW.
    """
    
    def __init__(
        self,
        dim: int,
        use_gpu: bool = False,
        index_type: str = "flat",  # "flat", "ivf", "hnsw"
        nlist: int = 100,  # Number of clusters for IVF
        nprobe: int = 10,  # Number of clusters to search
    ):
        """Initialize FAISS index.
        
        Args:
            dim: Vector dimension
            use_gpu: Whether to use GPU (requires faiss-gpu)
            index_type: Type of index ("flat", "ivf", "hnsw")
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search for IVF
        """
        if not _FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.dim = dim
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        # ID mapping (FAISS uses integer IDs internally)
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._next_idx = 0
        
        # Create index
        self._index = self._create_index()
    
    def _create_index(self) -> "faiss.Index":
        """Create FAISS index based on configuration."""
        if self.index_type == "flat":
            # Flat index - exact search, good for < 10K vectors
            index = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "ivf":
            # IVF index - approximate search, good for large datasets
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = self.nprobe
        elif self.index_type == "hnsw":
            # HNSW index - fast approximate search
            index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # GPU support
        if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
        
        return index
    
    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to the index.
        
        Args:
            vectors: Numpy array of shape (N, dim) - MUST be L2-normalized!
            ids: Optional string IDs for each vector
            metadata: Optional metadata for each vector
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        n_vectors = vectors.shape[0]
        
        # Ensure float32 and contiguous
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        # L2 normalize if not already (required for IP = cosine)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        vectors = vectors / norms
        
        # Train IVF index if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            if n_vectors >= self.nlist:
                self._index.train(vectors)
            else:
                logger.warning(f"Not enough vectors to train IVF ({n_vectors} < {self.nlist})")
                return
        
        # Map IDs
        if ids is None:
            ids = [str(i) for i in range(self._next_idx, self._next_idx + n_vectors)]
        
        for i, id_ in enumerate(ids):
            idx = self._next_idx + i
            self._id_to_idx[id_] = idx
            self._idx_to_id[idx] = id_
            if metadata and i < len(metadata):
                self._metadata[idx] = metadata[i]
        
        self._next_idx += n_vectors
        
        # Add to index
        self._index.add(vectors)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search for k nearest neighbors.
        
        Args:
            query: Query vector (will be L2-normalized)
            k: Number of neighbors to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        if self._index.ntotal == 0:
            return []
        
        # Prepare query
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = np.ascontiguousarray(query.astype(np.float32))
        
        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm
        
        # Limit k to index size
        k = min(k, self._index.ntotal)
        
        # Search
        similarities, indices = self._index.search(query, k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            if sim < threshold:
                continue
            
            results.append(SearchResult(
                index=int(idx),
                distance=1.0 - float(sim),  # Convert IP to distance
                similarity=float(sim),
                metadata=self._metadata.get(int(idx)),
            ))
        
        return results
    
    def get_id(self, idx: int) -> Optional[str]:
        """Get string ID for internal index."""
        return self._idx_to_id.get(idx)
    
    def get_metadata(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get metadata for internal index."""
        return self._metadata.get(idx)
    
    def reset(self) -> None:
        """Clear the index."""
        self._index.reset()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._metadata.clear()
        self._next_idx = 0
        
        # Recreate index (IVF needs retraining)
        self._index = self._create_index()
    
    def size(self) -> int:
        """Return number of vectors in index."""
        return self._index.ntotal


class NumpyFallbackIndex(VectorIndex):
    """Numpy-based fallback when FAISS is not available.
    
    Uses vectorized numpy operations for similarity search.
    Complexity: O(N) but optimized with BLAS/SIMD.
    """
    
    def __init__(self, dim: int):
        """Initialize numpy index.
        
        Args:
            dim: Vector dimension
        """
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._metadata: List[Optional[Dict[str, Any]]] = []
    
    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to the index."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        n_vectors = vectors.shape[0]
        
        # Normalize
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        vectors = vectors / norms
        
        # Generate IDs
        if ids is None:
            start = len(self._ids)
            ids = [str(i) for i in range(start, start + n_vectors)]
        
        # Store
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        
        self._ids.extend(ids)
        
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([None] * n_vectors)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search using matrix multiplication."""
        if self._vectors is None or len(self._vectors) == 0:
            return []
        
        # Prepare query
        query = query.astype(np.float32).reshape(-1)
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm
        
        # Compute all similarities at once (matrix-vector multiply)
        similarities = self._vectors @ query
        
        # Get top-k
        k = min(k, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < threshold:
                continue
            results.append(SearchResult(
                index=int(idx),
                distance=1.0 - sim,
                similarity=sim,
                metadata=self._metadata[idx] if idx < len(self._metadata) else None,
            ))
        
        return results
    
    def get_id(self, idx: int) -> Optional[str]:
        """Get string ID for index."""
        if 0 <= idx < len(self._ids):
            return self._ids[idx]
        return None
    
    def reset(self) -> None:
        """Clear the index."""
        self._vectors = None
        self._ids.clear()
        self._metadata.clear()
    
    def size(self) -> int:
        """Return number of vectors."""
        return len(self._ids)


def create_vector_index(
    dim: int,
    prefer_faiss: bool = True,
    **kwargs,
) -> VectorIndex:
    """Factory function to create appropriate vector index.
    
    Args:
        dim: Vector dimension
        prefer_faiss: Try FAISS first, fallback to numpy
        **kwargs: Additional arguments for FAISSIndex
        
    Returns:
        VectorIndex instance
    """
    if prefer_faiss and _FAISS_AVAILABLE:
        try:
            return FAISSIndex(dim, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create FAISS index: {e}")
    
    return NumpyFallbackIndex(dim)


class SessionVectorIndex:
    """Per-session vector index for question deduplication.
    
    Manages vector indices per session with automatic cleanup.
    """
    
    def __init__(self, dim: int, ttl_seconds: int = 7200):
        """Initialize session index manager.
        
        Args:
            dim: Embedding dimension
            ttl_seconds: Time-to-live for session indices
        """
        self.dim = dim
        self.ttl_seconds = ttl_seconds
        self._indices: Dict[str, VectorIndex] = {}
        self._questions: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_or_create_index(self, session_id: str) -> VectorIndex:
        """Get or create index for session."""
        if session_id not in self._indices:
            self._indices[session_id] = create_vector_index(self.dim)
            self._questions[session_id] = []
        return self._indices[session_id]
    
    def add_question(
        self,
        session_id: str,
        embedding: np.ndarray,
        question: Dict[str, Any],
    ) -> None:
        """Add question embedding to session index."""
        index = self.get_or_create_index(session_id)
        question_id = str(len(self._questions.get(session_id, [])))
        
        index.add(
            embedding.reshape(1, -1),
            ids=[question_id],
            metadata=[question],
        )
        
        self._questions.setdefault(session_id, []).append(question)
    
    def search_similar(
        self,
        session_id: str,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.6,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar questions in session.
        
        Returns:
            List of (question, similarity) tuples
        """
        if session_id not in self._indices:
            return []
        
        index = self._indices[session_id]
        results = index.search(query_embedding, k=k, threshold=threshold)
        
        questions = self._questions.get(session_id, [])
        output = []
        
        for result in results:
            if result.index < len(questions):
                output.append((questions[result.index], result.similarity))
        
        return output
    
    def rebuild_from_questions(
        self,
        session_id: str,
        questions: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Rebuild index from existing questions (e.g., from Redis).
        
        Args:
            session_id: Session ID
            questions: List of question dicts
            embeddings: Precomputed embeddings matrix (N, dim)
        """
        # Reset existing index
        if session_id in self._indices:
            self._indices[session_id].reset()
        
        index = self.get_or_create_index(session_id)
        self._questions[session_id] = list(questions)
        
        if len(questions) > 0 and embeddings is not None:
            ids = [str(i) for i in range(len(questions))]
            index.add(embeddings, ids=ids, metadata=questions)
    
    def clear_session(self, session_id: str) -> None:
        """Clear index for session."""
        if session_id in self._indices:
            self._indices[session_id].reset()
            del self._indices[session_id]
        if session_id in self._questions:
            del self._questions[session_id]
    
    def get_session_count(self, session_id: str) -> int:
        """Get number of questions in session."""
        return len(self._questions.get(session_id, []))
