"""SpeechBrain model repository - Manages speaker embedding models."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from core.exceptions import ModelLoadError

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None

# Environment setup for SpeechBrain
os.environ.setdefault("SPEECHBRAIN_DOWNLOAD_BACKEND", "local")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Using SYMLINK strategy on Windows.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*", category=FutureWarning)

logger = logging.getLogger(__name__)

try:
    from speechbrain.inference import EncoderClassifier
except ImportError:  # pragma: no cover
    from speechbrain.pretrained import EncoderClassifier


@dataclass
class EmbeddingResult:
    """Result from embedding extraction."""
    embedding: np.ndarray
    rms: float
    voiced_ratio: float
    snr_db: float
    clipping_ratio: float
    duration_sec: float
    quality_ok: bool
    quality_reason: str | None
    model_tag: str


class SpeechBrainModelRepository:
    """Repository for SpeechBrain speaker recognition models."""
    
    def __init__(
        self,
        model_source: str | None = None,
        models_dir: str | None = None,
        sample_rate: int = 16000,
    ) -> None:
        """
        Initialize SpeechBrain model repository.
        
        Args:
            model_source: HuggingFace model path or None to use environment variable
            models_dir: Directory to store models
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        
        # Determine model source
        embedder_env = os.getenv("EMBEDDER", "ecapa").strip().lower()
        if model_source is not None:
            self.model_source = model_source
            self.model_tag = "custom"
        elif embedder_env == "xvector":
            self.model_source = "speechbrain/spkrec-xvect-voxceleb"
            self.model_tag = "xvector"
        else:
            self.model_source = "speechbrain/spkrec-ecapa-voxceleb"
            self.model_tag = "ecapa"
        
        # Setup model directory
        model_folder = "xvector" if "xvect" in self.model_source else "ecapa_tdnn"
        base_dir = Path(models_dir) if models_dir else Path(".models")
        self.model_dir = base_dir / model_folder
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model state
        self._classifier: EncoderClassifier | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 192  # default for ECAPA-TDNN
        self._warmed_up = False
        
        # Load and warmup model
        self._load_model()
        self._warmup_model()
    
    def _load_model(self) -> None:
        """Load SpeechBrain model from HuggingFace or local cache."""
        if self._classifier is not None:
            return
        
        run_opts = {"device": str(self.device)}
        
        try:
            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_source,
                run_opts=run_opts,
                savedir=str(self.model_dir),
            )
        except OSError as err:
            # Handle Windows symlink permission error
            if getattr(err, "winerror", None) == 1314:
                logger.info("Fallback to manual model download (missing symlink privileges)")
                local_path = self._prepare_local_model()
                self._classifier = EncoderClassifier.from_hparams(
                    source=str(local_path),
                    run_opts=run_opts,
                )
            else:
                raise ModelLoadError(f"Failed to load model: {err}") from err
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e
        
        # Get embedding dimension
        emb_size = getattr(self._classifier.hparams, "emb_dim", None) or \
                   getattr(self._classifier.hparams, "embedding_dim", None)
        if emb_size is not None:
            self.embedding_dim = int(emb_size)
    
    def _prepare_local_model(self) -> Path:
        """Download model manually using HuggingFace Hub."""
        local_dir = self.model_dir / "local_copy"
        local_dir.mkdir(parents=True, exist_ok=True)
        
        if snapshot_download is None:
            raise ModelLoadError("huggingface_hub is required for manual model download")
        
        snapshot_download(
            repo_id=self.model_source,
            local_dir=str(local_dir),
            resume_download=True,
        )
        return local_dir
    
    def _warmup_model(self) -> None:
        """Warmup model with dummy input for faster first inference."""
        if self._warmed_up or self._classifier is None:
            return
        
        try:
            dummy_len = int(1.6 * self.sample_rate)
            warm_tensor = torch.randn(1, dummy_len, device=self.device)
            with torch.no_grad():
                _ = self._classifier.encode_batch(warm_tensor)
            self._warmed_up = True
            logger.info(f"Model warmed up successfully ({self.model_tag})")
        except Exception as exc:  # pragma: no cover
            logger.warning("Warmup skipped: %s", exc)
    
    def extract_embedding(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Extract speaker embedding from audio signal.
        
        Args:
            audio_signal: Normalized audio signal (float32, [-1, 1])
        
        Returns:
            Normalized embedding vector
        
        Raises:
            ModelLoadError: If model is not loaded
        """
        if self._classifier is None:
            raise ModelLoadError("Model not loaded")
        
        # Convert to tensor
        tensor = torch.from_numpy(audio_signal).float().unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            emb = self._classifier.encode_batch(tensor)
        
        # Convert to numpy
        emb_np = emb.squeeze().cpu().numpy().astype(np.float32)
        emb_np = np.reshape(emb_np, (-1,)).astype(np.float32)
        
        # Normalize
        norm = float(np.linalg.norm(emb_np) + 1e-8)
        if not np.isfinite(norm) or norm < 1e-6:
            raise ModelLoadError("Embedding norm too small or invalid")
        
        emb_np /= norm
        
        # Ensure correct dimension
        if emb_np.size != self.embedding_dim:
            emb_fixed = np.zeros(self.embedding_dim, dtype=np.float32)
            copy_len = min(self.embedding_dim, emb_np.size)
            emb_fixed[:copy_len] = emb_np[:copy_len]
            emb_np = emb_fixed
        
        return emb_np
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embedding vector."""
        return self.embedding_dim
    
    def get_model_tag(self) -> str:
        """Get the model tag (ecapa, xvector, custom)."""
        return self.model_tag
