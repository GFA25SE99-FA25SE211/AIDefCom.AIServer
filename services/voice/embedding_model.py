"""Speaker Embedding Model - Using Pyannote/WeSpeaker."""

from __future__ import annotations

import logging
import os
import warnings

warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*libtorchcodec.*", category=UserWarning)

import numpy as np
import torch

from core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


# Patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


class EmbeddingModel:
    """Speaker embedding model using Pyannote/WeSpeaker.
    
    Uses pyannote/wespeaker-voxceleb-resnet34-LM model:
    - 256-dimensional embeddings
    - EER ~1% on VoxCeleb
    - Good multilingual support
    """
    
    MODEL_NAME = "pyannote/wespeaker-voxceleb-resnet34-LM"
    
    def __init__(self, sample_rate: int = 16000) -> None:
        """Initialize embedding model.
        
        Args:
            sample_rate: Audio sample rate (must be 16000)
        """
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 256
        self.model_tag = "wespeaker"
        
        self._model = None
        self._inference = None
        self._warmed_up = False
        
        self._hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not self._hf_token:
            raise ModelLoadError("HF_TOKEN environment variable is required")
        
        self._load_model()
        self._warmup_model()
    
    def _load_model(self) -> None:
        """Load the Pyannote embedding model."""
        if self._model is not None:
            return
        
        try:
            from pyannote.audio import Model, Inference
            
            logger.info(f"Loading embedding model: {self.MODEL_NAME}")
            
            self._model = Model.from_pretrained(self.MODEL_NAME, token=self._hf_token)
            self._model = self._model.to(self.device)
            self._model.eval()
            
            self._inference = Inference(self._model, window="whole", device=self.device)
            
            if hasattr(self._model, 'dimension'):
                self.embedding_dim = self._model.dimension
            
            logger.info(f"Model loaded: {self.model_tag} (dim={self.embedding_dim})")
            
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise ModelLoadError(f"Failed to load Pyannote model: {e}") from e
    
    def _warmup_model(self) -> None:
        """Warmup model with dummy input."""
        if self._warmed_up or self._inference is None:
            return
        
        try:
            dummy_audio = {
                "waveform": torch.randn(1, int(1.5 * self.sample_rate)),
                "sample_rate": self.sample_rate,
            }
            with torch.no_grad():
                _ = self._inference(dummy_audio)
            self._warmed_up = True
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def extract_embedding(self, audio_signal: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio signal.
        
        Args:
            audio_signal: Normalized audio signal (float32, [-1, 1])
        
        Returns:
            L2-normalized embedding vector (256-dim)
        """
        if self._inference is None:
            raise ModelLoadError("Model not loaded")
        
        try:
            if audio_signal.dtype != np.float32:
                audio_signal = audio_signal.astype(np.float32)
            
            if audio_signal.ndim > 1:
                audio_signal = audio_signal.flatten()
            
            waveform = torch.from_numpy(audio_signal).unsqueeze(0)
            audio_dict = {"waveform": waveform, "sample_rate": self.sample_rate}
            
            with torch.no_grad():
                embedding = self._inference(audio_dict)
            
            if isinstance(embedding, torch.Tensor):
                emb_np = embedding.squeeze().cpu().numpy().astype(np.float32)
            else:
                emb_np = np.array(embedding).astype(np.float32)
            
            emb_np = emb_np.flatten()
            
            # L2 normalize
            norm = float(np.linalg.norm(emb_np) + 1e-8)
            if not np.isfinite(norm) or norm < 1e-6:
                raise ModelLoadError("Invalid embedding norm")
            
            return emb_np / norm
            
        except torch.cuda.OutOfMemoryError as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise ModelLoadError(f"GPU out of memory: {e}") from e
        except Exception as e:
            logger.exception(f"Embedding extraction failed: {e}")
            raise ModelLoadError(f"Embedding extraction failed: {e}") from e
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim
    
    def get_model_tag(self) -> str:
        return self.model_tag


# Backward compatibility alias
EmbeddingModelRepository = EmbeddingModel
