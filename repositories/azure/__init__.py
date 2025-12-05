"""Azure repositories - Azure Blob Storage and Azure Speech SDK."""

from repositories.azure.blob_repository import AzureBlobRepository, AzureBlobStorageError
from repositories.azure.speech_repository import AzureSpeechRepository

__all__ = [
    "AzureBlobRepository",
    "AzureBlobStorageError",
    "AzureSpeechRepository",
]
