"""Azure Blob Storage Repository - Manages voice profile uploads to Azure Storage."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError, AzureError

from core.exceptions import AIServerException

logger = logging.getLogger(__name__)


class AzureBlobStorageError(AIServerException):
    """Raised when Azure Blob Storage operations fail."""
    pass


class AzureBlobRepository:
    """Repository for Azure Blob Storage operations."""
    
    def __init__(self, connection_string: str, container_name: str = "voice-sample") -> None:
        """Initialize Azure Blob Storage repository.
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Container name (default: "voice-sample")
        """
        self.connection_string = connection_string
        self.container_name = container_name
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_client = self.blob_service_client.get_container_client(container_name)
            
            try:
                self.container_client.create_container()
                logger.info(f"Created container: {container_name}")
            except ResourceExistsError:
                pass  # Container already exists
                
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob Storage: {e}")
            raise AzureBlobStorageError(f"Blob storage initialization failed: {e}") from e
    
    def upload_voice_profile(self, user_id: str, profile_data: Dict[str, Any]) -> str:
        """Upload voice profile JSON to Azure Blob Storage.
        
        Args:
            user_id: User identifier
            profile_data: Profile data dictionary
        
        Returns:
            Blob URL
        
        Raises:
            AzureBlobStorageError: If upload fails
        """
        try:
            blob_name = f"{user_id}/profile.json"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            json_data = json.dumps(profile_data, indent=2, ensure_ascii=False)
            
            blob_client.upload_blob(
                json_data,
                overwrite=True,
                content_settings=ContentSettings(content_type='application/json')
            )
            
            blob_url = blob_client.url
            logger.info(f"Uploaded voice profile | user_id={user_id}")
            return blob_url
            
        except AzureError as e:
            logger.error(f"Failed to upload voice profile for user {user_id}: {e}")
            raise AzureBlobStorageError(f"Failed to upload profile: {e}") from e
    
    def download_voice_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Download voice profile JSON from Azure Blob Storage.
        
        Args:
            user_id: User identifier
        
        Returns:
            Profile data dictionary or None if not found
        """
        try:
            blob_name = f"{user_id}/profile.json"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            if not blob_client.exists():
                return None
            
            blob_data = blob_client.download_blob().readall()
            return json.loads(blob_data.decode('utf-8'))
            
        except AzureError as e:
            logger.error(f"Failed to download voice profile for user {user_id}: {e}")
            raise AzureBlobStorageError(f"Failed to download profile: {e}") from e
    
    def delete_voice_profile(self, user_id: str) -> bool:
        """Delete voice profile from Azure Blob Storage.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if deleted, False if not found
        """
        try:
            blob_name = f"{user_id}/profile.json"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            if not blob_client.exists():
                return False
            
            blob_client.delete_blob()
            logger.info(f"Deleted voice profile | user_id={user_id}")
            return True
            
        except AzureError as e:
            logger.error(f"Failed to delete voice profile for user {user_id}: {e}")
            raise AzureBlobStorageError(f"Failed to delete profile: {e}") from e
    
    def profile_exists_in_blob(self, user_id: str) -> bool:
        """Check if voice profile exists in Azure Blob Storage."""
        try:
            blob_name = f"{user_id}/profile.json"
            blob_client = self.container_client.get_blob_client(blob_name)
            return blob_client.exists()
        except Exception as e:
            logger.error(f"Failed to check profile existence for user {user_id}: {e}")
            return False

    def list_voice_profile_ids(self, max_results: int = 500) -> List[str]:
        """List user_ids that have a profile JSON in the container.
        
        Args:
            max_results: Maximum number of profiles to return
        """
        results: List[str] = []
        try:
            for blob in self.container_client.list_blobs(results_per_page=100):
                name = blob.name
                if not name.endswith('/profile.json'):
                    continue
                parts = name.split('/')
                if len(parts) < 2:
                    continue
                user_id = parts[-2]
                results.append(user_id)
                if len(results) >= max_results:
                    logger.warning(f"list_voice_profile_ids truncated at {max_results}")
                    break
            return results
        except Exception as e:
            logger.error(f"Failed listing voice profiles in blob: {e}")
            return results
