"""Azure Blob Storage Repository - Manages voice profile uploads to Azure Storage."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, ContentSettings
from azure.core.exceptions import ResourceExistsError, AzureError

from core.exceptions import AIServerException

logger = logging.getLogger(__name__)


class AzureBlobStorageError(AIServerException):
    """Raised when Azure Blob Storage operations fail."""
    pass


class AzureBlobRepository:
    """Repository for Azure Blob Storage operations."""
    
    def __init__(self, connection_string: str, container_name: str = "voice-sample") -> None:
        """
        Initialize Azure Blob Storage repository.
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Container name (default: "voice-sample", lowercase as per Azure requirements)
        """
        self.connection_string = connection_string
        self.container_name = container_name
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_client = self.blob_service_client.get_container_client(container_name)
            
            # Create container if not exists
            try:
                self.container_client.create_container()
                logger.info(f"Created container: {container_name}")
            except ResourceExistsError:
                logger.info(f"Container already exists: {container_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob Storage: {e}")
            raise AzureBlobStorageError(f"Blob storage initialization failed: {e}") from e
    
    def upload_voice_profile(self, user_id: str, profile_data: Dict[str, Any]) -> str:
        """
        Upload voice profile JSON to Azure Blob Storage.
        
        Args:
            user_id: User identifier
            profile_data: Profile data dictionary
        
        Returns:
            Blob URL (e.g., https://<account>.blob.core.windows.net/voice-sample/<user_id>/profile.json)
        
        Raises:
            AzureBlobStorageError: If upload fails
        """
        try:
            # Blob path: voice-sample/<user_id>/profile.json
            blob_name = f"{user_id}/profile.json"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Convert profile data to JSON string
            json_data = json.dumps(profile_data, indent=2, ensure_ascii=False)
            
            # Upload with overwrite
            blob_client.upload_blob(
                json_data,
                overwrite=True,
                content_settings=ContentSettings(content_type='application/json')
            )
            
            blob_url = blob_client.url
            logger.info(f"Uploaded voice profile to Azure Blob | user_id={user_id} | url={blob_url}")
            return blob_url
            
        except AzureError as e:
            logger.error(f"Failed to upload voice profile for user {user_id}: {e}")
            raise AzureBlobStorageError(f"Failed to upload profile: {e}") from e
    
    def download_voice_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Download voice profile JSON from Azure Blob Storage.
        
        Args:
            user_id: User identifier
        
        Returns:
            Profile data dictionary or None if not found
        
        Raises:
            AzureBlobStorageError: If download fails
        """
        try:
            blob_name = f"{user_id}/profile.json"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            if not blob_client.exists():
                logger.warning(f"Voice profile not found in Azure Blob | user_id={user_id}")
                return None
            
            # Download blob content
            blob_data = blob_client.download_blob().readall()
            profile_data = json.loads(blob_data.decode('utf-8'))
            
            logger.info(f"Downloaded voice profile from Azure Blob | user_id={user_id}")
            return profile_data
            
        except AzureError as e:
            logger.error(f"Failed to download voice profile for user {user_id}: {e}")
            raise AzureBlobStorageError(f"Failed to download profile: {e}") from e
    
    def delete_voice_profile(self, user_id: str) -> bool:
        """
        Delete voice profile from Azure Blob Storage.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if deleted, False if not found
        
        Raises:
            AzureBlobStorageError: If deletion fails
        """
        try:
            blob_name = f"{user_id}/profile.json"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            if not blob_client.exists():
                logger.warning(f"Voice profile not found for deletion | user_id={user_id}")
                return False
            
            blob_client.delete_blob()
            logger.info(f"Deleted voice profile from Azure Blob | user_id={user_id}")
            return True
            
        except AzureError as e:
            logger.error(f"Failed to delete voice profile for user {user_id}: {e}")
            raise AzureBlobStorageError(f"Failed to delete profile: {e}") from e
    
    def profile_exists_in_blob(self, user_id: str) -> bool:
        """
        Check if voice profile exists in Azure Blob Storage.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if exists, False otherwise
        """
        try:
            blob_name = f"{user_id}/profile.json"
            blob_client = self.container_client.get_blob_client(blob_name)
            return blob_client.exists()
        except Exception as e:
            logger.error(f"Failed to check profile existence for user {user_id}: {e}")
            return False

    def list_voice_profile_ids(self) -> list[str]:
        """List user_ids that have a profile JSON in the container.

        Tolerates extra nested paths by extracting the last two segments.
        Valid profile path pattern: <user_id>/profile.json
        """
        results: list[str] = []
        try:
            for blob in self.container_client.list_blobs():
                name = blob.name  # e.g. user123/profile.json
                if not name.endswith('/profile.json'):
                    continue
                parts = name.split('/')
                if len(parts) < 2:
                    continue
                # Extract user_id from path: user_id/profile.json
                user_id = parts[-2]
                results.append(user_id)
            return results
        except Exception as e:
            logger.error(f"Failed listing voice profiles in blob: {e}")
            return results
