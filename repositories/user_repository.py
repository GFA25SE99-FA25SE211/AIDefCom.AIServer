"""User Repository - Calls .NET Auth service to fetch user info by ID."""

from __future__ import annotations

import logging
from typing import Optional, Dict

import requests

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for .NET user API calls."""

    def __init__(self, base_url: str, verify_ssl: bool = False, timeout: int = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Fetch user by ID from .NET backend.

        Args:
            user_id: Identity user id (AspNetUsers.Id)
        Returns:
            Dict with user fields or None if not found
        """
        url = f"{self.base_url}/auth/users/{user_id}"
        try:
            resp = requests.get(url, timeout=self.timeout, verify=self.verify_ssl)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception as e:
                    logger.warning(f"User API returned non-JSON | id={user_id} | err={e}")
                    return None
            if resp.status_code == 404:
                logger.info(f"User not found | id={user_id}")
                return None
            logger.warning(
                "User API error | id=%s | status=%s | body=%s",
                user_id,
                resp.status_code,
                (resp.text[:500] if resp.text else ""),
            )
            return None
        except requests.RequestException as e:
            logger.error(f"User API request failed | id={user_id} | error={e}")
            return None
