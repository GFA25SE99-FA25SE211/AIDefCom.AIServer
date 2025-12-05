"""SQL Server Repository - Manages AppUser database operations.

Supports connection pooling via SQLAlchemy and legacy per-query connections.
Supports pymssql and pyodbc drivers.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

try:
    import pymssql
except ImportError:
    pymssql = None  # type: ignore

try:
    import pyodbc  # type: ignore
except Exception:
    pyodbc = None  # type: ignore

from core.exceptions import AIServerException
from repositories.interfaces.sql_repository import ISQLServerRepository
from repositories.database import DatabasePool, DatabaseError

logger = logging.getLogger(__name__)


class SQLServerError(AIServerException):
    """Raised when SQL Server operations fail."""
    pass


class SQLServerRepository(ISQLServerRepository):
    """Concrete implementation of ISQLServerRepository for AppUser + transcripts."""
    
    def __init__(
        self,
        server: str,
        database: str,
        username: str,
        password: str,
        users_table: str = "AspNetUsers",
        user_id_column: str = "Id",
        voice_path_column: str = "VoiceSamplePath",
        port: Optional[int] = None,
        encrypt: bool = False,
        trust_server_certificate: bool = True,
        use_pool: bool = True,
    ) -> None:
        """Initialize SQL Server repository."""
        self.server, self.port = self._parse_server_string(server, port)
        self.database = database
        self.username = username
        self.password = password
        self.users_table = self._sanitize_ident(users_table)
        self.user_id_column = self._sanitize_ident(user_id_column)
        self.voice_path_column = self._sanitize_ident(voice_path_column)
        self.encrypt = encrypt
        self.trust_server_certificate = trust_server_certificate
        
        self._use_pool = use_pool and DatabasePool.is_initialized()
        
        if not self._use_pool:
            has_named_instance = '\\' in self.server
            if has_named_instance and pyodbc is not None:
                self._driver: str = "pyodbc"
                self._has_pyodbc = False
            else:
                self._driver: str = "pymssql"
                self._has_pyodbc = pyodbc is not None
        
        logger.info(
            f"SQL Server repository initialized | server={self.server} | db={database} | use_pool={self._use_pool}"
        )
    
    @staticmethod
    def _parse_server_string(server: str, port: Optional[int] = None) -> tuple[str, Optional[int]]:
        """Parse server string to extract hostname and port."""
        if not server:
            raise ValueError("Server cannot be empty")
        
        if server.lower().startswith('tcp:'):
            server = server[4:]
        
        if port is not None:
            return (server, port)
        
        if ':' in server and '\\' not in server:
            parts = server.split(':', 1)
            try:
                return (parts[0], int(parts[1]))
            except (ValueError, IndexError):
                return (server, None)
        elif ',' in server and '\\' not in server.split(',')[0]:
            parts = server.split(',', 1)
            try:
                return (parts[0], int(parts[1]))
            except (ValueError, IndexError):
                return (server, None)
        
        return (server, None)

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        users_table: str = "AspNetUsers",
        user_id_column: str = "Id",
        voice_path_column: str = "VoiceSamplePath",
    ) -> "SQLServerRepository":
        """Create repository from a connection string."""
        if not connection_string or not connection_string.strip():
            raise ValueError("Connection string cannot be empty")
        
        params = cls._parse_connection_string(connection_string)
        
        server = params.get("server") or params.get("data source") or params.get("addr")
        database = params.get("database") or params.get("initial catalog")
        username = params.get("user id") or params.get("uid") or params.get("user")
        password = params.get("password") or params.get("pwd")
        port_str = params.get("port")
        port = int(port_str) if port_str and port_str.isdigit() else None
        encrypt = params.get("encrypt", "").lower() in ("true", "yes", "1")
        trust_cert = params.get("trustservercertificate", "").lower() in ("true", "yes", "1")
        
        if not all([server, database, username, password]):
            raise ValueError("Connection string missing required parameters")
        
        return cls(
            server=server,
            database=database,
            username=username,
            password=password,
            port=port,
            encrypt=encrypt,
            trust_server_certificate=trust_cert,
            users_table=users_table,
            user_id_column=user_id_column,
            voice_path_column=voice_path_column,
        )
    
    @staticmethod
    def _parse_connection_string(conn_str: str) -> dict[str, str]:
        """Parse connection string into key-value dictionary."""
        params: dict[str, str] = {}
        pattern = r'([^;=]+)=([^;]*(?:"[^"]*")?[^;]*)'
        matches = re.findall(pattern, conn_str, re.IGNORECASE)
        
        for key, value in matches:
            key = key.strip().lower()
            value = value.strip().strip('"').strip("'")
            if key and value:
                params[key] = value
        
        return params

    @staticmethod
    def _sanitize_ident(name: str) -> str:
        """Sanitize SQL identifier to prevent injection."""
        return "".join(ch for ch in (name or "") if ch.isalnum() or ch == "_") or "AspNetUsers"
    
    def _get_connection(self):
        """Get database connection using selected driver."""
        if self._driver == "pymssql":
            try:
                connect_params = {
                    "server": self.server,
                    "database": self.database,
                    "user": self.username,
                    "password": self.password,
                    "as_dict": False,
                }
                if self.port:
                    connect_params["port"] = self.port
                
                conn = pymssql.connect(**connect_params)
                return conn
            except Exception as e:
                if not self._has_pyodbc:
                    raise SQLServerError(f"Database connection failed (pymssql): {e}") from e
                self._driver = "pyodbc"

        if self._driver == "pyodbc":
            if pyodbc is None:
                raise SQLServerError("pyodbc not available")
            try:
                driver_names = [
                    "ODBC Driver 18 for SQL Server",
                    "ODBC Driver 17 for SQL Server",
                    "SQL Server",
                ]
                chosen = driver_names[1]
                try:
                    available = [d for d in pyodbc.drivers()]
                    for cand in driver_names:
                        if cand in available:
                            chosen = cand
                            break
                except Exception:
                    pass
                
                server_str = self.server
                if self.port and '\\' not in self.server:
                    server_str = f"{self.server},{self.port}"
                
                encrypt_str = "yes" if self.encrypt else "no"
                trust_cert_str = "yes" if self.trust_server_certificate else "no"
                
                conn_str = (
                    f"DRIVER={{{chosen}}};SERVER={server_str};DATABASE={self.database};"
                    f"UID={self.username};PWD={self.password};"
                    f"Encrypt={encrypt_str};TrustServerCertificate={trust_cert_str};"
                )
                conn = pyodbc.connect(conn_str, autocommit=True)
                return conn
            except Exception as e:
                raise SQLServerError(f"Database connection failed (pyodbc): {e}") from e
    
    def update_voice_sample_path(self, user_id: str, blob_url: str) -> bool:
        """Update VoiceSamplePath for AppUser."""
        try:
            if self._use_pool:
                return self._update_voice_sample_path_pooled(user_id, blob_url)
            else:
                return self._update_voice_sample_path_legacy(user_id, blob_url)
        except Exception as e:
            logger.error(f"Failed to update VoiceSamplePath for user {user_id}: {e}")
            raise SQLServerError(f"Failed to update database: {e}") from e
    
    def _update_voice_sample_path_pooled(self, user_id: str, blob_url: str) -> bool:
        query = (
            f"UPDATE {self.users_table} "
            f"SET {self.voice_path_column} = :blob_url "
            f"WHERE {self.user_id_column} = :user_id"
        )
        try:
            rows = DatabasePool.execute_update(query, {"blob_url": blob_url, "user_id": user_id})
            if rows == 0:
                return False
            logger.info(f"Updated VoiceSamplePath for user {user_id}")
            return True
        except DatabaseError as e:
            raise SQLServerError(str(e)) from e
    
    def _update_voice_sample_path_legacy(self, user_id: str, blob_url: str) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            placeholder = "%s" if self._driver == "pymssql" else "?"
            query = (
                f"UPDATE {self.users_table} "
                f"SET {self.voice_path_column} = {placeholder} "
                f"WHERE {self.user_id_column} = {placeholder}"
            )
            cursor.execute(query, (blob_url, user_id))
            try:
                conn.commit()
            except Exception:
                pass
            rows_affected = getattr(cursor, "rowcount", 0)
            return rows_affected > 0
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
    
    def get_voice_sample_path(self, user_id: str) -> Optional[str]:
        """Get VoiceSamplePath for AppUser."""
        try:
            if self._use_pool:
                return self._get_voice_sample_path_pooled(user_id)
            else:
                return self._get_voice_sample_path_legacy(user_id)
        except Exception as e:
            logger.error(f"Failed to get VoiceSamplePath for user {user_id}: {e}")
            raise SQLServerError(f"Failed to query database: {e}") from e
    
    def _get_voice_sample_path_pooled(self, user_id: str) -> Optional[str]:
        query = (
            f"SELECT {self.voice_path_column} "
            f"FROM {self.users_table} "
            f"WHERE {self.user_id_column} = :user_id"
        )
        try:
            rows = DatabasePool.execute_query(query, {"user_id": user_id})
            if not rows:
                return None
            return rows[0].get(self.voice_path_column)
        except DatabaseError as e:
            raise SQLServerError(str(e)) from e
    
    def _get_voice_sample_path_legacy(self, user_id: str) -> Optional[str]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            placeholder = "%s" if self._driver == "pymssql" else "?"
            query = (
                f"SELECT {self.voice_path_column} "
                f"FROM {self.users_table} "
                f"WHERE {self.user_id_column} = {placeholder}"
            )
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists in database."""
        try:
            if self._use_pool:
                return self._user_exists_pooled(user_id)
            else:
                return self._user_exists_legacy(user_id)
        except Exception:
            return False
    
    def _user_exists_pooled(self, user_id: str) -> bool:
        query = (
            f"SELECT COUNT(*) as cnt "
            f"FROM {self.users_table} "
            f"WHERE {self.user_id_column} = :user_id"
        )
        try:
            rows = DatabasePool.execute_query(query, {"user_id": user_id})
            return rows[0].get("cnt", 0) > 0 if rows else False
        except DatabaseError:
            return False
    
    def _user_exists_legacy(self, user_id: str) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            placeholder = "%s" if self._driver == "pymssql" else "?"
            query = (
                f"SELECT COUNT(*) "
                f"FROM {self.users_table} "
                f"WHERE {self.user_id_column} = {placeholder}"
            )
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()
            return row[0] > 0 if row else False
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    def save_transcript(
        self,
        session_id: str,
        start_time: str,
        end_time: str,
        duration_seconds: float,
        initial_speaker: str,
        lines: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        transcripts_table: str = "SpeechTranscripts",
    ) -> bool:
        """Save speech transcript to database."""
        try:
            if self._use_pool:
                return self._save_transcript_pooled(
                    session_id, start_time, end_time, duration_seconds,
                    initial_speaker, lines, user_id, transcripts_table
                )
            else:
                return self._save_transcript_legacy(
                    session_id, start_time, end_time, duration_seconds,
                    initial_speaker, lines, user_id, transcripts_table
                )
        except Exception as e:
            logger.error(f"Failed to save transcript for session {session_id}: {e}")
            return False
    
    def _save_transcript_pooled(
        self,
        session_id: str,
        start_time: str,
        end_time: str,
        duration_seconds: float,
        initial_speaker: str,
        lines: List[Dict[str, Any]],
        user_id: Optional[str],
        transcripts_table: str,
    ) -> bool:
        lines_json = json.dumps(lines, ensure_ascii=False)
        total_lines = len(lines)
        full_text = " ".join(line.get("text", "") for line in lines)
        table_name = self._sanitize_ident(transcripts_table)
        
        query = (
            f"INSERT INTO {table_name} "
            f"(SessionId, UserId, StartTime, EndTime, DurationSeconds, "
            f"InitialSpeaker, TotalLines, FullText, LinesJson, CreatedAt) "
            f"VALUES (:session_id, :user_id, :start_time, :end_time, "
            f":duration, :initial_speaker, :total_lines, :full_text, :lines_json, GETUTCDATE())"
        )
        
        try:
            DatabasePool.execute_update(query, {
                "session_id": session_id,
                "user_id": user_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration_seconds,
                "initial_speaker": initial_speaker,
                "total_lines": total_lines,
                "full_text": full_text,
                "lines_json": lines_json,
            })
            return True
        except DatabaseError:
            return False
    
    def _save_transcript_legacy(
        self,
        session_id: str,
        start_time: str,
        end_time: str,
        duration_seconds: float,
        initial_speaker: str,
        lines: List[Dict[str, Any]],
        user_id: Optional[str],
        transcripts_table: str,
    ) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            placeholder = "%s" if self._driver == "pymssql" else "?"
            
            lines_json = json.dumps(lines, ensure_ascii=False)
            total_lines = len(lines)
            full_text = " ".join(line.get("text", "") for line in lines)
            
            query = (
                f"INSERT INTO {self._sanitize_ident(transcripts_table)} "
                f"(SessionId, UserId, StartTime, EndTime, DurationSeconds, "
                f"InitialSpeaker, TotalLines, FullText, LinesJson, CreatedAt) "
                f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, "
                f"{placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, GETUTCDATE())"
            )
            
            cursor.execute(query, (
                session_id, user_id, start_time, end_time,
                duration_seconds, initial_speaker, total_lines, full_text, lines_json,
            ))
            conn.commit()
            return True
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
