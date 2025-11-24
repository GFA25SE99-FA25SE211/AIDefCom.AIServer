"""SQL Server Repository - Manages AppUser database operations.

Supports two drivers:
- pymssql (default)
- pyodbc (automatic fallback if pymssql fails or is unavailable)

Configuration methods:
1. Connection string: Automatically parsed to extract server, database, credentials
2. Individual parameters: Server, database, username, password specified separately
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from typing import Optional

import pymssql
try:  # Optional fallback driver on Windows/local SQL Express
    import pyodbc  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pyodbc = None  # type: ignore

from core.exceptions import AIServerException
from repositories.interfaces.i_sql_server_repository import ISQLServerRepository

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
    ) -> None:
        """Initialize SQL Server repository.
        
        Args:
            server: Server hostname or instance (e.g., 'localhost', 'SERVER\\INSTANCE')
            database: Database name
            username: Database username
            password: Database password
            users_table: Table name for users (default: AspNetUsers)
            user_id_column: Column name for user ID (default: Id)
            voice_path_column: Column name for voice sample path (default: VoiceSamplePath)
            port: Optional port number (default: 1433 for default instance)
            encrypt: Whether to encrypt the connection (required for Azure SQL)
            trust_server_certificate: Whether to trust the server certificate
        """
        # Parse server string to handle Windows named instances (SERVER\INSTANCE)
        self.server, self.port = self._parse_server_string(server, port)
        self.database = database
        self.username = username
        self.password = password
        self.users_table = self._sanitize_ident(users_table)
        self.user_id_column = self._sanitize_ident(user_id_column)
        self.voice_path_column = self._sanitize_ident(voice_path_column)
        self.encrypt = encrypt
        self.trust_server_certificate = trust_server_certificate
        self.database = database
        self.username = username
        self.password = password
        self.users_table = self._sanitize_ident(users_table)
        self.user_id_column = self._sanitize_ident(user_id_column)
        self.voice_path_column = self._sanitize_ident(voice_path_column)
        
        # Decide driver strategy
        # For Windows named instances (containing backslash), prefer pyodbc if available
        # because pymssql has issues with named instances
        has_named_instance = '\\' in self.server
        logger.info(
            "SQL Server driver selection | server='%s' | has_named_instance=%s | pyodbc_available=%s",
            self.server, has_named_instance, pyodbc is not None
        )
        
        if has_named_instance and pyodbc is not None:
            self._driver: str = "pyodbc"
            self._has_pyodbc = False  # Don't fallback to pymssql
            logger.info("Detected Windows named instance, using pyodbc driver")
        else:
            self._driver: str = "pymssql"
            if pyodbc is not None:
                # Prefer pymssql first; we'll fallback to pyodbc on connection failure
                self._has_pyodbc = True
            else:
                self._has_pyodbc = False
        
        logger.info(
            "SQL Server repository initialized | server=%s | port=%s | db=%s | table=%s | id_col=%s | path_col=%s",
            self.server, self.port or "default", database, self.users_table, self.user_id_column, self.voice_path_column,
        )
    
    @staticmethod
    def _parse_server_string(server: str, port: Optional[int] = None) -> tuple[str, Optional[int]]:
        """Parse server string to extract hostname and port.
        
        Handles formats:
        - 'hostname' -> ('hostname', None)
        - 'hostname:port' -> ('hostname', port)
        - 'hostname\\instance' -> ('hostname\\instance', None) for Windows named instances
        - 'hostname,port' -> ('hostname', port) - SQL Server alternate format
        - 'tcp:hostname,port' -> ('hostname', port) - Azure SQL format
        
        Args:
            server: Server string from config
            port: Explicit port override
            
        Returns:
            Tuple of (server, port)
        """
        if not server:
            raise ValueError("Server cannot be empty")
        
        # Remove 'tcp:' prefix if present (Azure SQL format)
        if server.lower().startswith('tcp:'):
            server = server[4:]
        
        # If explicit port provided, use it
        if port is not None:
            return (server, port)
        
        # Check for port in server string (format: hostname:port or hostname,port)
        # But be careful with Windows named instances (SERVER\INSTANCE)
        if ':' in server and '\\' not in server:
            # Standard hostname:port format
            parts = server.split(':', 1)
            try:
                return (parts[0], int(parts[1]))
            except (ValueError, IndexError):
                logger.warning(f"Invalid port in server string '{server}', using default")
                return (server, None)
        elif ',' in server and '\\' not in server.split(',')[0]:
            # SQL Server alternate format: hostname,port
            parts = server.split(',', 1)
            try:
                return (parts[0], int(parts[1]))
            except (ValueError, IndexError):
                logger.warning(f"Invalid port in server string '{server}', using default")
                return (server, None)
        
        # Otherwise, it's either a simple hostname or Windows named instance (SERVER\INSTANCE)
        return (server, None)

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        users_table: str = "AspNetUsers",
        user_id_column: str = "Id",
        voice_path_column: str = "VoiceSamplePath",
    ) -> "SQLServerRepository":
        """Create repository from a connection string.
        
        Supports multiple formats:
        - ADO.NET: "Server=HOST;Database=DB;User Id=USER;Password=PASS;"
        - ODBC: "Driver={SQL Server};Server=HOST;Database=DB;UID=USER;PWD=PASS;"
        - Semicolon-delimited key-value pairs
        
        Args:
            connection_string: SQL Server connection string
            users_table: Table name for users
            user_id_column: Column name for user ID
            voice_path_column: Column name for voice sample path
            
        Returns:
            SQLServerRepository instance
            
        Raises:
            ValueError: If connection string is invalid or missing required parameters
        """
        if not connection_string or not connection_string.strip():
            raise ValueError("Connection string cannot be empty")
        
        # Parse connection string into key-value pairs
        params = cls._parse_connection_string(connection_string)
        
        # Extract required parameters (case-insensitive keys)
        server = (
            params.get("server") 
            or params.get("data source") 
            or params.get("addr")
            or params.get("address")
        )
        database = (
            params.get("database") 
            or params.get("initial catalog")
        )
        username = (
            params.get("user id") 
            or params.get("uid") 
            or params.get("user")
        )
        password = (
            params.get("password") 
            or params.get("pwd")
        )
        port_str = params.get("port")
        port = int(port_str) if port_str and port_str.isdigit() else None
        
        # Extract encryption and trust settings for Azure SQL
        encrypt = params.get("encrypt", "").lower() in ("true", "yes", "1")
        trust_cert = params.get("trustservercertificate", "").lower() in ("true", "yes", "1")
        
        # Validate required parameters
        if not server:
            raise ValueError("Connection string missing 'Server' or 'Data Source'")
        if not database:
            raise ValueError("Connection string missing 'Database' or 'Initial Catalog'")
        if not username:
            raise ValueError("Connection string missing 'User Id' or 'UID'")
        if not password:
            raise ValueError("Connection string missing 'Password' or 'PWD'")
        
        logger.info(
            "Creating SQL Server repository from connection string | server=%s | db=%s | encrypt=%s | trust_cert=%s",
            server, database, encrypt, trust_cert
        )
        
        instance = cls(
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
        
        return instance
    
    @staticmethod
    def _parse_connection_string(conn_str: str) -> dict[str, str]:
        """Parse connection string into key-value dictionary.
        
        Handles various formats:
        - Key=Value;Key2=Value2;
        - Key=Value with spaces;Key2=Value2;
        - Values in quotes: Key="Value";
        
        Returns:
            Dictionary with lowercase keys and trimmed values
        """
        params: dict[str, str] = {}
        
        # Split by semicolon, but respect quoted values
        # Pattern: key=value or key="value with spaces"
        pattern = r'([^;=]+)=([^;]*(?:"[^"]*")?[^;]*)'
        matches = re.findall(pattern, conn_str, re.IGNORECASE)
        
        for key, value in matches:
            # Clean up key and value
            key = key.strip().lower()
            value = value.strip().strip('"').strip("'")
            
            if key and value:
                params[key] = value
        
        return params

    @staticmethod
    def _sanitize_ident(name: str) -> str:
        """Allow only alphanumerics and underscore in identifiers to prevent SQL injection in object names."""
        return "".join(ch for ch in (name or "") if ch.isalnum() or ch == "_") or "AspNetUsers"
    
    def _get_connection(self):  # returns a connection from the active driver
        """Get database connection using selected driver; fallback to pyodbc if pymssql fails."""
        # First try pymssql
        if self._driver == "pymssql":
            try:
                # Build connection parameters
                connect_params = {
                    "server": self.server,
                    "database": self.database,
                    "user": self.username,
                    "password": self.password,
                    "as_dict": False,
                }
                
                # Only add port if explicitly specified (pymssql handles named instances automatically)
                if self.port:
                    connect_params["port"] = self.port
                
                logger.info(
                    "Attempting SQL connection (pymssql) | server='%s' | port=%s | db='%s' | user='%s'",
                    self.server, self.port or "default", self.database, self.username
                )
                
                conn = pymssql.connect(**connect_params)
                logger.info("SQL Server connection successful (pymssql)")
                return conn
            except Exception as e:
                logger.error(
                    "pymssql connect failed | error=%s | type=%s | server=%s", 
                    str(e), type(e).__name__, self.server
                )
                if not self._has_pyodbc:
                    raise SQLServerError(f"Database connection failed (pymssql): {e}") from e
                # Fallback to pyodbc
                self._driver = "pyodbc"
                logger.warning("Falling back to pyodbc driver for SQL Server connection")

        # Try pyodbc
        if self._driver == "pyodbc":
            if pyodbc is None:
                raise SQLServerError("pyodbc not available for SQL Server connection")
            try:
                # Choose a common SQL Server ODBC driver name (17 or 18). Try 18 first.
                driver_names = [
                    "ODBC Driver 18 for SQL Server",
                    "ODBC Driver 17 for SQL Server",
                    "SQL Server",
                ]
                chosen = None
                try:
                    available = [d for d in pyodbc.drivers()]  # type: ignore[attr-defined]
                    for cand in driver_names:
                        if cand in available:
                            chosen = cand
                            break
                except Exception:
                    # If unable to enumerate drivers, assume 17
                    chosen = driver_names[1]
                if chosen is None:
                    chosen = driver_names[1]
                
                # Build server string for ODBC (may include port)
                server_str = self.server
                if self.port and '\\' not in self.server:
                    # Only append port if it's not a named instance
                    server_str = f"{self.server},{self.port}"
                
                # Build connection string with encryption settings
                encrypt_str = "yes" if self.encrypt else "no"
                trust_cert_str = "yes" if self.trust_server_certificate else "no"
                
                conn_str = (
                    f"DRIVER={{{chosen}}};SERVER={server_str};DATABASE={self.database};"
                    f"UID={self.username};PWD={self.password};"
                    f"Encrypt={encrypt_str};TrustServerCertificate={trust_cert_str};"
                    f"MARS_Connection=Yes;"
                )
                logger.info(
                    "Attempting SQL connection (pyodbc) | driver='%s' | server='%s' | encrypt=%s | trust_cert=%s", 
                    chosen, server_str, encrypt_str, trust_cert_str
                )
                conn = pyodbc.connect(conn_str, autocommit=True)  # type: ignore
                logger.info("SQL Server connection successful (pyodbc)")
                return conn
            except Exception as e:
                logger.error(
                    "pyodbc connect failed | error=%s | type=%s | server=%s", 
                    str(e), type(e).__name__, self.server
                )
                raise SQLServerError(f"Database connection failed (pyodbc): {e}") from e
    
    def update_voice_sample_path(self, user_id: str, blob_url: str) -> bool:
        """
        Update VoiceSamplePath for AppUser.
        
        Args:
            user_id: User ID (matches AspNetUsers.Id)
            blob_url: Azure Blob Storage URL for voice profile
        
        Returns:
            True if updated successfully
        
        Raises:
            SQLServerError: If update fails
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Parameter placeholder differs by driver
                placeholder = "%s" if self._driver == "pymssql" else "?"
                query = (
                    f"UPDATE {self.users_table} "
                    f"SET {self.voice_path_column} = {placeholder} "
                    f"WHERE {self.user_id_column} = {placeholder}"
                )
                cursor.execute(query, (blob_url, user_id))
                # For pymssql, explicit commit; pyodbc uses autocommit=True
                try:
                    conn.commit()
                except Exception:
                    pass
                rows_affected = getattr(cursor, "rowcount", 0)
                if rows_affected == 0:
                    logger.warning(f"No user found with ID: {user_id}")
                    return False
                logger.info(f"Updated VoiceSamplePath for user {user_id} | url={blob_url}")
                return True
            finally:
                try:
                    cursor.close()  # type: ignore
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to update VoiceSamplePath for user {user_id}: {e}")
            raise SQLServerError(f"Failed to update database: {e}") from e
    
    def get_voice_sample_path(self, user_id: str) -> Optional[str]:
        """
        Get VoiceSamplePath for AppUser.
        
        Args:
            user_id: User ID
        
        Returns:
            Blob URL or None if not set
        
        Raises:
            SQLServerError: If query fails
        """
        try:
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
                if row is None:
                    logger.warning(f"No user found with ID: {user_id}")
                    return None
                # pyodbc may return Row object; index by 0
                voice_sample_path = row[0]
                logger.info(f"Retrieved VoiceSamplePath for user {user_id} | path={voice_sample_path}")
                return voice_sample_path
            finally:
                try:
                    cursor.close()  # type: ignore
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to get VoiceSamplePath for user {user_id}: {e}")
            raise SQLServerError(f"Failed to query database: {e}") from e
    
    def user_exists(self, user_id: str) -> bool:
        """
        Check if user exists in database.
        
        Args:
            user_id: User ID
        
        Returns:
            True if user exists
        """
        try:
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
                count = row[0] if row else 0
                return count > 0
            finally:
                try:
                    cursor.close()  # type: ignore
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to check user existence for {user_id}: {e}")
            return False

    def save_transcript(
        self,
        session_id: str,
        start_time: str,
        end_time: str,
        duration_seconds: float,
        initial_speaker: str,
        lines: list[dict],
        user_id: Optional[str] = None,
        transcripts_table: str = "SpeechTranscripts",
    ) -> bool:
        """Save speech transcript to database.
        
        Args:
            session_id: Session identifier
            start_time: Session start timestamp (ISO format)
            end_time: Session end timestamp (ISO format)
            duration_seconds: Total duration in seconds
            initial_speaker: Initial speaker label
            lines: List of transcript lines (each with timestamp, speaker, text, user_id)
            user_id: Optional user ID for association
            transcripts_table: Table name for transcripts
            
        Returns:
            True if saved successfully, False otherwise
        """
        import json
        
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                placeholder = "%s" if self._driver == "pymssql" else "?"
                
                # Serialize lines as JSON
                lines_json = json.dumps(lines, ensure_ascii=False)
                total_lines = len(lines)
                
                # Full transcript text (concatenate all lines)
                full_text = " ".join(line.get("text", "") for line in lines)
                
                query = (
                    f"INSERT INTO {self._sanitize_ident(transcripts_table)} "
                    f"(SessionId, UserId, StartTime, EndTime, DurationSeconds, "
                    f"InitialSpeaker, TotalLines, FullText, LinesJson, CreatedAt) "
                    f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, "
                    f"{placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, GETUTCDATE())"
                )
                
                cursor.execute(
                    query,
                    (
                        session_id,
                        user_id,
                        start_time,
                        end_time,
                        duration_seconds,
                        initial_speaker,
                        total_lines,
                        full_text,
                        lines_json,
                    ),
                )
                conn.commit()
                
                logger.info(
                    f"Transcript saved | session_id={session_id} | lines={total_lines} | duration={duration_seconds:.1f}s"
                )
                return True
                
            except Exception as e:
                logger.exception(f"Failed to save transcript for session {session_id}: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                return False
            finally:
                try:
                    cursor.close()  # type: ignore
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to connect to database for saving transcript: {e}")
            return False
