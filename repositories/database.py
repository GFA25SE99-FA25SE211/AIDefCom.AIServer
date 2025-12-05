"""Database connection pooling using SQLAlchemy.

Provides centralized database connection management with connection pooling
for efficient resource usage and better performance under load.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Optional
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from app.config import Config
from core.exceptions import AIServerException

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class DatabaseError(AIServerException):
    """Raised when database operations fail."""
    pass


class DatabasePool:
    """Centralized database connection pool manager.
    
    Uses SQLAlchemy's QueuePool for efficient connection reuse.
    Supports both pymssql and pyodbc drivers.
    """
    
    # Pool configuration (can be overridden via env vars)
    POOL_SIZE = 5  # Number of connections to keep in pool
    MAX_OVERFLOW = 10  # Max connections beyond pool_size
    POOL_TIMEOUT = 30  # Seconds to wait for available connection
    POOL_RECYCLE = 1800  # Recycle connections after 30 minutes
    POOL_PRE_PING = True  # Test connections before using
    
    _instance: Optional["DatabasePool"] = None
    _engine: Optional[Engine] = None
    _initialized: bool = False
    
    def __new__(cls) -> "DatabasePool":
        """Singleton pattern for global database pool."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize is handled in initialize() method."""
        pass
    
    @classmethod
    def initialize(
        cls,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        port: Optional[int] = None,
        driver: str = "pymssql",
        pool_size: int = POOL_SIZE,
        max_overflow: int = MAX_OVERFLOW,
        pool_timeout: int = POOL_TIMEOUT,
        pool_recycle: int = POOL_RECYCLE,
    ) -> None:
        """Initialize the database connection pool.
        
        Args:
            server: SQL Server hostname or instance
            database: Database name
            username: Database username
            password: Database password
            port: Optional port number
            driver: Database driver ('pymssql' or 'pyodbc')
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections
            pool_timeout: Seconds to wait for connection
            pool_recycle: Recycle connections after this many seconds
            
        Raises:
            DatabaseError: If initialization fails
        """
        if cls._initialized:
            logger.warning("Database pool already initialized, skipping")
            return
        
        # Use config values if not provided
        server = server or Config.SQL_SERVER_HOST
        database = database or Config.SQL_SERVER_DATABASE
        username = username or Config.SQL_SERVER_USERNAME
        password = password or Config.SQL_SERVER_PASSWORD
        
        # Strip 'tcp:' prefix if present (common in Azure connection strings)
        if server and server.lower().startswith('tcp:'):
            server = server[4:]
        
        # Parse port from server string (various formats)
        # Azure format: "host,port" or standard "host:port"
        if port is None and server and '\\' not in server:
            # Try comma format first (Azure SQL)
            if ',' in server:
                parts = server.rsplit(',', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    server = parts[0]
                    port = int(parts[1])
            # Try colon format (only if no comma, as comma takes precedence)
            elif ':' in server:
                parts = server.rsplit(':', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    server = parts[0]
                    port = int(parts[1])
        
        # Also check SQL_SERVER_PORT config
        if port is None:
            port_str = Config.SQL_SERVER_PORT
            if port_str and port_str.isdigit():
                port = int(port_str)
        
        if not all([server, database, username, password]):
            logger.warning(
                "Database configuration incomplete, connection pool not initialized | "
                "server=%s | database=%s | username=%s",
                bool(server), bool(database), bool(username)
            )
            return
        
        try:
            # Build connection URL based on driver
            connection_url = cls._build_connection_url(
                server=server,
                database=database,
                username=username,
                password=password,
                port=port,
                driver=driver,
            )
            
            # Create engine with connection pooling
            cls._engine = create_engine(
                connection_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=cls.POOL_PRE_PING,
                # Performance optimizations
                echo=False,  # Set to True for SQL debugging
                future=True,  # Use SQLAlchemy 2.0 style
            )
            
            # Test connection
            with cls._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            cls._initialized = True
            logger.info(
                "Database connection pool initialized | server=%s | database=%s | "
                "pool_size=%d | max_overflow=%d | driver=%s",
                server, database, pool_size, max_overflow, driver
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize database pool | error=%s | type=%s",
                str(e), type(e).__name__
            )
            raise DatabaseError(f"Failed to initialize database pool: {e}") from e
    
    @classmethod
    def _build_connection_url(
        cls,
        server: str,
        database: str,
        username: str,
        password: str,
        port: Optional[int],
        driver: str,
    ) -> str:
        """Build SQLAlchemy connection URL.
        
        Args:
            server: SQL Server hostname
            database: Database name
            username: Username
            password: Password
            port: Optional port
            driver: Driver name ('pymssql' or 'pyodbc')
            
        Returns:
            SQLAlchemy connection URL string
        """
        # URL-encode password to handle special characters
        encoded_password = quote_plus(password)
        
        if driver == "pymssql":
            # pymssql URL format: mssql+pymssql://user:pass@host:port/database
            if port:
                return f"mssql+pymssql://{username}:{encoded_password}@{server}:{port}/{database}"
            else:
                return f"mssql+pymssql://{username}:{encoded_password}@{server}/{database}"
        
        elif driver == "pyodbc":
            # pyodbc URL format with ODBC driver specification
            # mssql+pyodbc://user:pass@host:port/database?driver=ODBC+Driver+18+for+SQL+Server
            odbc_driver = quote_plus("ODBC Driver 18 for SQL Server")
            
            if port and '\\' not in server:
                # Standard port format (not named instance)
                server_str = f"{server}:{port}"
            elif '\\' in server:
                # Windows named instance - keep as is
                server_str = server
            else:
                server_str = server
            
            return (
                f"mssql+pyodbc://{username}:{encoded_password}@{server_str}/{database}"
                f"?driver={odbc_driver}&TrustServerCertificate=yes"
            )
        
        else:
            raise ValueError(f"Unsupported database driver: {driver}")
    
    @classmethod
    def get_engine(cls) -> Optional[Engine]:
        """Get the SQLAlchemy engine.
        
        Returns:
            Engine instance or None if not initialized
        """
        return cls._engine
    
    @classmethod
    @contextmanager
    def get_connection(cls) -> Generator["Connection", None, None]:
        """Get a connection from the pool.
        
        Usage:
            with DatabasePool.get_connection() as conn:
                result = conn.execute(text("SELECT * FROM users"))
                
        Yields:
            SQLAlchemy Connection object
            
        Raises:
            DatabaseError: If pool not initialized or connection fails
        """
        if not cls._initialized or cls._engine is None:
            raise DatabaseError("Database pool not initialized")
        
        try:
            with cls._engine.connect() as connection:
                yield connection
        except SQLAlchemyError as e:
            logger.error(
                "Database connection error | error=%s | type=%s",
                str(e), type(e).__name__
            )
            raise DatabaseError(f"Database connection error: {e}") from e
    
    @classmethod
    def execute_query(
        cls,
        query: str,
        params: Optional[dict] = None,
    ) -> list[dict]:
        """Execute a query and return results as list of dicts.
        
        Args:
            query: SQL query string (use :param_name for parameters)
            params: Optional dict of parameter values
            
        Returns:
            List of row dictionaries
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            with cls.get_connection() as conn:
                result = conn.execute(text(query), params or {})
                # Convert rows to dicts
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise DatabaseError(f"Query execution error: {e}") from e
    
    @classmethod
    def execute_update(
        cls,
        query: str,
        params: Optional[dict] = None,
    ) -> int:
        """Execute an update/insert/delete and return rows affected.
        
        Args:
            query: SQL query string (use :param_name for parameters)
            params: Optional dict of parameter values
            
        Returns:
            Number of rows affected
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            with cls.get_connection() as conn:
                result = conn.execute(text(query), params or {})
                conn.commit()
                return result.rowcount
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Update execution error: {e}")
            raise DatabaseError(f"Update execution error: {e}") from e
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the database pool is initialized."""
        return cls._initialized
    
    @classmethod
    def get_pool_status(cls) -> dict:
        """Get current pool status for monitoring.
        
        Returns:
            Dict with pool statistics
        """
        if not cls._initialized or cls._engine is None:
            return {"initialized": False}
        
        pool = cls._engine.pool
        return {
            "initialized": True,
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalidated_connections if hasattr(pool, 'invalidated_connections') else 0,
        }
    
    @classmethod
    def shutdown(cls) -> None:
        """Shutdown the connection pool and release all connections."""
        if cls._engine is not None:
            logger.info("Shutting down database connection pool...")
            cls._engine.dispose()
            cls._engine = None
            cls._initialized = False
            logger.info("Database connection pool shutdown complete")


# Convenience function for getting singleton instance
def get_database_pool() -> DatabasePool:
    """Get the database pool singleton instance."""
    return DatabasePool()
