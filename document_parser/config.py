"""Configuration management using Pydantic settings."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OutputFormat(str, Enum):
    """Output format for processed documents."""

    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MinerU vLLM Server Configuration
    mineru_vllm_url: str = Field(
        default="http://localhost:4444",
        description="MinerU 2.5 vLLM server base URL (API server will communicate with this)",
    )
    mineru_model: str = Field(
        default="opendatalab/MinerU2.5-2509-1.2B",
        description="MinerU model ID (for reference only, hosted on vLLM server)",
    )

    # Processing Configuration
    concurrency: int = Field(
        default=16,
        description="Number of concurrent pages to process within each batch",
    )
    batch_size: int = Field(
        default=64,
        description="Number of pages to process per batch (reduces memory usage)",
    )
    max_pages_per_batch: int = Field(
        default=1000,
        description="Maximum pages per job",
    )
    timeout: int = Field(
        default=600,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed requests",
    )

    # Output Configuration
    output_dir: Path = Field(default=Path("./output"), description="Output directory")
    cache_dir: Path = Field(default=Path("./cache"), description="Cache directory")
    log_level: str = Field(default="INFO", description="Logging level")

    # Graph-Vector DB (optional)
    neo4j_uri: Optional[str] = Field(
        default=None,
        description="Neo4j connection URI",
    )
    neo4j_user: Optional[str] = Field(default=None, description="Neo4j username")
    neo4j_password: Optional[str] = Field(default=None, description="Neo4j password")

    # API Server (optional)
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8080, description="API server port")
    api_workers: int = Field(default=1, description="API server workers")

    # Redis (for job queue persistence)
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for job queue",
    )
    num_workers: int = Field(
        default=2,
        description="Number of background workers for async job processing",
    )

    def model_post_init(self, __context):
        """Create directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
