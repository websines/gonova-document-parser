"""Configuration management using Pydantic settings."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceMode(str, Enum):
    """Inference mode for model execution."""

    TRANSFORMERS = "transformers"  # Direct HuggingFace Transformers
    VLLM = "vllm"  # vLLM server (faster, batching)


class AccuracyMode(str, Enum):
    """Accuracy mode for document processing."""

    FAST = "fast"  # DeepSeek-OCR only (fastest)
    BALANCED = "balanced"  # Hybrid routing (recommended)
    MAXIMUM = "maximum"  # TableFormer for all tables (slowest, most accurate)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # GPU Configuration
    cuda_visible_devices: str = Field(default="0", description="CUDA device IDs")
    torch_device: str = Field(default="cuda", description="PyTorch device")
    vram_limit_gb: int = Field(default=22, description="VRAM limit in GB (3090: 22/24)")

    # Inference Mode
    inference_mode: InferenceMode = Field(
        default=InferenceMode.TRANSFORMERS,
        description="Inference backend: transformers or vllm",
    )

    # vLLM Configuration
    vllm_deepseek_url: str = Field(
        default="http://localhost:8000/v1/chat/completions",
        description="DeepSeek-OCR vLLM endpoint",
    )
    vllm_nanonets_url: str = Field(
        default="http://localhost:8001/v1/chat/completions",
        description="Nanonets-OCR2-3B vLLM endpoint",
    )
    vllm_granite_url: str = Field(
        default="http://localhost:8002/v1/chat/completions",
        description="Granite-Docling vLLM endpoint",
    )
    vllm_api_key: str = Field(default="EMPTY", description="vLLM API key")

    # Model Paths
    deepseek_model: str = Field(
        default="deepseek-ai/DeepSeek-OCR",
        description="DeepSeek-OCR model ID",
    )
    nanonets_model: str = Field(
        default="nanonets/Nanonets-OCR2-3B",
        description="Nanonets-OCR2-3B model ID",
    )
    granite_model: str = Field(
        default="ibm-granite/granite-docling-258m",
        description="Granite-Docling model ID",
    )

    # Processing Configuration
    default_accuracy_mode: AccuracyMode = Field(
        default=AccuracyMode.BALANCED,
        description="Default accuracy mode",
    )
    default_batch_size: int = Field(default=4, description="Default batch size")
    max_pages_per_batch: int = Field(
        default=500,
        description="Maximum pages per batch",
    )
    enable_signature_detection: bool = Field(
        default=True,
        description="Enable signature detection (Nanonets)",
    )
    enable_handwriting_detection: bool = Field(
        default=True,
        description="Enable handwriting detection (Nanonets)",
    )
    enable_enrichment: bool = Field(
        default=False,
        description="Enable Granite semantic enrichment (slower but better structure)",
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

    def model_post_init(self, __context):
        """Create directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
