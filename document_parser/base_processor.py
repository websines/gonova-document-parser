"""Base processor abstract class for document processing."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ProcessingResult(BaseModel):
    """Result from document processing."""

    success: bool
    output: Optional[str] = None
    nodes: Optional[list] = None
    edges: Optional[list] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None


class BaseProcessor(ABC):
    """Abstract base class for all document processors."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize processor with configuration.

        Args:
            config: Processor-specific configuration
        """
        self.config = config
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """Load the model and processor. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def process(
        self,
        pdf_path: str,
        output_format: str = "markdown",
        **kwargs,
    ) -> ProcessingResult:
        """
        Process a PDF document.

        Args:
            pdf_path: Path to PDF file
            output_format: Output format (markdown, json, html)
            **kwargs: Additional processing options

        Returns:
            ProcessingResult with output and metadata
        """
        pass

    def unload_model(self):
        """Unload model from memory (optional, for memory management)."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        # Force garbage collection
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities of this processor.

        Returns:
            Dict with capability flags (e.g., handwriting, signatures, vqa)
        """
        pass
