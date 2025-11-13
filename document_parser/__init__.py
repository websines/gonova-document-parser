"""
Document Parser - Hybrid document processing for financial/legal documents.

Combines DeepSeek-OCR, Nanonets-OCR2-3B, and Granite-Docling for optimal
accuracy and speed. Designed for agentic RAG systems with graph-vector DB integration.
"""

__version__ = "0.1.0"

from document_parser.hybrid_processor import HybridDocumentProcessor
from document_parser.config import Settings

__all__ = ["HybridDocumentProcessor", "Settings"]
