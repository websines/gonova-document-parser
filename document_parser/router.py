"""Intelligent routing logic for document processing."""

from enum import Enum
from typing import Dict, Tuple

from loguru import logger

from document_parser.config import AccuracyMode


class ProcessorType(str, Enum):
    """Available processor types."""

    DEEPSEEK = "deepseek"
    NANONETS = "nanonets"
    GRANITE = "granite"
    DOCLING_TABLEFORMER = "docling_tableformer"  # Future: for maximum accuracy


class DocumentRouter:
    """
    Intelligent router for selecting optimal processor based on document characteristics.

    Routes documents to:
    - DeepSeek-OCR: Fast processing, good tables, standard content
    - Nanonets-OCR2-3B: Handwriting, signatures, forms, VQA
    - Granite-Docling: Enrichment, semantic structure
    - Docling TableFormer: Maximum table accuracy (future)
    """

    def __init__(self, accuracy_mode: AccuracyMode = AccuracyMode.BALANCED):
        """
        Initialize router.

        Args:
            accuracy_mode: fast, balanced, or maximum
        """
        self.accuracy_mode = accuracy_mode
        logger.info(f"Initialized router with accuracy_mode={accuracy_mode}")

    def route(self, analysis: Dict) -> Tuple[ProcessorType, str]:
        """
        Determine which processor to use based on document analysis.

        Args:
            analysis: Document analysis from DocumentAnalyzer

        Returns:
            Tuple of (ProcessorType, reasoning string)
        """
        # Fast mode: Always use DeepSeek
        if self.accuracy_mode == AccuracyMode.FAST:
            return ProcessorType.DEEPSEEK, "Fast mode: DeepSeek-OCR only"

        # Maximum accuracy mode: Use TableFormer (future implementation)
        if self.accuracy_mode == AccuracyMode.MAXIMUM:
            # TODO: Implement Docling StandardPdfPipeline with TableFormer
            logger.warning("Maximum accuracy mode not yet implemented, using balanced")
            # For now, fall through to balanced mode

        # Balanced mode: Intelligent routing
        # Priority 1: Forms/Signatures → Nanonets
        if analysis.get("has_forms", False):
            return (
                ProcessorType.NANONETS,
                "Document contains forms/signatures → Nanonets for detection",
            )

        # Priority 2: High handwriting content → Nanonets
        handwriting_ratio = analysis.get("handwriting_ratio", 0.0)
        if handwriting_ratio > 0.2:  # More than 20% potentially handwritten
            return (
                ProcessorType.NANONETS,
                f"Estimated {handwriting_ratio:.0%} handwritten content → Nanonets",
            )

        # Priority 3: Scanned documents (low native text) → DeepSeek with good OCR
        native_ratio = analysis.get("native_ratio", 1.0)
        if native_ratio < 0.3:  # Mostly scanned
            return (
                ProcessorType.DEEPSEEK,
                f"Mostly scanned ({native_ratio:.0%} native text) → DeepSeek with OCR",
            )

        # Default: Standard native/digital PDF → DeepSeek (fastest)
        return (
            ProcessorType.DEEPSEEK,
            f"Standard document ({native_ratio:.0%} native text) → DeepSeek for speed",
        )

    def should_enrich_with_granite(self, analysis: Dict, primary_processor: ProcessorType) -> bool:
        """
        Determine if document should be enriched with Granite after primary processing.

        Args:
            analysis: Document analysis
            primary_processor: Which processor was used for primary processing

        Returns:
            True if Granite enrichment recommended
        """
        # Enrich if document is complex and we used fast processing
        if primary_processor == ProcessorType.DEEPSEEK:
            # Large documents benefit from semantic structure
            if analysis.get("total_pages", 0) > 100:
                return True

            # Documents with many images/tables benefit from structure
            if analysis.get("has_images", False):
                return True

        return False

    def needs_vqa(self, vqa_questions: list) -> bool:
        """
        Determine if VQA processing is needed.

        Args:
            vqa_questions: List of VQA questions (if any)

        Returns:
            True if VQA should be run
        """
        return bool(vqa_questions)

    def explain_routing(self, analysis: Dict) -> Dict[str, any]:
        """
        Get detailed routing explanation for debugging/logging.

        Args:
            analysis: Document analysis

        Returns:
            Dict with routing details
        """
        primary_processor, reasoning = self.route(analysis)

        return {
            "primary_processor": primary_processor.value,
            "reasoning": reasoning,
            "accuracy_mode": self.accuracy_mode.value,
            "document_stats": {
                "total_pages": analysis.get("total_pages", 0),
                "native_ratio": analysis.get("native_ratio", 0.0),
                "handwriting_ratio": analysis.get("handwriting_ratio", 0.0),
                "has_forms": analysis.get("has_forms", False),
                "has_images": analysis.get("has_images", False),
            },
            "enrichment_recommended": self.should_enrich_with_granite(
                analysis, primary_processor
            ),
        }
