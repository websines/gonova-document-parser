"""Nanonets-OCR2-3B processor with VQA, handwriting, and signature detection."""

import time
from typing import Any, Dict, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from loguru import logger

from document_parser.base_processor import BaseProcessor, ProcessingResult
from document_parser.config import InferenceMode


class NanonetsProcessor(BaseProcessor):
    """
    Nanonets-OCR2-3B processor with specialized capabilities:
    - Handwriting recognition (11 languages)
    - Signature detection
    - Checkbox recognition
    - VQA (Visual Question Answering)
    - Financial document optimization
    """

    DEFAULT_PROMPT = "Convert this document to high-quality markdown format. Preserve all text including handwritten content, signatures, checkboxes, tables with correct alignment, and maintain precise formatting. Use proper markdown syntax for headings, lists, emphasis, and tables."

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Nanonets-OCR2-3B processor.

        Args:
            config: Configuration dict with keys:
                - inference_mode: "transformers" or "vllm"
                - model_id: HuggingFace model ID
                - device: "cuda" or "cpu"
                - vllm_url: vLLM endpoint (if using vllm mode)
                - detect_signatures: Enable signature detection
                - detect_handwriting: Enable handwriting detection
        """
        super().__init__(config)
        self.inference_mode = config.get("inference_mode", InferenceMode.TRANSFORMERS)
        self.model_id = config.get("model_id", "nanonets/Nanonets-OCR2-3B")
        self.device = config.get("device", "cuda")
        self.vllm_url = config.get("vllm_url")
        self.detect_signatures = config.get("detect_signatures", True)
        self.detect_handwriting = config.get("detect_handwriting", True)
        self.converter = None
        self.vqa_model = None  # Separate model for VQA if needed

        logger.info(
            f"Initialized Nanonets processor: {self.inference_mode}, "
            f"signatures={self.detect_signatures}, handwriting={self.detect_handwriting}"
        )

    def load_model(self):
        """Load Nanonets-OCR2-3B model via Docling."""
        if self.converter is not None:
            logger.debug("Model already loaded")
            return

        logger.info(f"Loading Nanonets-OCR2-3B ({self.inference_mode})...")

        if self.inference_mode == InferenceMode.VLLM:
            # vLLM server mode - optimized for handwriting and forms
            vlm_options = ApiVlmOptions(
                url=self.vllm_url,
                params={
                    "model": self.model_id,
                    "max_tokens": 15000,  # Nanonets supports longer contexts
                    "temperature": 0.0,  # Deterministic for accuracy
                    "top_p": 0.95,  # Higher for better handwriting recognition
                    "repetition_penalty": 1.05,  # Prevent repetition in OCR
                },
                prompt=self.DEFAULT_PROMPT,
                response_format=ResponseFormat.MARKDOWN,
                temperature=0.0,
                timeout=600,  # 10 minute timeout for slow models/large docs
            )
            logger.info(f"Using vLLM server at {self.vllm_url} (optimized for handwriting/forms, timeout=600s)")
        else:
            # Direct Transformers mode
            vlm_options = InlineVlmOptions(
                repo_id=self.model_id,
                prompt=self.DEFAULT_PROMPT,
                response_format=ResponseFormat.MARKDOWN,
                inference_framework=InferenceFramework.TRANSFORMERS,
                temperature=0.0,
                max_new_tokens=15000,
                supported_devices=["cuda", "mps", "cpu"],
            )
            logger.info(f"Using Transformers with model {self.model_id}")

        # OPTIMIZATION: Enable batching for vLLM
        # Nanonets is slower but handles complex content (handwriting, forms)
        # Use moderate batch size to balance speed and quality
        if self.inference_mode == InferenceMode.VLLM:
            from docling.datamodel.settings import settings as docling_settings
            docling_settings.perf.page_batch_size = 16  # Moderate batching (Nanonets is heavier)

            pipeline_options = VlmPipelineOptions(
                vlm_options=vlm_options,
                enable_remote_services=True,
                concurrency=16,  # Process 16 pages concurrently
            )
        else:
            pipeline_options = VlmPipelineOptions(
                vlm_options=vlm_options,
                enable_remote_services=False,
            )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

        logger.success("Nanonets-OCR2-3B model loaded successfully")

    def process(
        self,
        pdf_path: str,
        output_format: str = "markdown",
        vqa_questions: Optional[List[str]] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process PDF with Nanonets-OCR2-3B.

        Args:
            pdf_path: Path to PDF file
            output_format: Output format (markdown, json, html)
            vqa_questions: Optional list of questions for VQA
            **kwargs: Additional options

        Returns:
            ProcessingResult with processed document and VQA answers
        """
        if self.converter is None:
            self.load_model()

        logger.info(f"Processing {pdf_path} with Nanonets-OCR2-3B...")
        start_time = time.time()

        try:
            # Convert document
            result = self.converter.convert(pdf_path)

            # Export in requested format
            if output_format == "markdown":
                output = result.document.export_to_markdown()
            elif output_format == "json":
                output = result.document.export_to_json()
            elif output_format == "html":
                output = result.document.export_to_html()
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            # Extract metadata
            num_pages = len(result.document.pages)
            elapsed = time.time() - start_time
            pages_per_sec = num_pages / elapsed if elapsed > 0 else 0

            # Extract nodes and edges
            nodes = self._extract_nodes(result.document)
            edges = self._extract_edges(result.document)

            # Detect signatures and handwriting
            signatures_found = self._detect_signatures(output) if self.detect_signatures else []
            handwritten_pages = (
                self._detect_handwriting(output) if self.detect_handwriting else []
            )

            # VQA processing if questions provided
            vqa_answers = {}
            if vqa_questions:
                logger.info(f"Processing {len(vqa_questions)} VQA questions...")
                vqa_answers = self._process_vqa(pdf_path, vqa_questions)

            logger.success(
                f"Processed {num_pages} pages in {elapsed:.1f}s ({pages_per_sec:.2f} pages/sec)"
            )

            return ProcessingResult(
                success=True,
                output=output,
                nodes=nodes,
                edges=edges,
                metadata={
                    "processor": "nanonets-ocr2-3b",
                    "num_pages": num_pages,
                    "processing_time": elapsed,
                    "pages_per_second": pages_per_sec,
                    "output_format": output_format,
                    "signatures_found": signatures_found,
                    "handwritten_pages": handwritten_pages,
                    "table_count": len([n for n in nodes if n["type"] == "table"]),
                    "vqa_answers": vqa_answers,
                },
            )

        except Exception as e:
            logger.error(f"Nanonets processing failed: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                metadata={"processor": "nanonets-ocr2-3b"},
            )

    def _extract_nodes(self, document) -> list:
        """Extract graph nodes from Docling document using iterate_items()."""
        nodes = []
        for idx, (item, level) in enumerate(document.iterate_items()):
            node = {
                "id": f"node_{idx}",
                "type": str(item.label) if hasattr(item, "label") else "unknown",
                "content": item.text if hasattr(item, "text") else str(item),
                "page": item.prov[0].page_no if hasattr(item, "prov") and item.prov else None,
                "level": level,
            }

            # Add type-specific attributes
            if hasattr(item, "text"):
                node["text"] = item.text
            if hasattr(item, "data"):
                node["data"] = item.data
            if hasattr(item, "caption"):
                node["caption"] = item.caption

            nodes.append(node)

        return nodes

    def _extract_edges(self, document) -> list:
        """Extract graph edges from Docling document."""
        edges = []

        # Sequential relationships
        items = list(document.iterate_items())
        for i in range(len(items) - 1):
            edges.append(
                {
                    "source": f"node_{i}",
                    "target": f"node_{i+1}",
                    "type": "follows",
                }
            )

        return edges

    def _detect_signatures(self, markdown_output: str) -> List[str]:
        """
        Detect signatures in document.

        Nanonets includes signature markers in output.
        This is a simple heuristic - can be enhanced with ML.
        """
        signatures = []
        lines = markdown_output.split("\n")

        for i, line in enumerate(lines):
            # Look for signature-related keywords
            if any(
                keyword in line.lower()
                for keyword in ["signature", "signed", "authorized", "signatory"]
            ):
                signatures.append(f"page_{i // 50}")  # Rough page estimation

        return list(set(signatures))  # Deduplicate

    def _detect_handwriting(self, markdown_output: str) -> List[int]:
        """
        Detect pages with handwritten content.

        Nanonets can recognize handwriting - look for specific markers.
        This is a heuristic - can be enhanced.
        """
        # TODO: Implement handwriting detection logic
        # For now, return empty list
        return []

    def _process_vqa(self, pdf_path: str, questions: List[str]) -> Dict[str, str]:
        """
        Process VQA questions on document.

        Args:
            pdf_path: Path to PDF
            questions: List of questions

        Returns:
            Dict mapping questions to answers
        """
        # TODO: Implement VQA using Transformers directly
        # This requires loading the model separately for VQA mode
        logger.warning("VQA processing not yet implemented")
        return {q: "Not implemented" for q in questions}

    def get_capabilities(self) -> Dict[str, bool]:
        """Return Nanonets-OCR2-3B capabilities."""
        return {
            "handwriting": True,
            "signatures": True,
            "vqa": True,
            "tables": True,
            "charts": True,
            "multilingual": True,
            "checkboxes": True,
            "latex_equations": True,
            "fast_processing": False,  # Slower than DeepSeek
        }
