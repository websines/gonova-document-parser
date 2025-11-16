"""DeepSeek-OCR processor implementation."""

import time
from typing import Any, Dict

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


class DeepSeekProcessor(BaseProcessor):
    """
    DeepSeek-OCR processor for fast, efficient document processing.

    Supports multiple resolution modes:
    - base: 100 tokens/page (fastest)
    - gundam: 795 tokens/page (balanced)
    - gundam-m: 1,853 tokens/page (highest accuracy)
    """

    PROMPTS = {
        "base": "<image>\n<|grounding|>Convert to markdown.",
        "gundam": "<image>\n<|grounding|>Convert to markdown with proper formatting. Preserve table structure, headers, and formatting. Use proper markdown syntax for headings, lists, and emphasis.",
        "gundam-m": "<image>\n<|grounding|>Convert to markdown with maximum fidelity. Preserve complex tables with correct alignment, multi-level headers, nested lists, code blocks, and mathematical equations. Maintain document structure and formatting precisely.",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DeepSeek-OCR processor.

        Args:
            config: Configuration dict with keys:
                - inference_mode: "transformers" or "vllm"
                - model_id: HuggingFace model ID
                - resolution_mode: "base", "gundam", or "gundam-m"
                - device: "cuda" or "cpu"
                - vllm_url: vLLM endpoint (if using vllm mode)
        """
        super().__init__(config)
        self.inference_mode = config.get("inference_mode", InferenceMode.TRANSFORMERS)
        self.model_id = config.get("model_id", "deepseek-ai/DeepSeek-OCR")
        self.resolution_mode = config.get("resolution_mode", "gundam")
        self.device = config.get("device", "cuda")
        self.vllm_url = config.get("vllm_url")
        self.converter = None

        logger.info(
            f"Initialized DeepSeek processor: {self.inference_mode}, "
            f"mode={self.resolution_mode}, device={self.device}"
        )

    def load_model(self):
        """Load DeepSeek-OCR model via Docling."""
        if self.converter is not None:
            logger.debug("Model already loaded")
            return

        logger.info(f"Loading DeepSeek-OCR ({self.inference_mode})...")

        if self.inference_mode == InferenceMode.VLLM:
            # vLLM server mode - optimized for quality and speed
            vlm_options = ApiVlmOptions(
                url=self.vllm_url,
                params={
                    "model": self.model_id,
                    "max_tokens": 8192,
                    "temperature": 0.0,  # Deterministic output
                    "top_p": 0.9,  # Slightly higher for better quality
                },
                prompt=self.PROMPTS[self.resolution_mode],
                response_format=ResponseFormat.MARKDOWN,
                temperature=0.0,
                timeout=600,  # 10 minute timeout for large documents
            )
            logger.info(f"Using vLLM server at {self.vllm_url} (optimized for quality, timeout=600s)")
        else:
            # Direct Transformers mode
            vlm_options = InlineVlmOptions(
                repo_id=self.model_id,
                prompt=self.PROMPTS[self.resolution_mode],
                response_format=ResponseFormat.MARKDOWN,
                inference_framework=InferenceFramework.TRANSFORMERS,
                temperature=0.0,
                max_new_tokens=8192,
                supported_devices=["cuda", "mps", "cpu"],
            )
            logger.info(f"Using Transformers with model {self.model_id}")

        # OPTIMIZATION: Enable batching and concurrency for vLLM
        # Based on research: Docling supports page_batch_size and concurrency
        # For vLLM servers, we can process 20-64 pages in parallel
        if self.inference_mode == InferenceMode.VLLM:
            # Use higher batch size for vLLM (server handles batching efficiently)
            from docling.datamodel.settings import settings as docling_settings
            docling_settings.perf.page_batch_size = 32  # Optimal for 3090 (default is 4)

            pipeline_options = VlmPipelineOptions(
                vlm_options=vlm_options,
                enable_remote_services=True,
                concurrency=32,  # Process 32 pages concurrently via vLLM
            )
        else:
            # Lower batch size for local transformers (memory constraints)
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

        logger.success("DeepSeek-OCR model loaded successfully")

    def process(
        self,
        pdf_path: str,
        output_format: str = "markdown",
        **kwargs,
    ) -> ProcessingResult:
        """
        Process PDF with DeepSeek-OCR.

        Args:
            pdf_path: Path to PDF file
            output_format: Output format (markdown, json, html)
            **kwargs: Additional options

        Returns:
            ProcessingResult with processed document
        """
        if self.converter is None:
            self.load_model()

        logger.info(f"Processing {pdf_path} with DeepSeek-OCR ({self.resolution_mode})...")
        start_time = time.time()

        try:
            # Convert document
            result = self.converter.convert(pdf_path)

            # Export in requested format
            if output_format == "markdown":
                output = result.document.export_to_markdown()
            elif output_format == "json":
                import json
                output = json.dumps(result.document.export_to_dict(), indent=2)
            elif output_format == "html":
                output = result.document.export_to_html()
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            # Extract metadata
            num_pages = len(result.document.pages)
            elapsed = time.time() - start_time
            pages_per_sec = num_pages / elapsed if elapsed > 0 else 0

            # Extract nodes and edges for graph DB
            nodes = self._extract_nodes(result.document)
            edges = self._extract_edges(result.document)

            logger.success(
                f"Processed {num_pages} pages in {elapsed:.1f}s ({pages_per_sec:.2f} pages/sec)"
            )

            return ProcessingResult(
                success=True,
                output=output,
                nodes=nodes,
                edges=edges,
                metadata={
                    "processor": "deepseek-ocr",
                    "resolution_mode": self.resolution_mode,
                    "num_pages": num_pages,
                    "processing_time": elapsed,
                    "pages_per_second": pages_per_sec,
                    "output_format": output_format,
                    "table_count": len([n for n in nodes if n["type"] == "table"]),
                },
            )

        except Exception as e:
            logger.error(f"DeepSeek processing failed: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                metadata={"processor": "deepseek-ocr", "resolution_mode": self.resolution_mode},
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
        """Extract graph edges (relationships) from Docling document."""
        edges = []

        # Simple sequential relationships
        items = list(document.iterate_items())
        for i in range(len(items) - 1):
            edges.append(
                {
                    "source": f"node_{i}",
                    "target": f"node_{i+1}",
                    "type": "follows",
                }
            )

        # TODO: Add more sophisticated relationship extraction
        # (e.g., heading-to-content, caption-to-figure)

        return edges

    def get_capabilities(self) -> Dict[str, bool]:
        """Return DeepSeek-OCR capabilities."""
        return {
            "handwriting": False,
            "signatures": False,
            "vqa": False,
            "tables": True,
            "charts": True,
            "multilingual": True,
            "fast_processing": True,
        }
