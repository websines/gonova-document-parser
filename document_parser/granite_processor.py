"""Granite-Docling processor for semantic structure enrichment."""

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


class GraniteProcessor(BaseProcessor):
    """
    Granite-Docling-258M processor for semantic enrichment.

    Small but powerful model (258M params) specialized for:
    - Table structure (TEDS 0.97)
    - Code blocks (F1 0.988)
    - Equation recognition
    - Semantic structure preservation
    """

    DEFAULT_PROMPT = "Convert the document to markdown with semantic structure preservation."

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Granite-Docling processor.

        Args:
            config: Configuration dict with keys:
                - inference_mode: "transformers" or "vllm"
                - model_id: HuggingFace model ID
                - device: "cuda" or "cpu"
                - vllm_url: vLLM endpoint (if using vllm mode)
        """
        super().__init__(config)
        self.inference_mode = config.get("inference_mode", InferenceMode.TRANSFORMERS)
        self.model_id = config.get("model_id", "ibm-granite/granite-docling-258m")
        self.device = config.get("device", "cuda")
        self.vllm_url = config.get("vllm_url")
        self.converter = None

        logger.info(
            f"Initialized Granite processor: {self.inference_mode}, device={self.device}"
        )

    def load_model(self):
        """Load Granite-Docling model."""
        if self.converter is not None:
            logger.debug("Model already loaded")
            return

        logger.info(f"Loading Granite-Docling-258M ({self.inference_mode})...")

        if self.inference_mode == InferenceMode.VLLM:
            vlm_options = ApiVlmOptions(
                url=self.vllm_url,
                params={"model": self.model_id, "max_tokens": 8192},
                prompt=self.DEFAULT_PROMPT,
                response_format=ResponseFormat.MARKDOWN,
                temperature=0.0,
            )
        else:
            vlm_options = InlineVlmOptions(
                repo_id=self.model_id,
                prompt=self.DEFAULT_PROMPT,
                response_format=ResponseFormat.MARKDOWN,
                inference_framework=InferenceFramework.TRANSFORMERS,
                temperature=0.0,
                max_new_tokens=8192,
                supported_devices=["cuda", "mps", "cpu"],
            )

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=(self.inference_mode == InferenceMode.VLLM),
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

        logger.success("Granite-Docling model loaded successfully")

    def enrich(self, processing_result: ProcessingResult) -> ProcessingResult:
        """
        Enrich an existing processing result with semantic structure.

        Args:
            processing_result: Result from primary processor

        Returns:
            Enriched ProcessingResult
        """
        logger.info("Enriching document with Granite-Docling semantic analysis...")

        # TODO: Implement enrichment logic
        # For now, just pass through
        logger.warning("Granite enrichment not yet implemented")
        return processing_result

    def process(
        self,
        pdf_path: str,
        output_format: str = "markdown",
        **kwargs,
    ) -> ProcessingResult:
        """
        Process PDF with Granite-Docling.

        Args:
            pdf_path: Path to PDF file
            output_format: Output format
            **kwargs: Additional options

        Returns:
            ProcessingResult
        """
        if self.converter is None:
            self.load_model()

        logger.info(f"Processing {pdf_path} with Granite-Docling...")
        start_time = time.time()

        try:
            result = self.converter.convert(pdf_path)

            if output_format == "markdown":
                output = result.document.export_to_markdown()
            elif output_format == "json":
                output = result.document.export_to_json()
            elif output_format == "html":
                output = result.document.export_to_html()
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            num_pages = len(result.document.pages)
            elapsed = time.time() - start_time

            return ProcessingResult(
                success=True,
                output=output,
                metadata={
                    "processor": "granite-docling-258m",
                    "num_pages": num_pages,
                    "processing_time": elapsed,
                },
            )

        except Exception as e:
            logger.error(f"Granite processing failed: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                metadata={"processor": "granite-docling-258m"},
            )

    def get_capabilities(self) -> Dict[str, bool]:
        """Return Granite-Docling capabilities."""
        return {
            "handwriting": False,
            "signatures": False,
            "vqa": False,
            "tables": True,
            "charts": False,
            "code_blocks": True,
            "equations": True,
            "semantic_structure": True,
            "fast_processing": True,  # Small model, fast
        }
