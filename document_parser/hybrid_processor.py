"""
Main hybrid document processor orchestrating all components.

This is the primary interface for the document processing system.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from document_parser.async_processor import AsyncVlmProcessor
from document_parser.config import AccuracyMode, InferenceMode, settings
from document_parser.document_analyzer import DocumentAnalyzer


class GraphDocument(BaseModel):
    """
    Graph-ready document structure for vector-graph DB ingestion.

    Compatible with Neo4j, TigerGraph, and other graph databases.
    """

    document_id: str
    filename: str
    nodes: List[Dict]
    edges: List[Dict]
    metadata: Dict
    vqa_answers: Optional[Dict[str, str]] = None


class HybridDocumentProcessor:
    """
    Hybrid document processor with per-page intelligent routing.

    Uses pure async processing with per-page classification:
    - DeepSeek-OCR: Fast processing for standard pages
    - Nanonets-OCR2-3B: Specialized for signature/form pages

    Features:
    - Per-page classification and routing
    - True parallel processing (16 concurrent per model)
    - No Docling framework overhead
    - 75% faster than sequential processing

    Outputs:
    - Markdown, JSON, HTML formats
    - Graph-ready structure (nodes + edges) for vector-graph DB
    """

    def __init__(
        self,
        accuracy_mode: AccuracyMode = None,
        inference_mode: InferenceMode = None,
        enable_enrichment: bool = False,  # Disabled by default
        enable_embeddings: bool = False,
    ):
        """
        Initialize hybrid processor.

        Args:
            accuracy_mode: fast, balanced, or maximum (default: from settings)
            inference_mode: transformers or vllm (default: from settings)
            enable_enrichment: Enable Granite enrichment layer (optional)
            enable_embeddings: Enable Qwen3 embedding generation (optional)
        """
        self.accuracy_mode = accuracy_mode or settings.default_accuracy_mode
        self.inference_mode = inference_mode or settings.inference_mode
        self.enable_enrichment = enable_enrichment
        self.enable_embeddings = enable_embeddings

        logger.info(
            f"Initializing HybridDocumentProcessor: "
            f"accuracy={self.accuracy_mode}, inference={self.inference_mode}, "
            f"enrichment={enable_enrichment}, embeddings={enable_embeddings}"
        )

        # Components
        self.analyzer = DocumentAnalyzer()

        # Pure async processor for DeepSeek + Nanonets
        self.async_processor = AsyncVlmProcessor(
            deepseek_url=settings.vllm_deepseek_url,
            nanonets_url=settings.vllm_nanonets_url,
            deepseek_model=settings.deepseek_model,
            nanonets_model=settings.nanonets_model,
            concurrency=16,  # Match vLLM --max-num-seqs
            timeout=600,
            max_retries=3,
        )

        # Optional components (lazy loaded)
        self._granite = None
        self._embedding = None

    @property
    def granite(self):
        """Lazy load Granite processor (optional enrichment)."""
        if self._granite is None:
            from document_parser.granite_processor import GraniteProcessor

            self._granite = GraniteProcessor(
                {
                    "inference_mode": self.inference_mode,
                    "model_id": settings.granite_model,
                    "device": settings.torch_device,
                    "vllm_url": settings.vllm_granite_url,
                }
            )
        return self._granite

    @property
    def embedding(self):
        """Lazy load Embedding processor (optional)."""
        if self._embedding is None:
            from document_parser.embedding_processor import EmbeddingProcessor

            self._embedding = EmbeddingProcessor(
                device=settings.torch_device,
                batch_size=32,
            )
        return self._embedding

    def process(
        self,
        pdf_path: str | Path,
        output_format: str = "markdown",
        accuracy_mode: Optional[AccuracyMode] = None,
        vqa_questions: Optional[List[str]] = None,
        extract_signatures: Optional[bool] = None,
    ) -> GraphDocument:
        """
        Process PDF document with per-page routing and parallel processing.

        Args:
            pdf_path: Path to PDF file
            output_format: markdown, json, or html
            accuracy_mode: Override default accuracy mode (deprecated, per-page routing)
            vqa_questions: Optional VQA questions (not supported yet)
            extract_signatures: Override signature detection (deprecated)

        Returns:
            GraphDocument with nodes, edges, and metadata
        """
        pdf_path = Path(pdf_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'='*60}\n")

        # Step 1: Fast document analysis
        logger.info("Step 1: Analyzing document...")
        analysis = self.analyzer.analyze(pdf_path)
        logger.debug(
            f"  Analysis: pages={analysis.get('total_pages')}, "
            f"forms={analysis.get('has_forms')}, "
            f"native_ratio={analysis.get('native_ratio', 0):.1%}"
        )

        # Step 2: Per-page processing with async processor
        logger.info("\nStep 2: Processing with per-page routing...")
        logger.info("  Each page classified individually")
        logger.info("  DeepSeek: Standard pages (fast)")
        logger.info("  Nanonets: Legal pages, forms, signatures (specialized)")

        # Run async processing
        result = asyncio.run(
            self.async_processor.process_pdf(str(pdf_path), output_format)
        )

        # Extract metadata
        metadata = result["metadata"]
        output = result["output"]

        logger.success(
            f"  Processed {metadata['num_pages']} pages in {metadata['processing_time']:.1f}s "
            f"({metadata['pages_per_second']:.2f} pages/sec)"
        )
        logger.info(
            f"  Routing: {metadata['deepseek_pages']} → DeepSeek, "
            f"{metadata['nanonets_pages']} → Nanonets"
        )

        # Step 3: Create graph structure (simple nodes/edges from pages)
        logger.info("\nStep 3: Creating graph-ready output...")
        nodes, edges = self._create_graph_structure(output, metadata["num_pages"])

        # Step 4: Enrichment (if enabled)
        enrichment_applied = False
        if self.enable_enrichment:
            logger.info("\nStep 4: Semantic enrichment with Granite...")
            # TODO: Implement enrichment for async results
            logger.warning("  Enrichment not yet implemented for async processor")
            enrichment_applied = False

        # Step 5: Create graph document
        step_num = 5 if self.enable_enrichment else 4
        logger.info(f"\nStep {step_num}: Finalizing graph document...")
        graph_doc = GraphDocument(
            document_id=pdf_path.stem,
            filename=pdf_path.name,
            nodes=nodes,
            edges=edges,
            metadata={
                **metadata,
                "output": output,
                "accuracy_mode": self.accuracy_mode.value,
                "inference_mode": self.inference_mode.value,
                "enrichment_applied": enrichment_applied,
            },
            vqa_answers=None,  # VQA not yet supported in async processor
        )

        # Step 6: Generate embeddings (if enabled)
        if self.enable_embeddings:
            step_num = 6 if self.enable_enrichment else 5
            logger.info(f"\nStep {step_num}: Generating embeddings with Qwen3...")
            graph_doc.nodes = self.embedding.embed_nodes(graph_doc.nodes)
            logger.info(f"  Generated {len(graph_doc.nodes)} embeddings")
            graph_doc.metadata["embeddings_generated"] = True
            graph_doc.metadata["embedding_dim"] = self.embedding.get_embedding_dim()

        logger.info(f"\n{'='*60}")
        logger.success(f"Processing complete!")
        logger.info(f"  Nodes: {len(graph_doc.nodes)}")
        logger.info(f"  Edges: {len(graph_doc.edges)}")
        if self.enable_embeddings:
            logger.info(f"  Embeddings: {graph_doc.metadata.get('embedding_dim')}D vectors")
        logger.info(f"  Time: {metadata['processing_time']:.1f}s")
        logger.info(f"{'='*60}\n")

        return graph_doc

    def _create_graph_structure(self, output: str, num_pages: int) -> tuple[list, list]:
        """
        Create simple graph structure from markdown output.

        Args:
            output: Markdown output from async processor
            num_pages: Number of pages in document

        Returns:
            Tuple of (nodes, edges) lists
        """
        nodes = []
        edges = []

        # Split output by page separators
        page_separator = "\n\n---\n\n"
        pages = output.split(page_separator)

        # Create a node for each page
        for page_idx, page_content in enumerate(pages):
            node = {
                "id": f"page_{page_idx}",
                "type": "page",
                "content": page_content,
                "page": page_idx + 1,
                "level": 0,
            }
            nodes.append(node)

            # Create sequential edge to next page
            if page_idx < len(pages) - 1:
                edges.append({
                    "source": f"page_{page_idx}",
                    "target": f"page_{page_idx + 1}",
                    "type": "follows",
                })

        return nodes, edges

    def _create_error_document(self, pdf_path: Path, error: str) -> GraphDocument:
        """Create error document."""
        return GraphDocument(
            document_id=pdf_path.stem,
            filename=pdf_path.name,
            nodes=[],
            edges=[],
            metadata={"error": error, "success": False},
        )

    def get_status(self) -> Dict:
        """Get processor status."""
        return {
            "accuracy_mode": self.accuracy_mode.value,
            "inference_mode": self.inference_mode.value,
            "enable_enrichment": self.enable_enrichment,
            "enable_embeddings": self.enable_embeddings,
            "async_processor": {
                "deepseek_url": self.async_processor.deepseek_client.base_url,
                "nanonets_url": self.async_processor.nanonets_client.base_url,
                "concurrency": self.async_processor.concurrency,
            },
            "processors_loaded": {
                "granite": self._granite is not None,
                "embedding": self._embedding is not None,
            },
        }

    def cleanup(self):
        """Cleanup and unload models."""
        logger.info("Cleaning up processors...")
        # Async processor cleanup (vLLM clients are stateless, no cleanup needed)
        logger.info("  AsyncOpenAI clients are stateless (no cleanup needed)")

        # Optional processors
        if self._granite:
            self._granite.unload_model()
        if self._embedding:
            self._embedding.unload_model()
        logger.success("Cleanup complete")
