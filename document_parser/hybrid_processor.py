"""
Main hybrid document processor orchestrating all components.

This is the primary interface for the document processing system.
"""

from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from document_parser.base_processor import ProcessingResult
from document_parser.config import AccuracyMode, InferenceMode, settings
from document_parser.deepseek_processor import DeepSeekProcessor
from document_parser.document_analyzer import DocumentAnalyzer
from document_parser.granite_processor import GraniteProcessor
from document_parser.nanonets_processor import NanonetsProcessor
from document_parser.router import DocumentRouter, ProcessorType


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
    Hybrid document processor with intelligent routing.

    Orchestrates:
    - DeepSeek-OCR: Fast, efficient processing for standard content
    - Nanonets-OCR2-3B: Handwriting, signatures, VQA
    - Granite-Docling: Semantic enrichment

    Outputs:
    - Markdown, JSON, HTML formats
    - Graph-ready structure (nodes + edges) for vector-graph DB
    - VQA answers for agent integration
    """

    def __init__(
        self,
        accuracy_mode: AccuracyMode = None,
        inference_mode: InferenceMode = None,
        enable_enrichment: bool = True,
        enable_embeddings: bool = False,
    ):
        """
        Initialize hybrid processor.

        Args:
            accuracy_mode: fast, balanced, or maximum (default: from settings)
            inference_mode: transformers or vllm (default: from settings)
            enable_enrichment: Enable Granite enrichment layer
            enable_embeddings: Enable Qwen3 embedding generation
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
        self.router = DocumentRouter(accuracy_mode=self.accuracy_mode)

        # Processors (lazy loaded)
        self._deepseek = None
        self._nanonets = None
        self._granite = None
        self._embedding = None

    @property
    def deepseek(self) -> DeepSeekProcessor:
        """Lazy load DeepSeek processor."""
        if self._deepseek is None:
            self._deepseek = DeepSeekProcessor(
                {
                    "inference_mode": self.inference_mode,
                    "model_id": settings.deepseek_model,
                    "resolution_mode": "gundam",  # Balanced mode
                    "device": settings.torch_device,
                    "vllm_url": settings.vllm_deepseek_url,
                }
            )
        return self._deepseek

    @property
    def nanonets(self) -> NanonetsProcessor:
        """Lazy load Nanonets processor."""
        if self._nanonets is None:
            self._nanonets = NanonetsProcessor(
                {
                    "inference_mode": self.inference_mode,
                    "model_id": settings.nanonets_model,
                    "device": settings.torch_device,
                    "vllm_url": settings.vllm_nanonets_url,
                    "detect_signatures": settings.enable_signature_detection,
                    "detect_handwriting": settings.enable_handwriting_detection,
                }
            )
        return self._nanonets

    @property
    def granite(self) -> GraniteProcessor:
        """Lazy load Granite processor."""
        if self._granite is None:
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
        """Lazy load Embedding processor."""
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
        Process PDF document with optimized hybrid approach.

        Args:
            pdf_path: Path to PDF file
            output_format: markdown, json, or html
            accuracy_mode: Override default accuracy mode
            vqa_questions: Optional VQA questions for Nanonets
            extract_signatures: Override signature detection setting

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

        # Step 2: Intelligent routing
        accuracy_mode = accuracy_mode or self.accuracy_mode
        router = DocumentRouter(accuracy_mode=accuracy_mode)

        logger.info("Step 2: Determining routing strategy...")
        routing_info = router.explain_routing(analysis)
        logger.info(f"  Primary processor: {routing_info['primary_processor']}")
        logger.info(f"  Reasoning: {routing_info['reasoning']}")
        logger.info(f"  Enrichment: {'enabled' if self.enable_enrichment and routing_info['enrichment_recommended'] else 'disabled'}")

        # Step 3: Primary processing (OPTIMIZED - no duplication)
        logger.info("\nStep 3: Primary processing...")
        primary_processor_type = ProcessorType(routing_info["primary_processor"])

        # OPTIMIZATION: If we need VQA and routing to Nanonets anyway, do it in one pass
        if primary_processor_type == ProcessorType.NANONETS or (vqa_questions and primary_processor_type != ProcessorType.NANONETS):
            # Use Nanonets for everything (OCR + VQA in single pass)
            logger.info("  Using Nanonets (handles both OCR and VQA in single pass)")
            result = self.nanonets.process(
                str(pdf_path),
                output_format=output_format,
                vqa_questions=vqa_questions,
            )
        else:
            # Use DeepSeek for fast processing (no VQA needed)
            logger.info("  Using DeepSeek (fast processing)")
            result = self.deepseek.process(
                str(pdf_path),
                output_format=output_format,
            )

        if not result.success:
            logger.error(f"Primary processing failed: {result.error}")
            return self._create_error_document(pdf_path, result.error)

        logger.success(f"  Processed {result.metadata.get('num_pages', 0)} pages in {result.metadata.get('processing_time', 0):.1f}s")
        if vqa_questions and result.metadata.get("vqa_answers"):
            logger.success(f"  Answered {len(vqa_questions)} VQA questions")

        # Step 4: Enrichment (if recommended) - SKIP if disabled globally
        if self.enable_enrichment and routing_info["enrichment_recommended"]:
            logger.info("\nStep 4: Semantic enrichment with Granite...")
            result = self.granite.enrich(result)
            logger.info("  Enrichment completed")

        # Step 5: Create graph document
        logger.info(f"\n{'Step 5' if self.enable_enrichment and routing_info['enrichment_recommended'] else 'Step 4'}: Creating graph-ready output...")
        graph_doc = self._create_graph_document(pdf_path, result, routing_info)

        # Step 6: Generate embeddings (if enabled)
        if self.enable_embeddings:
            step_num = 6 if self.enable_enrichment and routing_info["enrichment_recommended"] else 5
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
        logger.info(f"  Time: {result.metadata.get('processing_time', 0):.1f}s")
        logger.info(f"{'='*60}\n")

        return graph_doc

    def _create_graph_document(
        self,
        pdf_path: Path,
        result: ProcessingResult,
        routing_info: Dict,
    ) -> GraphDocument:
        """Create graph-ready document structure."""
        return GraphDocument(
            document_id=pdf_path.stem,
            filename=pdf_path.name,
            nodes=result.nodes or [],
            edges=result.edges or [],
            metadata={
                **result.metadata,
                "routing_info": routing_info,
                "accuracy_mode": self.accuracy_mode.value,
                "inference_mode": self.inference_mode.value,
            },
            vqa_answers=result.metadata.get("vqa_answers"),
        )

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
            "processors_loaded": {
                "deepseek": self._deepseek is not None,
                "nanonets": self._nanonets is not None,
                "granite": self._granite is not None,
                "embedding": self._embedding is not None,
            },
        }

    def cleanup(self):
        """Cleanup and unload models."""
        logger.info("Cleaning up processors...")
        if self._deepseek:
            self._deepseek.unload_model()
        if self._nanonets:
            self._nanonets.unload_model()
        if self._granite:
            self._granite.unload_model()
        if self._embedding:
            self._embedding.unload_model()
        logger.success("Cleanup complete")
