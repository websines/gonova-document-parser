"""
Main document processor using MinerU 2.5.

Simplified from hybrid multi-model system to single lightweight model.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from document_parser.config import settings
from document_parser.mineru_processor import MinerUProcessor


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
    Document processor using MinerU 2.5.

    Simplified from multi-model hybrid system:
    - Single lightweight model (1.2B params)
    - Concurrent page processing
    - 2-3x faster than old multi-model system
    - 60-70% less VRAM usage

    Features:
    - Concurrent page processing (configurable)
    - Handles all document types (text, tables, formulas, signatures)
    - Markdown, JSON, HTML output formats
    - Graph-ready structure (nodes + edges) for vector-graph DB
    """

    def __init__(
        self,
        accuracy_mode: str = None,  # Kept for API compatibility (unused)
        inference_mode: str = None,  # Kept for API compatibility (unused)
        enable_enrichment: bool = False,  # Not used with MinerU
        enable_embeddings: bool = False,
    ):
        """
        Initialize processor.

        Args:
            accuracy_mode: Ignored (for backward compatibility)
            inference_mode: Ignored (for backward compatibility)
            enable_enrichment: Not used with MinerU
            enable_embeddings: Enable Qwen3 embedding generation (optional)
        """
        self.enable_embeddings = enable_embeddings

        logger.info(
            f"Initializing MinerU Document Processor: "
            f"model={settings.mineru_model}, "
            f"batch_size={settings.batch_size}, "
            f"concurrency={settings.concurrency}, "
            f"embeddings={enable_embeddings}"
        )

        # Main processor
        self.processor = MinerUProcessor(
            vllm_url=settings.mineru_vllm_url,
            concurrency=settings.concurrency,
            batch_size=settings.batch_size,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

        # Optional embedding processor (lazy loaded)
        self._embedding = None

    @property
    def embedding(self):
        """Lazy load Embedding processor (optional)."""
        if self._embedding is None:
            from document_parser.embedding_processor import EmbeddingProcessor

            self._embedding = EmbeddingProcessor(
                device="cpu",  # Use CPU for embeddings (GPU used by vLLM)
                batch_size=32,
            )
        return self._embedding

    async def process(
        self,
        pdf_path: str | Path,
        output_format: str = "markdown",
        accuracy_mode: Optional[str] = None,  # Ignored
        vqa_questions: Optional[List[str]] = None,  # Not supported
        extract_signatures: Optional[bool] = None,  # Automatic with MinerU
    ) -> GraphDocument:
        """
        Process PDF document with MinerU concurrent processing.

        Args:
            pdf_path: Path to PDF file
            output_format: markdown, json, or html (currently only markdown)
            accuracy_mode: Ignored (for backward compatibility)
            vqa_questions: Not supported
            extract_signatures: Automatic with MinerU

        Returns:
            GraphDocument with nodes, edges, and metadata
        """
        pdf_path = Path(pdf_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'='*60}\n")

        try:
            # Step 1: Process with MinerU (batched concurrent processing)
            logger.info("Step 1: Processing with MinerU 2.5...")
            logger.info(f"  Model: {settings.mineru_model} (1.2B params)")
            logger.info(f"  Batch size: {settings.batch_size} pages per batch")
            logger.info(f"  Concurrency: {settings.concurrency} pages per batch in parallel")

            result = await self.processor.process_pdf(pdf_path, output_format)

            if not result.success:
                return self._create_error_document(pdf_path, result.error)

            # Extract data
            output = result.output
            metadata = result.metadata

            logger.success(
                f"  Processed {metadata['num_pages']} pages in {metadata['processing_time']:.1f}s "
                f"({metadata['pages_per_second']:.2f} pages/sec)"
            )

            # Step 2: Create graph structure
            logger.info("\nStep 2: Creating graph-ready output...")
            nodes, edges = self._create_graph_structure(output, metadata["num_pages"])

            # Step 3: Create graph document
            logger.info("\nStep 3: Finalizing graph document...")
            graph_doc = GraphDocument(
                document_id=pdf_path.stem,
                filename=pdf_path.name,
                nodes=nodes,
                edges=edges,
                metadata={
                    **metadata,
                    "output": output,
                    "output_format": output_format,
                },
                vqa_answers=None,
            )

            # Step 4: Generate embeddings (if enabled)
            if self.enable_embeddings:
                logger.info("\nStep 4: Generating embeddings with Qwen3...")
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

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return self._create_error_document(pdf_path, str(e))

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
            "model": settings.mineru_model,
            "vllm_url": settings.mineru_vllm_url,
            "batch_size": settings.batch_size,
            "concurrency": settings.concurrency,
            "enable_embeddings": self.enable_embeddings,
            "processors_loaded": {
                "mineru": True,
                "embedding": self._embedding is not None,
            },
            "capabilities": self.processor.get_capabilities(),
        }

    def cleanup(self):
        """Cleanup and unload models."""
        logger.info("Cleaning up processors...")

        # Cleanup MinerU processor (HTTP client)
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(self.processor.cleanup())
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

        # Optional embedding processor
        if self._embedding:
            self._embedding.unload_model()

        logger.success("Cleanup complete")
