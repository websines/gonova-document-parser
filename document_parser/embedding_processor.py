"""Embedding generation using Qwen3-Embedding-0.6B.

Generates high-quality embeddings for document nodes to enable
semantic search and retrieval in graph-vector databases.
"""

import time
from typing import Dict, List, Optional

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingProcessor:
    """
    Qwen3-Embedding-0.6B processor for document embeddings.

    Features:
    - 8192 context length (handles long documents)
    - Multi-lingual support
    - 1536 dimensions
    - Batch processing for efficiency
    - GPU/CPU support

    Model: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 8192,
    ):
        """
        Initialize embedding processor.

        Args:
            model_name: HuggingFace model ID
            device: Device to use (cuda/cpu), auto-detected if None
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[SentenceTransformer] = None

        logger.info(
            f"Initialized EmbeddingProcessor: {model_name}, device={self.device}"
        )

    def load_model(self):
        """Load embedding model."""
        if self.model is not None:
            logger.debug("Model already loaded")
            return

        logger.info(f"Loading {self.model_name}...")
        start_time = time.time()

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )

            # Set max sequence length
            self.model.max_seq_length = self.max_length

            elapsed = time.time() - start_time
            logger.success(f"Embedding model loaded in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Embedding model unloaded")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector (1536 dimensions)
        """
        if self.model is None:
            self.load_model()

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=1,
        )

        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batched).

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.load_model()

        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} texts in batches of {self.batch_size}")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )

        return embeddings.tolist()

    def embed_nodes(
        self,
        nodes: List[Dict],
        content_field: str = "content",
    ) -> List[Dict]:
        """
        Add embeddings to document nodes.

        Args:
            nodes: List of document nodes
            content_field: Field name containing text to embed

        Returns:
            Nodes with added 'embedding' field
        """
        if not nodes:
            return nodes

        logger.info(f"Generating embeddings for {len(nodes)} nodes...")
        start_time = time.time()

        # Extract texts to embed
        texts = []
        for node in nodes:
            text = node.get(content_field, "")
            if isinstance(text, str):
                texts.append(text)
            else:
                texts.append(str(text))

        # Generate embeddings in batches
        embeddings = self.embed_texts(texts)

        # Add embeddings to nodes
        for node, embedding in zip(nodes, embeddings):
            node["embedding"] = embedding

        elapsed = time.time() - start_time
        nodes_per_sec = len(nodes) / elapsed if elapsed > 0 else 0

        logger.success(
            f"Generated {len(nodes)} embeddings in {elapsed:.1f}s "
            f"({nodes_per_sec:.1f} nodes/sec)"
        )

        return nodes

    def get_embedding_dim(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension (1536 for Qwen3-Embedding-0.6B)
        """
        if self.model is None:
            self.load_model()

        return self.model.get_sentence_embedding_dimension()

    def get_stats(self) -> Dict:
        """
        Get processor statistics.

        Returns:
            Dict with model stats
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "embedding_dim": self.get_embedding_dim() if self.model else None,
            "model_loaded": self.model is not None,
        }


# Convenience function for quick embedding
def embed_document_nodes(
    nodes: List[Dict],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    device: Optional[str] = None,
) -> List[Dict]:
    """
    Quick function to embed document nodes.

    Args:
        nodes: Document nodes
        model_name: Embedding model to use
        device: Device (cuda/cpu)

    Returns:
        Nodes with embeddings

    Example:
        ```python
        nodes = [
            {"id": "node_0", "content": "Section 1 text..."},
            {"id": "node_1", "content": "Section 2 text..."},
        ]
        nodes_with_embeddings = embed_document_nodes(nodes)
        ```
    """
    processor = EmbeddingProcessor(model_name=model_name, device=device)
    return processor.embed_nodes(nodes)
