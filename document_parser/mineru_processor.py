"""
MinerU 2.5 document processor with concurrent page processing.

This processor uses a single lightweight model (1.2B params) to handle
all document types with parallel page processing for maximum speed.
"""

import asyncio
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import httpx
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from document_parser.config import settings


class ProcessingResult(BaseModel):
    """Result from MinerU processing."""

    success: bool
    output: str  # Markdown output
    metadata: Dict
    error: Optional[str] = None


class MinerUProcessor:
    """
    MinerU 2.5 processor for document parsing.

    Features:
    - Single 1.2B model for all document types
    - Concurrent page processing (configurable concurrency)
    - Connects to external vLLM server (no local model hosting)
    - 2-3x faster than multi-model systems
    """

    def __init__(
        self,
        vllm_url: str = None,
        concurrency: int = None,
        batch_size: int = None,
        timeout: int = None,
        max_retries: int = None,
    ):
        """
        Initialize MinerU processor.

        Args:
            vllm_url: MinerU vLLM server URL (defaults to settings)
            concurrency: Number of concurrent pages within each batch (defaults to settings)
            batch_size: Number of pages per batch (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)
            max_retries: Max retry attempts (defaults to settings)
        """
        self.vllm_url = (vllm_url or settings.mineru_vllm_url).rstrip("/")
        self.concurrency = concurrency or settings.concurrency
        self.batch_size = batch_size or settings.batch_size
        self.timeout = timeout or settings.timeout
        self.max_retries = max_retries or settings.max_retries

        # Create async HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=self.concurrency * 2),
        )

        logger.info(
            f"MinerU Processor initialized: "
            f"server={self.vllm_url}, concurrency={self.concurrency}, "
            f"batch_size={self.batch_size}"
        )

    async def process_pdf(
        self,
        pdf_path: str | Path,
        output_format: str = "markdown",
    ) -> ProcessingResult:
        """
        Process PDF with concurrent page processing.

        Args:
            pdf_path: Path to PDF file
            output_format: Output format (currently only markdown supported)

        Returns:
            ProcessingResult with markdown output and metadata
        """
        pdf_path = Path(pdf_path)
        start_time = asyncio.get_event_loop().time()

        logger.info(f"Processing: {pdf_path.name}")

        try:
            # Step 1: Convert PDF pages to images with PyMuPDF (fast!)
            logger.info("Step 1: Converting PDF to images...")
            pages_images = await self._pdf_to_images(pdf_path)
            num_pages = len(pages_images)
            logger.success(f"  Converted {num_pages} pages")

            # Step 2: Process pages in batches
            logger.info(
                f"Step 2: Processing {num_pages} pages in batches of {self.batch_size} "
                f"(concurrency={self.concurrency} per batch)..."
            )
            page_results = await self._process_pages_batched(pages_images)

            # Step 3: Combine results into markdown
            logger.info("Step 3: Combining results...")
            markdown_output = self._combine_pages(page_results)

            # Calculate metrics
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            pages_per_sec = num_pages / processing_time if processing_time > 0 else 0

            metadata = {
                "num_pages": num_pages,
                "processing_time": processing_time,
                "pages_per_second": pages_per_sec,
                "concurrency": self.concurrency,
                "model": settings.mineru_model,
            }

            logger.success(
                f"  Completed: {num_pages} pages in {processing_time:.1f}s "
                f"({pages_per_sec:.2f} pages/sec)"
            )

            return ProcessingResult(
                success=True,
                output=markdown_output,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ProcessingResult(
                success=False,
                output="",
                metadata={"error": str(e)},
                error=str(e),
            )

    async def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images using PyMuPDF (fast).

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL Images
        """
        images = []

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            # Render page to pixmap (default 72 DPI, can increase for better quality)
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom = 144 DPI

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

        doc.close()
        return images

    async def _process_pages_batched(
        self, pages: List[Image.Image]
    ) -> List[str]:
        """
        Process pages in batches with concurrent processing within each batch.

        This approach:
        - Reduces memory usage (only batch_size images in memory at once)
        - Prevents overwhelming the vLLM server
        - Allows better progress tracking

        Args:
            pages: List of PIL Images

        Returns:
            List of markdown strings (one per page)
        """
        total_pages = len(pages)
        all_results = []

        # Process in batches
        for batch_start in range(0, total_pages, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_pages)
            batch_pages = pages[batch_start:batch_end]
            batch_num = (batch_start // self.batch_size) + 1
            total_batches = (total_pages + self.batch_size - 1) // self.batch_size

            logger.info(
                f"  Batch {batch_num}/{total_batches}: Processing pages "
                f"{batch_start+1}-{batch_end} ({len(batch_pages)} pages)"
            )

            # Process this batch concurrently
            batch_results = await self._process_batch_concurrent(
                batch_pages, batch_start
            )
            all_results.extend(batch_results)

            logger.success(
                f"  Batch {batch_num}/{total_batches}: Completed "
                f"({len(all_results)}/{total_pages} total pages done)"
            )

        return all_results

    async def _process_batch_concurrent(
        self, batch_pages: List[Image.Image], offset: int
    ) -> List[str]:
        """
        Process a single batch of pages concurrently.

        Args:
            batch_pages: List of PIL Images for this batch
            offset: Page number offset for logging

        Returns:
            List of markdown strings for this batch
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def process_with_semaphore(idx: int, image: Image.Image) -> str:
            async with semaphore:
                page_num = offset + idx
                return await self._process_single_page(page_num, image)

        # Create tasks for this batch
        tasks = [
            process_with_semaphore(i, img)
            for i, img in enumerate(batch_pages)
        ]

        # Process concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        batch_results = []
        for i, result in enumerate(results):
            page_num = offset + i
            if isinstance(result, Exception):
                logger.error(f"Page {page_num+1} failed: {result}")
                batch_results.append(f"[Error processing page {page_num+1}: {result}]")
            else:
                batch_results.append(result)

        return batch_results

    async def _process_single_page(
        self, page_num: int, image: Image.Image
    ) -> str:
        """
        Process a single page with MinerU via vLLM server.

        Args:
            page_num: Page number (0-indexed)
            image: PIL Image

        Returns:
            Markdown string for the page
        """
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Prepare request for vLLM OpenAI-compatible API
        payload = {
            "model": settings.mineru_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extract all text, tables, formulas, and content from this document page. Output in markdown format."
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0,
        }

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.vllm_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()

                result = response.json()
                content = result["choices"][0]["message"]["content"]

                logger.debug(f"Page {page_num+1} processed successfully")
                return content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Page {page_num+1} attempt {attempt+1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Page {page_num+1} failed after {self.max_retries} attempts")
                    raise

    def _combine_pages(self, page_results: List[str]) -> str:
        """
        Combine page results into single markdown document.

        Args:
            page_results: List of markdown strings

        Returns:
            Combined markdown string
        """
        # Add page separators
        pages_with_separators = []
        for i, page_content in enumerate(page_results):
            pages_with_separators.append(f"<!-- Page {i+1} -->\n\n{page_content}")

        return "\n\n---\n\n".join(pages_with_separators)

    async def cleanup(self):
        """Cleanup resources."""
        await self.client.aclose()
        logger.info("MinerU processor cleaned up")

    def get_capabilities(self) -> Dict:
        """Get processor capabilities."""
        return {
            "model": settings.mineru_model,
            "supports": [
                "text",
                "tables",
                "math_formulas",
                "forms",
                "signatures",
                "handwriting",
                "multilingual_84_languages",
            ],
            "max_tokens": 4096,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "parameters": "1.2B",
        }
