"""Pure async PDF processor with per-page routing to vLLM models."""

import asyncio
import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger
from openai import AsyncOpenAI
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader

from document_parser.page_classifier import PageClassifier, ProcessorType


class AsyncVlmProcessor:
    """
    Pure async processor for DeepSeek and Nanonets with per-page routing.

    Features:
    - Per-page classification and routing
    - True 16-concurrent requests per model (vLLM --max-num-seqs)
    - No Docling framework overhead
    - Retry logic and timeout handling
    """

    def __init__(
        self,
        deepseek_url: str,
        nanonets_url: str,
        deepseek_model: str,
        nanonets_model: str,
        concurrency: int = 16,
        timeout: int = 600,
        max_retries: int = 3,
    ):
        """
        Initialize async processor.

        Args:
            deepseek_url: vLLM endpoint for DeepSeek
            nanonets_url: vLLM endpoint for Nanonets
            deepseek_model: DeepSeek model name
            nanonets_model: Nanonets model name
            concurrency: Max concurrent requests per model
            timeout: Request timeout in seconds
            max_retries: Max retry attempts
        """
        self.deepseek_client = AsyncOpenAI(base_url=deepseek_url, api_key="dummy")
        self.nanonets_client = AsyncOpenAI(base_url=nanonets_url, api_key="dummy")
        self.deepseek_model = deepseek_model
        self.nanonets_model = nanonets_model
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.classifier = PageClassifier()

        logger.info(f"Initialized AsyncVlmProcessor (concurrency={concurrency})")

    async def process_pdf(
        self, pdf_path: str | Path, output_format: str = "markdown"
    ) -> Dict:
        """
        Process PDF with per-page classification and routing.

        Args:
            pdf_path: Path to PDF file
            output_format: Output format (markdown, json, html)

        Returns:
            Dict with output, metadata, routing stats
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()

        logger.info(f"Processing {pdf_path.name} with per-page routing...")

        # Step 1: Convert PDF to images (thread pool to avoid blocking)
        logger.info("Step 1: Converting PDF to images...")
        images = await asyncio.to_thread(
            convert_from_path, str(pdf_path), dpi=200, thread_count=4
        )
        logger.info(f"  Converted {len(images)} pages to images")

        # Step 2: Classify each page (fast, < 0.5s total)
        logger.info("Step 2: Classifying pages...")
        reader = PdfReader(str(pdf_path))
        classifications = self.classifier.classify_all_pages(reader, pdf_path.name)

        # Step 3: Group pages by model
        logger.info("Step 3: Grouping pages by model...")
        deepseek_pages: List[Tuple[int, Image.Image]] = []
        nanonets_pages: List[Tuple[int, Image.Image]] = []

        for idx, (image, classification) in enumerate(zip(images, classifications)):
            if classification == "deepseek":
                deepseek_pages.append((idx, image))
            else:  # nanonets
                nanonets_pages.append((idx, image))

        logger.info(
            f"  {len(deepseek_pages)} pages → DeepSeek, {len(nanonets_pages)} pages → Nanonets"
        )

        # Step 4: Process batches in parallel
        logger.info("Step 4: Processing pages in parallel...")
        deepseek_results, nanonets_results = await asyncio.gather(
            self._process_batch(
                deepseek_pages, self.deepseek_client, self.deepseek_model, "DeepSeek"
            ),
            self._process_batch(
                nanonets_pages, self.nanonets_client, self.nanonets_model, "Nanonets"
            ),
        )

        # Step 5: Merge results in page order
        logger.info("Step 5: Merging results...")
        all_results = {**deepseek_results, **nanonets_results}
        final_output = self._merge_pages(all_results, output_format)

        elapsed = time.time() - start_time
        pages_per_sec = len(images) / elapsed if elapsed > 0 else 0

        logger.success(
            f"Processed {len(images)} pages in {elapsed:.1f}s ({pages_per_sec:.2f} pages/sec)"
        )

        return {
            "output": final_output,
            "metadata": {
                "num_pages": len(images),
                "processing_time": elapsed,
                "pages_per_second": pages_per_sec,
                "output_format": output_format,
                "deepseek_pages": len(deepseek_pages),
                "nanonets_pages": len(nanonets_pages),
                "routing_accuracy": "per-page",
            },
        }

    async def _process_batch(
        self,
        pages: List[Tuple[int, Image.Image]],
        client: AsyncOpenAI,
        model: str,
        model_name: str,
    ) -> Dict[int, str]:
        """
        Process batch of pages with one model (16 concurrent).

        Args:
            pages: List of (page_index, image) tuples
            client: AsyncOpenAI client
            model: Model name
            model_name: Human-readable model name for logging

        Returns:
            Dict mapping page_index to markdown output
        """
        if not pages:
            return {}

        logger.info(f"  {model_name}: Processing {len(pages)} pages...")

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.concurrency)

        async def process_one_page(page_idx: int, image: Image.Image) -> Tuple[int, str]:
            async with semaphore:
                for attempt in range(self.max_retries):
                    try:
                        # Convert image to base64
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()

                        # Send to vLLM
                        response = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/png;base64,{img_base64}"
                                                },
                                            },
                                            {
                                                "type": "text",
                                                "text": "Convert this page to markdown. Preserve tables, headings, lists, and text formatting accurately.",
                                            },
                                        ],
                                    }
                                ],
                                max_tokens=4096,
                                temperature=0.1,
                                top_p=0.95,
                                repetition_penalty=1.1,  # Prevent repetition loops
                            ),
                            timeout=self.timeout,
                        )

                        markdown = response.choices[0].message.content
                        logger.debug(f"    {model_name} page {page_idx + 1} completed")
                        return (page_idx, markdown)

                    except asyncio.TimeoutError:
                        if attempt == self.max_retries - 1:
                            logger.error(
                                f"    {model_name} page {page_idx + 1} timed out after {self.timeout}s"
                            )
                            return (page_idx, f"[ERROR: Page {page_idx + 1} timed out]")
                        logger.warning(
                            f"    {model_name} page {page_idx + 1} timeout, retrying ({attempt + 1}/{self.max_retries})..."
                        )
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff

                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            logger.error(
                                f"    {model_name} page {page_idx + 1} failed: {e}"
                            )
                            return (page_idx, f"[ERROR: Page {page_idx + 1} - {str(e)}]")
                        logger.warning(
                            f"    {model_name} page {page_idx + 1} error, retrying: {e}"
                        )
                        await asyncio.sleep(2 ** attempt)

        # Process all pages concurrently
        tasks = [process_one_page(idx, img) for idx, img in pages]
        results = await asyncio.gather(*tasks)

        logger.info(f"  {model_name}: Completed {len(results)} pages")

        return dict(results)

    def _merge_pages(self, results: Dict[int, str], output_format: str) -> str:
        """
        Merge page results in correct order.

        Args:
            results: Dict mapping page_index to markdown
            output_format: Output format (currently only markdown supported)

        Returns:
            Combined output
        """
        # Sort by page index
        sorted_pages = sorted(results.items())

        # Combine with page separators
        combined = []
        for page_idx, content in sorted_pages:
            # Add page separator
            combined.append(f"<!-- Page {page_idx + 1} -->\n\n{content}")

        return "\n\n---\n\n".join(combined)
