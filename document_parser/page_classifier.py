"""Per-page document classification for intelligent routing."""

from pathlib import Path
from typing import Literal

from loguru import logger
from pypdf import PageObject

ProcessorType = Literal["deepseek", "nanonets"]


class PageClassifier:
    """
    Fast per-page classification using pypdf + heuristics.

    Routes pages to:
    - DeepSeek: Fast processing for standard text, tables, charts
    - Nanonets: Specialized for signatures, forms, handwriting
    """

    # Legal/compliance keywords that suggest Nanonets
    LEGAL_KEYWORDS = [
        "contract",
        "agreement",
        "signed",
        "form",
        "compliance",
        "legal",
        "tax",
        "regulatory",
        "signature",
        "amendment",
    ]

    def classify_page(
        self,
        page: PageObject,
        page_num: int,
        total_pages: int,
        filename: str,
    ) -> ProcessorType:
        """
        Classify single page for routing.

        Args:
            page: pypdf PageObject
            page_num: 0-indexed page number
            total_pages: Total pages in document
            filename: Document filename

        Returns:
            'deepseek' or 'nanonets'
        """
        filename_lower = filename.lower()

        # Priority 1: Legal/compliance documents (filename keywords)
        # These often have signatures on last pages
        is_legal_doc = any(kw in filename_lower for kw in self.LEGAL_KEYWORDS)

        if is_legal_doc:
            # Last 30% of legal documents often contain signatures
            if page_num >= total_pages * 0.7:
                logger.debug(
                    f"Page {page_num + 1}/{total_pages}: Legal doc, last 30% → nanonets"
                )
                return "nanonets"

        # Priority 2: Check for REAL form fields (fillable forms)
        if "/AcroForm" in page:
            acroform = page.get("/AcroForm")
            if acroform and "/Fields" in acroform:
                logger.debug(
                    f"Page {page_num + 1}/{total_pages}: Has fillable form fields → nanonets"
                )
                return "nanonets"

        # Priority 3: Check for annotations (might include handwritten notes)
        if "/Annots" in page:
            annots = page.get("/Annots")
            try:
                # Try to resolve annotations
                if annots and hasattr(annots, "get_object"):
                    annots_obj = annots.get_object()
                    # If there are many annotations, might be handwritten
                    if isinstance(annots_obj, list) and len(annots_obj) > 5:
                        logger.debug(
                            f"Page {page_num + 1}/{total_pages}: Many annotations (>5) → nanonets"
                        )
                        return "nanonets"
            except Exception as e:
                # Ignore annotation parsing errors
                logger.debug(f"Could not parse annotations: {e}")

        # Priority 4: Text extractability (scanned vs native)
        try:
            text = page.extract_text() or ""
            text_length = len(text.strip())

            # Very little text = might be scanned/handwritten
            if text_length < 50:
                # If near end of document, might be signature page
                if page_num >= total_pages * 0.8:
                    logger.debug(
                        f"Page {page_num + 1}/{total_pages}: Low text + end of doc → nanonets"
                    )
                    return "nanonets"
        except Exception as e:
            logger.debug(f"Could not extract text: {e}")

        # Default: Standard page → DeepSeek (fast, good quality)
        logger.debug(f"Page {page_num + 1}/{total_pages}: Standard page → deepseek")
        return "deepseek"

    def classify_all_pages(
        self, pdf_reader, filename: str
    ) -> list[ProcessorType]:
        """
        Classify all pages in document.

        Args:
            pdf_reader: pypdf.PdfReader instance
            filename: Document filename

        Returns:
            List of processor types per page
        """
        total_pages = len(pdf_reader.pages)
        classifications = []

        for page_num, page in enumerate(pdf_reader.pages):
            processor = self.classify_page(page, page_num, total_pages, filename)
            classifications.append(processor)

        # Log classification summary
        deepseek_count = classifications.count("deepseek")
        nanonets_count = classifications.count("nanonets")

        logger.info(
            f"Classification summary: {deepseek_count} pages → DeepSeek, "
            f"{nanonets_count} pages → Nanonets"
        )

        return classifications
