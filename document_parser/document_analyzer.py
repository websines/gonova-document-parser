"""Fast PDF document analysis without full OCR processing."""

from pathlib import Path
from typing import Dict

from loguru import logger
from pypdf import PdfReader


class DocumentAnalyzer:
    """
    Fast PDF analysis to determine routing strategy.

    Analyzes PDF characteristics without expensive OCR:
    - Page count
    - Text layers (native vs scanned)
    - Form fields (checkboxes, signatures)
    - Images presence
    - Estimated complexity
    """

    @staticmethod
    def analyze(pdf_path: str | Path) -> Dict:
        """
        Perform fast analysis of PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with analysis results:
                - total_pages: Number of pages
                - has_forms: Boolean, contains form fields
                - has_images: Boolean, contains images
                - text_layers: List[bool], True if page has extractable text
                - native_ratio: Float, ratio of native text pages
                - estimated_handwriting: List[bool], heuristic detection
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.debug(f"Analyzing {pdf_path.name}...")

        try:
            reader = PdfReader(str(pdf_path))

            analysis = {
                "total_pages": len(reader.pages),
                "has_forms": False,
                "has_images": False,
                "text_layers": [],
                "estimated_handwriting": [],
            }

            for page_num, page in enumerate(reader.pages):
                # Check for extractable text (native vs scanned)
                text = page.extract_text() or ""
                has_text = len(text.strip()) > 50
                analysis["text_layers"].append(has_text)

                # Check for form fields (AcroForm, annotations)
                if "/AcroForm" in page or "/Annots" in page:
                    analysis["has_forms"] = True

                # Check for images
                if "/XObject" in page.get("/Resources", {}):
                    analysis["has_images"] = True

                # Heuristic: very short text + images might be handwritten
                if not has_text and analysis["has_images"]:
                    analysis["estimated_handwriting"].append(True)
                else:
                    analysis["estimated_handwriting"].append(False)

            # Calculate native text ratio
            native_pages = sum(analysis["text_layers"])
            analysis["native_ratio"] = (
                native_pages / analysis["total_pages"] if analysis["total_pages"] > 0 else 0
            )

            # Estimate handwriting ratio
            handwriting_pages = sum(analysis["estimated_handwriting"])
            analysis["handwriting_ratio"] = (
                handwriting_pages / analysis["total_pages"]
                if analysis["total_pages"] > 0
                else 0
            )

            logger.debug(
                f"Analysis: {analysis['total_pages']} pages, "
                f"{analysis['native_ratio']:.1%} native text, "
                f"forms={analysis['has_forms']}, "
                f"handwriting~{analysis['handwriting_ratio']:.1%}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Analysis failed for {pdf_path}: {e}")
            return {
                "total_pages": 0,
                "has_forms": False,
                "has_images": False,
                "text_layers": [],
                "native_ratio": 0.0,
                "handwriting_ratio": 0.0,
                "error": str(e),
            }
