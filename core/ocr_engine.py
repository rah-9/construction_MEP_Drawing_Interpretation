import os
import logging
from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
from schemas.document_schema import OCRElement, LayoutRegion

logger = logging.getLogger(__name__)

# Lazy imports for heavy OCR libraries to avoid startup overhead
PADDLE_AVAILABLE = False
SURYA_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    pass

try:
    # Example Surya import (adjust based on surya package layout if needed)
    # from surya.ocr import run_ocr
    # SURYA_AVAILABLE = True
    pass
except ImportError:
    pass

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass


class OCREngine:
    def __init__(self):
        self.easyocr_reader = None
        self.paddle_ocr = None
        
        # Initialize engines if available
        if PADDLE_AVAILABLE:
            try:
                # use_angle_cls=True helps detect text rotation
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                logger.info("PaddleOCR engine initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
                
        if EASYOCR_AVAILABLE and not self.paddle_ocr:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR engine initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")

    def extract_text(
        self, 
        page: fitz.Page, 
        image_path: Path, 
        page_num: int, 
        layout_regions: List[LayoutRegion]
    ) -> List[OCRElement]:
        """
        Extract text, coordinates, bounding boxes, and confidence.
        Falls back through: PaddleOCR -> EasyOCR -> PyMuPDF Vector Text.
        """
        elements = []
        width = page.rect.width
        height = page.rect.height

        # 1. Try PyMuPDF vector text extraction first (since MEP drawings are often vector PDFs)
        # This is 100% accurate, fast, and does not require neural network inference.
        words = page.get_text("words")  # List of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        
        if words:
            logger.info(f"Page {page_num}: Extracted {len(words)} digital words using PyMuPDF vector extraction.")
            for w in words:
                x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
                
                # Normalize bounding box
                ymin = max(0.0, min(1.0, y0 / height))
                xmin = max(0.0, min(1.0, x0 / width))
                ymax = max(0.0, min(1.0, y1 / height))
                xmax = max(0.0, min(1.0, x1 / width))
                
                bbox = [ymin, xmin, ymax, xmax]
                evidence_type = self._determine_evidence_type(bbox, layout_regions)

                elements.append(OCRElement(
                    text=text,
                    bbox=bbox,
                    confidence=1.0,  # Vector text has perfect extraction confidence
                    page_number=page_num,
                    evidence_type=evidence_type
                ))
            return elements

        # 2. If vector text is empty, it's a scanned PDF. Attempt ML-based OCR.
        # Try PaddleOCR
        if self.paddle_ocr:
            try:
                logger.info(f"Page {page_num}: Running PaddleOCR...")
                result = self.paddle_ocr.ocr(str(image_path), cls=True)
                if result and result[0]:
                    for line in result[0]:
                        box, (text, conf) = line
                        # box is [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        
                        # Scale from image pixels to normalized coordinates
                        # PaddleOCR runs on the rendered high-res image
                        with Image.open(image_path) as img:
                            img_w, img_h = img.size
                        
                        ymin = max(0.0, min(1.0, min(ys) / img_h))
                        xmin = max(0.0, min(1.0, min(xs) / img_w))
                        ymax = max(0.0, min(1.0, max(ys) / img_h))
                        xmax = max(0.0, min(1.0, max(xs) / img_w))
                        
                        bbox = [ymin, xmin, ymax, xmax]
                        evidence_type = self._determine_evidence_type(bbox, layout_regions)

                        elements.append(OCRElement(
                            text=text,
                            bbox=bbox,
                            confidence=float(conf),
                            page_number=page_num,
                            evidence_type=evidence_type
                        ))
                    return elements
            except Exception as e:
                logger.error(f"PaddleOCR extraction failed: {e}")

        # Try EasyOCR
        if self.easyocr_reader:
            try:
                logger.info(f"Page {page_num}: Running EasyOCR...")
                results = self.easyocr_reader.readtext(str(image_path))
                for (bbox_pts, text, conf) in results:
                    # bbox_pts is [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                    xs = [p[0] for p in bbox_pts]
                    ys = [p[1] for p in bbox_pts]
                    
                    with Image.open(image_path) as img:
                        img_w, img_h = img.size
                    
                    ymin = max(0.0, min(1.0, min(ys) / img_h))
                    xmin = max(0.0, min(1.0, min(xs) / img_w))
                    ymax = max(0.0, min(1.0, max(ys) / img_h))
                    xmax = max(0.0, min(1.0, max(xs) / img_w))
                    
                    bbox = [ymin, xmin, ymax, xmax]
                    evidence_type = self._determine_evidence_type(bbox, layout_regions)

                    elements.append(OCRElement(
                        text=text,
                        bbox=bbox,
                        confidence=float(conf),
                        page_number=page_num,
                        evidence_type=evidence_type
                    ))
                return elements
            except Exception as e:
                logger.error(f"EasyOCR extraction failed: {e}")

        logger.warning(f"Page {page_num}: No text could be extracted. Returning empty list.")
        return []

    def _determine_evidence_type(self, bbox: List[float], layout_regions: List[LayoutRegion]) -> str:
        """Assign the evidence type based on which layout region the bounding box falls into."""
        ymin, xmin, ymax, xmax = bbox
        box_area = (ymax - ymin) * (xmax - xmin)
        if box_area <= 0:
            return "ocr_text"

        best_region_type = "ocr_text"
        max_overlap_ratio = 0.0

        for r in layout_regions:
            rymin, rxmin, rymax, rxmax = r.bbox
            
            # Compute intersection box
            ixmin = max(xmin, rxmin)
            iymin = max(ymin, rymin)
            ixmax = min(xmax, rxmax)
            iymax = min(ymax, rymax)
            
            if ixmax > ixmin and iymax > iymin:
                intersection_area = (iymax - iymin) * (ixmax - ixmin)
                overlap_ratio = intersection_area / box_area
                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    # Map region_type to evidence_type values
                    if r.region_type == "notes":
                        best_region_type = "note"
                    else:
                        best_region_type = r.region_type

        # If overlap is significant (e.g. > 40%), classify it under that region type
        if max_overlap_ratio > 0.4:
            return best_region_type

        return "ocr_text"
