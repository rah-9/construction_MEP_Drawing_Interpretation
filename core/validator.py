import logging
import re
from typing import List, Tuple, Set
from schemas.response_schema import InterpretationResponse
from schemas.document_schema import PageEntry, OCRElement
from core.evidence_store import EvidenceStore

logger = logging.getLogger(__name__)

class Validator:
    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract code-like tags, uppercase acronyms, or numbers from the text.
        Examples: 'FCU-1', 'CP', 'VAV', '1/4"'.
        """
        # Find uppercase acronyms or alphanumeric drawing codes (e.g., FCU-1, CP, 120V)
        codes = re.findall(r"\b[A-Z0-9_\-\/]{2,}\b", text)
        
        # Filter out common English stop words or pure units if needed, but keep drawing abbreviations
        stop_words = {"THE", "AND", "FOR", "OUT", "NOT", "YES", "BUT", "ALL", "ANY", "ARE", "WAS"}
        filtered = {code for code in codes if code not in stop_words}
        return filtered

    def validate(
        self, 
        response: InterpretationResponse, 
        page_entry: PageEntry, 
        evidence_store: EvidenceStore
    ) -> Tuple[bool, str, List[OCRElement]]:
        """
        Validates the model response:
        1. Checks if page exists.
        2. Validates bounding box coordinates (must be in range [0.0, 1.0]).
        3. Enforces bounding boxes exist if evidence_found is True.
        4. Cross-verifies key terms in the response against the OCR Evidence Store.
        
        Returns:
            (is_valid, validation_message, list_of_matching_ocr_elements)
        """
        # 1. Check page number
        if response.page_number != page_entry.page_number:
            return False, f"Page number mismatch: response has {response.page_number}, target is {page_entry.page_number}", []

        # 2. Check evidence grounding
        if response.evidence_found and not response.bbox:
            # Evidence found but no bounding box provided is a violation
            return False, "Evidence was declared found, but bounding box (bbox) is empty.", []

        # 3. Check bounding box coordinates
        for box in response.bbox:
            if len(box) != 4:
                return False, f"Malformed bounding box {box}. Must contain exactly 4 values: [ymin, xmin, ymax, xmax].", []
            ymin, xmin, ymax, xmax = box
            if not (0.0 <= ymin <= 1.0 and 0.0 <= xmin <= 1.0 and 0.0 <= ymax <= 1.0 and 0.0 <= xmax <= 1.0):
                return False, f"Bounding box {box} is out of page boundaries (0.0 to 1.0).", []
            if ymin > ymax or xmin > xmax:
                return False, f"Invalid bounding box boundaries: ymin {ymin} > ymax {ymax} or xmin {xmin} > xmax {xmax}.", []

        # 4. OCR Cross-Verification
        # Skip validation search if no evidence was found or human review is already requested
        if not response.evidence_found or response.review_required:
            return True, "No grounding verification required (evidence not found or review requested).", []

        keywords = self._extract_keywords(response.answer)
        logger.info(f"Validator: Extracted keywords for cross-verification: {keywords}")

        matching_elements = []
        missing_keywords = []

        for kw in keywords:
            # Search for keyword in this page's OCR elements
            matches = evidence_store.search_text(kw, page_num=page_entry.page_number)
            if matches:
                matching_elements.extend(matches)
            else:
                # If a keyword is completely absent from OCR, record it
                missing_keywords.append(kw)

        if missing_keywords:
            msg = f"OCR Cross-Verification Warning: Key term(s) {missing_keywords} mentioned in the answer were not found in the page OCR."
            logger.warning(msg)
            # If critical codes (like FCU-1 or CP) are not found, we don't invalidate completely,
            # but we return validation message so the confidence engine can penalize it.
            return True, msg, matching_elements

        return True, "All verified terms found in OCR Store.", matching_elements
