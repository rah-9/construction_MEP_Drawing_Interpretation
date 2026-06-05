import logging
import re
from typing import Dict, Any, List

from core.evidence_store import EvidenceStore
from schemas.response_schema import InterpretationResponse

logger = logging.getLogger(__name__)

class CountingHandler:
    @staticmethod
    def extract_target_object(question: str) -> str:
        """
        Extracts the target noun phrase to count from the question.
        Examples: 'how many fire dampers?' -> 'fire damper'
                  'count the number of VAV boxes' -> 'vav box'
        """
        q_lower = question.lower()
        
        # Strip common prefixes
        prefixes = [
            r"how many",
            r"count the number of",
            r"count",
            r"total number of",
            r"number of"
        ]
        
        clean_text = q_lower
        for p in prefixes:
            clean_text = re.sub(p, "", clean_text)
        
        # Strip question marks, plural 's' at the end, and cleanup whitespace
        clean_text = clean_text.replace("?", "").strip()
        
        # De-pluralize common words
        if clean_text.endswith("s"):
            # De-pluralize (e.g. dampers -> damper, boxes -> box, diffusers -> diffuser)
            if clean_text.endswith("es") and clean_text[:-2] in ["box", "bush", "switch"]:
                clean_text = clean_text[:-2]
            else:
                clean_text = clean_text[:-1]

        # Standard acronym mappings
        acronym_mappings = {
            "vav box": "vav",
            "fcu unit": "fcu",
            "rtu unit": "rtu",
            "fire damper": "fd"
        }
        
        for k, v in acronym_mappings.items():
            if k in clean_text:
                return v

        logger.info(f"CountingHandler: Extracted target object '{clean_text}' from question.")
        return clean_text

    @staticmethod
    def verify_counting_evidence(
        model_answer: str, 
        target_object: str, 
        page_num: int, 
        evidence_store: EvidenceStore
    ) -> Dict[str, Any]:
        """
        Extracts the count stated by the model and cross-references it
        with the occurrences of target_object in the OCR Evidence Store.
        """
        # Parse the first integer from model's answer
        digits = re.findall(r"\b\d+\b", model_answer)
        model_count = int(digits[0]) if digits else None

        # Search OCR text for direct occurrences of the target object
        ocr_matches = evidence_store.search_text(target_object, page_num)
        ocr_count = len(ocr_matches)

        # Handle abbreviation conversions (e.g. if target is "fd" which represents fire dampers)
        if ocr_count == 0 and len(target_object) > 2:
            # Try matching abbreviation (first letters of words, e.g. Fire Damper -> FD)
            abbrev = "".join([w[0] for w in target_object.split() if w])
            if len(abbrev) >= 2:
                ocr_matches = evidence_store.search_text(abbrev, page_num)
                ocr_count = len(ocr_matches)

        ocr_support = 0.0
        if model_count is not None:
            if ocr_count == model_count:
                ocr_support = 1.0
            elif ocr_count > 0:
                # Relative difference penalty
                diff = abs(model_count - ocr_count)
                ocr_support = max(0.0, 1.0 - (diff / max(ocr_count, model_count)))
            else:
                # No OCR occurrences found to back up the model count
                ocr_support = 0.1
        else:
            ocr_support = 0.0

        logger.info(f"CountingHandler: Model Count: {model_count}, OCR Occurrences: {ocr_count}, Support: {ocr_support:.2f}")

        return {
            "model_count": model_count,
            "ocr_count": ocr_count,
            "ocr_support": ocr_support,
            "ocr_matches": ocr_matches
        }

    @staticmethod
    def adjust_counting_confidence(
        base_confidence: float, 
        ocr_support: float, 
        agreement_score: float
    ) -> float:
        """
        Enforce: Counting confidence cannot exceed Medium (0.79) unless:
        - Multiple models agree (agreement_score >= 0.8)
        - AND OCR evidence supports the result (ocr_support >= 0.8)
        """
        if agreement_score >= 0.80 and ocr_support >= 0.80:
            # Both models agree and OCR matches support it, allow High confidence
            adjusted_conf = base_confidence
        else:
            # Cap confidence to Medium (0.79 max)
            adjusted_conf = min(0.79, base_confidence)
            logger.info(f"CountingHandler: Capping counting confidence to {adjusted_conf:.2f} due to insufficient model agreement or OCR verification.")

        return adjusted_conf
