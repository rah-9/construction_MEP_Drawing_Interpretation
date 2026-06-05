import logging
from typing import List, Dict, Any, Optional

from core.evidence_store import EvidenceStore
from schemas.document_schema import OCRElement

logger = logging.getLogger(__name__)

class ContextRetriever:
    @staticmethod
    def format_elements(elements: List[OCRElement]) -> str:
        """Format list of OCRElements into a readable block with coordinates."""
        if not elements:
            return "No matching text evidence found in this section."
        
        # De-duplicate elements to avoid repeated strings
        seen_ids = set()
        unique_elements = []
        for el in elements:
            el_id = f"{el.text}_{el.bbox[0]:.4f}_{el.bbox[1]:.4f}"
            if el_id not in seen_ids:
                seen_ids.add(el_id)
                unique_elements.append(el)

        # Sort elements by coordinate (top-to-bottom, then left-to-right) to form readable text
        unique_elements.sort(key=lambda el: (el.bbox[0], el.bbox[1]))
        
        lines = []
        for el in unique_elements:
            ymin, xmin, ymax, xmax = el.bbox
            lines.append(f"'{el.text}' [at y: {ymin:.2f}, x: {xmin:.2f}] (Type: {el.evidence_type})")
        
        return "\n".join(lines)

    @staticmethod
    def retrieve_notes(evidence_store: EvidenceStore, page_num: int, query: str) -> str:
        """Retrieve drawing notes text by notes region and relevant keyword queries."""
        elements = evidence_store.search_by_region("notes", page_num)
        if not elements:
            # Fallback search notes keywords
            elements = evidence_store.search_text("note", page_num) + evidence_store.search_text("general", page_num)
        
        # If query has specific words, pull nearest neighbors of those query terms
        words = [w for w in query.split() if len(w) > 3]
        for w in words:
            elements.extend(evidence_store.search_nearest_text(w, page_num, limit=5))

        return ContextRetriever.format_elements(elements)

    @staticmethod
    def retrieve_legends(evidence_store: EvidenceStore, page_num: int, query: str) -> str:
        """Retrieve drawing legend text by legend region and queried terms."""
        elements = evidence_store.search_by_region("legend", page_num)
        if not elements:
            # Fallback
            elements = evidence_store.search_text("legend", page_num) + evidence_store.search_text("symbol", page_num)

        # Search for queried acronyms or abbreviations
        words = [w for w in query.split() if w.isupper() or len(w) > 2]
        for w in words:
            elements.extend(evidence_store.search_nearest_text(w, page_num, limit=5))

        return ContextRetriever.format_elements(elements)

    @staticmethod
    def retrieve_equipment(evidence_store: EvidenceStore, page_num: int, query: str) -> str:
        """Retrieve equipment specification table details and tags."""
        # Find matches for common equipment tags (FCU, VAV, RTU, diffuser, damper, etc.)
        equipment_keywords = ["fcu", "vav", "rtu", "diffuser", "grille", "damper", "fan", "pump", "boiler"]
        elements = []
        
        # Extract potential tag from query
        words = [w.strip("?,.()\"'") for w in query.split()]
        target_tags = [w for w in words if any(ek in w.lower() for ek in equipment_keywords) or w.isupper()]
        
        if target_tags:
            for tag in target_tags:
                logger.info(f"ContextRetriever: Retrieving nearest annotations for target tag '{tag}'")
                elements.extend(evidence_store.search_nearest_text(tag, page_num, limit=15))
        else:
            # Retrieve all elements of type 'legend' and elements in drawing area matching equipment
            for ek in equipment_keywords:
                elements.extend(evidence_store.search_text(ek, page_num))
        
        return ContextRetriever.format_elements(elements)

    @staticmethod
    def retrieve_conflicts(evidence_store: EvidenceStore, page_num: int, query: str) -> str:
        """Retrieve text nodes associated with clash detection keywords."""
        conflict_keywords = ["conflict", "clash", "interfere", "overlap", "joist", "pipe", "duct", "light", "beam"]
        elements = []
        for kw in conflict_keywords:
            elements.extend(evidence_store.search_nearest_text(kw, page_num, limit=5))
        return ContextRetriever.format_elements(elements)

    @staticmethod
    def retrieve_locations(evidence_store: EvidenceStore, page_num: int, query: str) -> str:
        """Retrieve location-related drawing markers (rooms, coordinate zones)."""
        location_keywords = ["room", "floor", "area", "zone", "corridor", "office", "lobby"]
        elements = []
        for kw in location_keywords:
            elements.extend(evidence_store.search_text(kw, page_num))
            
        # Add nearest text elements to target location keywords
        words = [w for w in query.split() if w.isupper() or len(w) > 3]
        for w in words:
            elements.extend(evidence_store.search_nearest_text(w, page_num, limit=8))
            
        return ContextRetriever.format_elements(elements)

    @staticmethod
    def retrieve_relevant_regions(evidence_store: EvidenceStore, page_num: int, region_type: str) -> str:
        """Retrieve OCR elements contained in a specific layout region."""
        elements = evidence_store.search_by_region(region_type, page_num)
        return ContextRetriever.format_elements(elements)

    @staticmethod
    def retrieve_nearest_matches(evidence_store: EvidenceStore, page_num: int, query: str, limit: int = 10) -> str:
        """Retrieves exact match OCR text blocks and their nearest surrounding context."""
        words = [w for w in query.split() if len(w) > 2]
        elements = []
        for w in words:
            elements.extend(evidence_store.search_nearest_text(w, page_num, limit=limit))
        return ContextRetriever.format_elements(elements)

    @classmethod
    def retrieve_context(cls, evidence_store: EvidenceStore, page_num: int, question: str, question_type: str) -> str:
        """Main routing function to gather target context evidence."""
        logger.info(f"ContextRetriever: Performing strategy retrieval for type '{question_type}'")
        
        if question_type == "notes":
            return cls.retrieve_notes(evidence_store, page_num, question)
        elif question_type == "legend":
            return cls.retrieve_legends(evidence_store, page_num, question)
        elif question_type == "equipment":
            return cls.retrieve_equipment(evidence_store, page_num, question)
        elif question_type == "location":
            return cls.retrieve_locations(evidence_store, page_num, question)
        elif question_type == "conflict":
            return cls.retrieve_conflicts(evidence_store, page_num, question)
        elif question_type == "counting":
            # Counting requires Legend details + Layout boundaries + nearest matching tags
            legend_ctx = cls.retrieve_legends(evidence_store, page_num, question)
            drawing_ctx = cls.retrieve_relevant_regions(evidence_store, page_num, "drawing_area")
            matches_ctx = cls.retrieve_nearest_matches(evidence_store, page_num, question, limit=15)
            return f"--- Legend Context ---\n{legend_ctx}\n\n--- Drawing Area Context ---\n{drawing_ctx}\n\n--- Relevant Matches ---\n{matches_ctx}"
        else:
            # General drawing QA pulls title block details & general matches
            tb_ctx = cls.retrieve_relevant_regions(evidence_store, page_num, "title_block")
            general_ctx = cls.retrieve_nearest_matches(evidence_store, page_num, question, limit=10)
            return f"--- Title Block ---\n{tb_ctx}\n\n--- General Context ---\n{general_ctx}"
