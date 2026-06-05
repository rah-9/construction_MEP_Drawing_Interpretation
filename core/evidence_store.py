import logging
import json
import math
from pathlib import Path
from typing import List, Optional, Dict, Any
from difflib import SequenceMatcher

from schemas.document_schema import DocumentRepository, PageEntry, OCRElement, LayoutRegion
import config

logger = logging.getLogger(__name__)

class EvidenceStore:
    def __init__(self, repository: Optional[DocumentRepository] = None):
        """Initialize the store with an optional pre-existing DocumentRepository."""
        if repository is None:
            self.repository = DocumentRepository(
                document_name="empty",
                file_hash="",
                pages=[]
            )
        else:
            self.repository = repository

    def add_page(self, page_entry: PageEntry) -> None:
        """Add a new page entry to the repository."""
        # Remove if page number already exists
        self.repository.pages = [p for p in self.repository.pages if p.page_number != page_entry.page_number]
        self.repository.pages.append(page_entry)
        self.repository.pages.sort(key=lambda p: p.page_number)
        logger.info(f"Added page {page_entry.page_number} to Evidence Store.")

    def add_evidence(self, page_num: int, evidence: OCRElement) -> None:
        """Add an OCR element to a specific page's evidence list."""
        page = self.get_page(page_num)
        if page:
            page.ocr_elements.append(evidence)
            logger.debug(f"Added OCR evidence '{evidence.text}' to page {page_num}.")
        else:
            logger.error(f"Cannot add evidence: Page {page_num} not found in Evidence Store.")

    def get_page(self, page_num: int) -> Optional[PageEntry]:
        """Retrieve a specific page entry by its 1-based page number."""
        for page in self.repository.pages:
            if page.page_number == page_num:
                return page
        return None

    def get_all_pages(self) -> List[PageEntry]:
        """Retrieve all page entries in the repository."""
        return self.repository.pages

    def export_json(self) -> str:
        """Export the current state of the repository as a JSON string."""
        return self.repository.model_dump_json(indent=2)

    def save_to_disk(self) -> Path:
        """Persist the repository to disk in the repositories folder."""
        repo_path = config.REPOSITORIES_DIR / f"{self.repository.file_hash}.json"
        with open(repo_path, "w", encoding="utf-8") as f:
            f.write(self.export_json())
        logger.info(f"Persisted Evidence Store to {repo_path}")
        return repo_path

    def search_text(self, query: str, page_num: Optional[int] = None) -> List[OCRElement]:
        """
        Search for text in the OCR elements. Supports:
        - Single word search (checks if query is inside the word).
        - Multi-word phrase search (concatenates words in reading order and returns intersecting elements).
        
        If page_num is provided, searches only that page. Otherwise, searches all pages.
        """
        query_clean = query.strip().lower()
        if not query_clean:
            return []

        results = []
        pages_to_search = [self.get_page(page_num)] if page_num is not None else self.repository.pages
        pages_to_search = [p for p in pages_to_search if p is not None]

        for page in pages_to_search:
            elements = page.ocr_elements
            if not elements:
                continue

            query_words = query_clean.split()
            
            # Case 1: Single word query
            if len(query_words) == 1:
                for el in elements:
                    if query_clean in el.text.lower():
                        results.append(el)
            
            # Case 2: Multi-word phrase query
            else:
                text_segments = []
                offsets = []
                current_offset = 0
                
                for el in elements:
                    el_text = el.text
                    text_segments.append(el_text)
                    start = current_offset
                    end = current_offset + len(el_text)
                    offsets.append((start, end, el))
                    current_offset = end + 1
                
                full_text = " ".join(text_segments)
                full_text_lower = full_text.lower()
                
                start_idx = 0
                while True:
                    match_pos = full_text_lower.find(query_clean, start_idx)
                    if match_pos == -1:
                        break
                    
                    match_end_pos = match_pos + len(query_clean)
                    
                    phrase_elements = []
                    for start, end, el in offsets:
                        if not (end <= match_pos or start >= match_end_pos):
                            phrase_elements.append(el)
                            
                    if phrase_elements:
                        results.extend(phrase_elements)
                        
                    start_idx = match_pos + 1
                    
        return results

    # =========================================================================
    # V2 UPGRADED SEARCH METHODS
    # =========================================================================

    def search_by_type(self, evidence_type: str) -> List[OCRElement]:
        """Search all OCR elements belonging to a specific evidence type classification."""
        results = []
        for page in self.repository.pages:
            for el in page.ocr_elements:
                if el.evidence_type == evidence_type:
                    results.append(el)
        return results

    def search_by_page(self, page_num: int) -> List[OCRElement]:
        """Retrieve all OCR elements extracted on a specific page."""
        page = self.get_page(page_num)
        return page.ocr_elements if page else []

    def search_by_region(self, region_type: str, page_num: Optional[int] = None) -> List[OCRElement]:
        """
        Retrieves all OCR elements falling inside a layout region of a specific type.
        If page_num is None, searches across all pages.
        """
        results = []
        pages_to_search = [self.get_page(page_num)] if page_num is not None else self.repository.pages
        pages_to_search = [p for p in pages_to_search if p is not None]

        for page in pages_to_search:
            # Find the layout regions matching region_type
            regions = [r for r in page.layout_regions if r.region_type == region_type]
            if not regions:
                continue

            for el in page.ocr_elements:
                eymin, exmin, eymax, exmax = el.bbox
                el_area = (eymax - eymin) * (exmax - exmin)
                if el_area <= 0:
                    continue

                # Check intersection with any matching regions
                for r in regions:
                    rymin, rxmin, rymax, rxmax = r.bbox
                    ixmin = max(exmin, rxmin)
                    iymin = max(eymin, rymin)
                    ixmax = min(exmax, rxmax)
                    iymax = min(eymax, rymax)

                    if ixmax > ixmin and iymax > iymin:
                        intersection_area = (iymax - iymin) * (ixmax - ixmin)
                        # If at least 40% of the word area lies within the region, include it
                        if intersection_area / el_area > 0.4:
                            results.append(el)
                            break
        return results

    def search_by_phrase(self, phrase: str, page_num: Optional[int] = None) -> List[OCRElement]:
        """Wrapper method around standard search_text for explicit phrase search calls."""
        return self.search_text(phrase, page_num)

    def search_fuzzy(self, query: str, threshold: int = 80, page_num: Optional[int] = None) -> List[OCRElement]:
        """
        Performs fuzzy matching on OCR words using difflib.SequenceMatcher.
        threshold is a percentage from 0 to 100.
        """
        query_clean = query.strip().lower()
        if not query_clean:
            return []

        results = []
        pages_to_search = [self.get_page(page_num)] if page_num is not None else self.repository.pages
        pages_to_search = [p for p in pages_to_search if p is not None]

        for page in pages_to_search:
            for el in page.ocr_elements:
                # Calculate difflib similarity ratio
                ratio = SequenceMatcher(None, query_clean, el.text.lower()).ratio()
                if ratio >= (threshold / 100.0):
                    results.append(el)
        return results

    def search_nearest_text(self, target_text: str, page_num: int, limit: int = 5) -> List[OCRElement]:
        """
        Finds OCR elements on a page that are closest geometrically to the occurrences of target_text.
        """
        page = self.get_page(page_num)
        if not page or not page.ocr_elements:
            return []

        # Find target anchor elements
        anchors = self.search_text(target_text, page_num)
        if not anchors:
            return []

        # Compute average center coordinates of the anchor elements
        anchor_cx = sum((a.bbox[1] + a.bbox[3]) / 2.0 for a in anchors) / len(anchors)
        anchor_cy = sum((a.bbox[0] + a.bbox[2]) / 2.0 for a in anchors) / len(anchors)

        # Calculate Euclidean distances to all non-anchor elements
        distance_map = []
        anchor_ids = {id(a) for a in anchors}

        for el in page.ocr_elements:
            if id(el) in anchor_ids:
                continue

            eymin, exmin, eymax, exmax = el.bbox
            el_cx = (exmin + exmax) / 2.0
            el_cy = (eymin + eymax) / 2.0

            dist = math.sqrt((anchor_cx - el_cx) ** 2 + (anchor_cy - el_cy) ** 2)
            distance_map.append((dist, el))

        # Sort by distance and return top matches
        distance_map.sort(key=lambda item: item[0])
        return [item[1] for item in distance_map[:limit]]
