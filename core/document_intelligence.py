import logging
import re
from typing import Dict, Any, List, Set
from schemas.document_schema import DocumentRepository

logger = logging.getLogger(__name__)

class DocumentIntelligence:
    @staticmethod
    def analyze(repository: DocumentRepository) -> Dict[str, Any]:
        """
        Scans all pages in the DocumentRepository to extract document-wide intelligence:
        - legend_pages: page numbers containing legends/symbols
        - notes_pages: page numbers containing general notes or specs
        - schedule_pages: page numbers containing schedule lists
        - title_block_pages: page numbers containing title block metadata
        - equipment_tags: unique equipment tags present in text (e.g. FCU-1, VAV-2)
        - symbols: alphanumeric symbols or codes found in legend areas
        - abbreviations: unit abbreviations or shorthand words (e.g. CFM, GPM)
        - specifications: specification code sections found in text (e.g. Section 15010)
        """
        logger.info(f"Running Document Intelligence on: {repository.document_name}")

        legend_pages: List[int] = []
        notes_pages: List[int] = []
        schedule_pages: List[int] = []
        title_block_pages: List[int] = []

        all_equipment_tags: Set[str] = set()
        all_symbols: Set[str] = set()
        all_abbreviations: Set[str] = set()
        all_specifications: Set[str] = set()

        # Regex patterns
        equip_pattern = re.compile(
            r"\b(FCU|VAV|RTU|CD|RG|TG|FD|AHU|EF|SF|CH|B|M|ACCU|CU|HP|UH|VFD|FSD)-\d+[A-Z]?\b", 
            re.IGNORECASE
        )
        spec_pattern = re.compile(
            r"\b(?:SECTION|DIV|DIVISION)\s*(?:\d{5}|\d{2})\b", 
            re.IGNORECASE
        )
        
        # Candidate unit abbreviations
        abbrev_candidates = {
            "CFM", "GPM", "AFF", "BOB", "BOJ", "NTS", "ADA", "MEP", "PVC", "PSI", "MBH", 
            "F", "C", "WG", "RPM", "HP", "DB", "WB", "EWT", "LWT", "LWT", "MCA", "MOP", "FLA"
        }

        for page in repository.pages:
            p_num = page.page_number
            text_lower = page.embedded_text.lower()
            
            # Form page OCR text corpus
            ocr_text_list = [el.text for el in page.ocr_elements]
            ocr_corpus = " ".join(ocr_text_list)
            ocr_corpus_lower = ocr_corpus.lower()

            # Check Layout regions
            has_legend_region = any(r.region_type == "legend" for r in page.layout_regions)
            has_notes_region = any(r.region_type == "notes" for r in page.layout_regions)
            has_tb_region = any(r.region_type == "title_block" for r in page.layout_regions)

            # Calculate evidence scores
            scores = {
                "cover": 0,
                "legend": 0,
                "notes": 0,
                "schedule": 0,
                "title_block": 0,
                "drawing": 0
            }
            ocr_elements_count = len(page.ocr_elements)
            has_drawing_region = any(r.region_type == "drawing_area" for r in page.layout_regions)

            # COVER signals:
            cover_keywords = ["cover sheet", "index of drawings", "list of drawings", "drawing index", "title sheet"]
            if any(kw in ocr_corpus_lower for kw in cover_keywords):
                scores["cover"] += 5
            if p_num == 1 and ocr_elements_count < 200:
                scores["cover"] += 2

            # LEGEND signals:
            legend_word_count = ocr_corpus_lower.count("legend") + ocr_corpus_lower.count("symbol")
            if has_legend_region and legend_word_count >= 2:
                scores["legend"] += 5
            if legend_word_count >= 4:
                scores["legend"] += 3
            if has_legend_region:
                scores["legend"] += 2
            scores["legend"] += min(4, legend_word_count)
            if has_drawing_region:
                scores["legend"] -= 3

            # NOTES signals:
            if ocr_corpus_lower.count("general notes") >= 2:
                scores["notes"] += 5
            notes_indicators = ["general notes", "testing requirement", "all ductwork", "contractor shall", "all work shall", "specifications"]
            notes_indicator_hits = sum(1 for ind in notes_indicators if ind in ocr_corpus_lower)
            if has_notes_region and notes_indicator_hits >= 2:
                scores["notes"] += 3
            if has_notes_region:
                scores["notes"] += 2
            scores["notes"] += min(4, notes_indicator_hits)
            if has_drawing_region:
                scores["notes"] -= 3

            # SCHEDULE signals:
            schedule_word_count = ocr_corpus_lower.count("schedule")
            if schedule_word_count >= 3:
                scores["schedule"] += 5
            schedule_indicators = ["schedule", "capacity", "cfm", "gpm", "voltage", "flow rate", "weight", "dimensions", "chiller schedule", "fan schedule", "diffuser schedule"]
            schedule_indicator_hits = sum(1 for ind in schedule_indicators if ind in ocr_corpus_lower)
            if schedule_indicator_hits >= 4:
                scores["schedule"] += 3
            if ocr_elements_count > 300 and schedule_word_count >= 2:
                scores["schedule"] += 2
            scores["schedule"] += min(4, schedule_indicator_hits)

            # TITLE_BLOCK signals:
            tb_indicators = ["drawing no", "title block", "drawn by", "checked by", "approved by", "revision"]
            if has_tb_region and any(term in ocr_corpus_lower for term in tb_indicators):
                scores["title_block"] += 3
            if has_tb_region:
                scores["title_block"] += 2
            if any(term in ocr_corpus_lower for term in ["project name", "scale", "date"]):
                scores["title_block"] += 1
            if ocr_elements_count > 150:
                scores["title_block"] -= 5

            # DRAWING signals:
            if has_drawing_region:
                scores["drawing"] += 5
            equipment_tags_on_page_count = len(list(equip_pattern.finditer(ocr_corpus)))
            if equipment_tags_on_page_count >= 3:
                scores["drawing"] += 3
            if ocr_elements_count > 400:
                scores["drawing"] += 2
            elif ocr_elements_count > 200:
                scores["drawing"] += 1
            if p_num == 1:
                scores["drawing"] -= 3

            # Determine page type with prioritized tiebreaker
            page_type = max(scores, key=lambda k: (scores[k], 
                ["drawing","notes","legend","schedule","title_block","cover"].index(k) * -1
            ))

            if not page.metadata:
                page.metadata = {}
            page.metadata["page_type"] = page_type

            # Populate list indexes AFTER classification
            if page_type == "legend":
                legend_pages.append(p_num)
            elif page_type == "notes":
                notes_pages.append(p_num)
            elif page_type == "schedule":
                schedule_pages.append(p_num)
            elif page_type == "title_block":
                title_block_pages.append(p_num)

            logger.info(f"Page {p_num} scores: {scores} → classified as '{page_type}'")

            # Extract Equipment tags
            for match in equip_pattern.finditer(ocr_corpus):
                all_equipment_tags.add(match.group(0).upper())

            # Extract Specs
            for match in spec_pattern.finditer(ocr_corpus):
                all_specifications.add(match.group(0).upper())

            # Extract Abbreviations
            words = re.findall(r"\b[A-Za-z]{2,5}\b", ocr_corpus)
            for w in words:
                wu = w.upper()
                if wu in abbrev_candidates:
                    all_abbreviations.add(wu)

            # Extract Symbols from Legend layout region
            for r in page.layout_regions:
                if r.region_type == "legend":
                    # Extract OCR text elements within this legend box
                    rymin, rxmin, rymax, rxmax = r.bbox
                    for el in page.ocr_elements:
                        eymin, exmin, eymax, exmax = el.bbox
                        # Check overlap
                        ixmin = max(exmin, rxmin)
                        iymin = max(eymin, rymin)
                        ixmax = min(exmax, rxmax)
                        iymax = min(eymax, rymax)
                        if ixmax > ixmin and iymax > iymin:
                            # If overlap exists, candidate symbol
                            # Only include codes/tags/acronyms as symbols
                            if re.match(r"^[A-Z0-9\-\/]{2,10}$", el.text):
                                all_symbols.add(el.text.upper())

        # Build page types list
        page_types_list = []
        for page in repository.pages:
            p_num = page.page_number
            page_type = page.metadata.get("page_type", "mixed")
            page_types_list.append({
                "page": p_num,
                "page_type": page_type
            })

        # Clean duplicates
        intelligence = {
            "legend_pages": sorted(list(set(legend_pages))),
            "notes_pages": sorted(list(set(notes_pages))),
            "schedule_pages": sorted(list(set(schedule_pages))),
            "title_block_pages": sorted(list(set(title_block_pages))),
            "equipment_tags": sorted(list(all_equipment_tags)),
            "symbols": sorted(list(all_symbols)),
            "abbreviations": sorted(list(all_abbreviations)),
            "specifications": sorted(list(all_specifications)),
            "page_types": page_types_list
        }

        logger.info(f"Extracted intelligence: Legends={intelligence['legend_pages']}, Notes={intelligence['notes_pages']}, Schedules={intelligence['schedule_pages']}, Page Types={page_types_list}")
        return intelligence
