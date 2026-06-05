import logging
import re
from typing import List, Dict, Any
from core.evidence_store import EvidenceStore
from schemas.document_schema import OCRElement

logger = logging.getLogger(__name__)

def format_elements(elements: List[OCRElement]) -> str:
    if not elements:
        return ""
    seen_ids = set()
    unique_elements = []
    for el in elements:
        el_id = f"{el.text}_{el.bbox[0]:.4f}_{el.bbox[1]:.4f}"
        if el_id not in seen_ids:
            seen_ids.add(el_id)
            unique_elements.append(el)
    unique_elements.sort(key=lambda el: (el.bbox[0], el.bbox[1]))
    lines = []
    for el in unique_elements:
        ymin, xmin, ymax, xmax = el.bbox
        lines.append(f"[{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f}] {el.text}")
    return "\n".join(lines)


class ContextBuilder:
    @staticmethod
    def build_context(
        evidence_store: EvidenceStore,
        ranked_pages: List[int],
        question: str,
        question_type: str,
        doc_intelligence: Dict[str, Any]
    ) -> str:
        """
        Assembles a structured context packet from the ranked pages based on the question type.
        Avoids full page OCR dumps.
        """
        logger.info(f"ContextBuilder: Assembling packet for intent '{question_type}' across pages {ranked_pages}")
        
        # Identify the definition pages
        definition_pages = set()
        if doc_intelligence:
            definition_pages.update(doc_intelligence.get("legend_pages", []))
            definition_pages.update(doc_intelligence.get("notes_pages", []))

        # Ensure get_all_ocr alias exists on evidence_store
        if not hasattr(evidence_store, "get_all_ocr"):
            evidence_store.get_all_ocr = evidence_store.search_by_page

        # Pre-assemble definition context
        def_context = ""
        for def_page in sorted(definition_pages):
            if def_page not in ranked_pages:  # avoid duplication
                def_elements = evidence_store.get_all_ocr(def_page)
                if def_elements:
                    def_context += f"=== DEFINITION PAGE (Sheet {def_page}) ===\n"
                    def_context += format_elements(def_elements) + "\n\n"

        logger.info(f"Definition pages injected: {sorted(definition_pages)}")

        if not ranked_pages:
            if def_context:
                return def_context.strip()
            return "No relevant pages could be resolved for context retrieval."

        normal_context = ""

        if question_type in ["location", "counting", "conflict"]:
            # Resolve all legend pages in the document
            legend_pages_in_doc = doc_intelligence.get("legend_pages", [])
            if not legend_pages_in_doc:
                for pt_dict in doc_intelligence.get("page_types", []):
                    if pt_dict.get("page_type") == "legend":
                        legend_pages_in_doc.append(pt_dict.get("page"))
            legend_pages_in_doc = sorted(list(set(legend_pages_in_doc)))

            # Gather legend elements
            legend_elements = []
            for lp in legend_pages_in_doc:
                # Try region first
                lp_elements = evidence_store.search_by_region("legend", lp)
                if not lp_elements:
                    # Fallback to full page
                    lp_elements = evidence_store.search_by_page(lp)
                legend_elements.extend(lp_elements)

            legend_page_nums = legend_pages_in_doc
            drawing_page_nums = [dp for dp in ranked_pages if dp not in legend_pages_in_doc]
            logger.info(
                f"ContextBuilder counting: legend_pages={legend_page_nums}, "
                f"drawing_pages={drawing_page_nums}, "
                f"legend_elements_added={len(legend_elements)}"
            )

            formatted_legend = ""
            if legend_elements:
                seen_ids = set()
                unique_legend = []
                for el in legend_elements:
                    el_id = f"{el.text}_{el.bbox[0]:.4f}_{el.bbox[1]:.4f}"
                    if el_id not in seen_ids:
                        seen_ids.add(el_id)
                        unique_legend.append(el)
                unique_legend.sort(key=lambda el: (el.bbox[0], el.bbox[1]))
                lines = []
                for el in unique_legend:
                    ymin, xmin, ymax, xmax = el.bbox
                    lines.append(f"[{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f}] {el.text}")
                formatted_legend = "\n".join(lines)

            # Gather drawing elements from drawing sheets
            formatted_drawing_parts = []
            for dp in ranked_pages:
                if dp not in legend_pages_in_doc:
                    page_elements = evidence_store.search_by_region("drawing_area", dp)
                    if not page_elements:
                        page_elements = evidence_store.search_by_page(dp)
                    
                    if page_elements:
                        seen_ids = set()
                        unique_els = []
                        for el in page_elements:
                            el_id = f"{el.text}_{el.bbox[0]:.4f}_{el.bbox[1]:.4f}"
                            if el_id not in seen_ids:
                                seen_ids.add(el_id)
                                unique_els.append(el)
                        unique_els.sort(key=lambda el: (el.bbox[0], el.bbox[1]))
                        
                        lines = []
                        for el in unique_els:
                            ymin, xmin, ymax, xmax = el.bbox
                            lines.append(f"[{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f}] {el.text}")
                        formatted_page_ocr = "\n".join(lines)
                        formatted_drawing_parts.append(f"=== SHEET {dp} ===\n{formatted_page_ocr}")

            context_parts = []
            if formatted_legend:
                context_parts.append(f"=== SYMBOL DEFINITIONS (use this to identify what to count) ===\n{formatted_legend}")
            if formatted_drawing_parts:
                drawings_ocr = "\n\n".join(formatted_drawing_parts)
                context_parts.append(f"=== DRAWING AREA (count symbols here) ===\n{drawings_ocr}")

            normal_context = "\n\n".join(context_parts)

        else:
            packet_parts = []

            for page_num in ranked_pages:
                page_elements: List[OCRElement] = []
                
                # Get page type
                p_entry = evidence_store.get_page(page_num)
                page_type = "drawing"
                if p_entry:
                    page_type = p_entry.metadata.get("page_type")
                    if not page_type:
                        for pt_dict in doc_intelligence.get("page_types", []):
                            if pt_dict.get("page") == page_num:
                                page_type = pt_dict.get("page_type")
                                break
                if not page_type:
                    if page_num in doc_intelligence.get("legend_pages", []):
                        page_type = "legend"
                    elif page_num in doc_intelligence.get("notes_pages", []):
                        page_type = "notes"
                    elif page_num in doc_intelligence.get("schedule_pages", []):
                        page_type = "schedule"
                    elif page_num in doc_intelligence.get("title_block_pages", []):
                        page_type = "title_block"
                    else:
                        page_type = "drawing"

                # Form header for the page context with instructions
                if page_type == "legend":
                    page_header = f"=== SHEET {page_num} [LEGEND SHEET] (For Symbol Definitions Reference Only - DO NOT COUNT PLACED SYMBOLS ON THIS SHEET) ==="
                elif page_type == "notes":
                    page_header = f"=== SHEET {page_num} [GENERAL NOTES SHEET] (For Notes/Specifications Reference Only - DO NOT COUNT PLACED SYMBOLS ON THIS SHEET) ==="
                elif page_type == "schedule":
                    page_header = f"=== SHEET {page_num} [EQUIPMENT SCHEDULE SHEET] (For Schedule Details Reference Only - DO NOT COUNT PLACED SYMBOLS ON THIS SHEET) ==="
                elif page_type == "cover":
                    page_header = f"=== SHEET {page_num} [COVER/TITLE SHEET] (DO NOT COUNT PLACED SYMBOLS ON THIS SHEET) ==="
                elif page_type == "title_block":
                    page_header = f"=== SHEET {page_num} [TITLE BLOCK SHEET] (DO NOT COUNT PLACED SYMBOLS ON THIS SHEET) ==="
                else:
                    page_header = f"=== SHEET {page_num} [ACTUAL PLACED DRAWING SHEET] (COUNT PLACED SYMBOLS ON THIS SHEET) ==="
                
                # 1. Legend / Abbreviation Queries
                if question_type in ["legend", "abbreviation"]:
                    # Pull legend regions on this page
                    page_elements.extend(evidence_store.search_by_region("legend", page_num))
                    
                    # Extract fuzzy symbol matches in question (e.g. CD-1)
                    symbol_pattern = re.compile(r"\b(FCU|VAV|RTU|CD|RG|TG|FD|AHU|EF|SF)-\d+[A-Z]?\b", re.IGNORECASE)
                    symbols_in_q = [m.group(0) for m in symbol_pattern.finditer(question)]
                    
                    # If no pattern matches, look for acronyms or capitalized abbreviations
                    if not symbols_in_q:
                        symbols_in_q = [w for w in question.split() if w.isupper() and len(w) >= 2]
                    
                    for sym in symbols_in_q:
                        # search nearest surrounding text of the symbol
                        page_elements.extend(evidence_store.search_text(sym, page_num))
                        page_elements.extend(evidence_store.search_nearest_text(sym, page_num, limit=5))

                # 2. Notes / Specification Queries
                elif question_type in ["notes", "specification"]:
                    # Pull notes regions
                    page_elements.extend(evidence_store.search_by_region("notes", page_num))
                    
                    # Check for spec sections matching question
                    spec_pattern = re.compile(r"\b(?:SECTION|DIV|DIVISION)\s*(?:\d{5}|\d{2})\b", re.IGNORECASE)
                    specs_in_q = [m.group(0) for m in spec_pattern.finditer(question)]
                    
                    for spec in specs_in_q:
                        page_elements.extend(evidence_store.search_nearest_text(spec, page_num, limit=8))
                    
                    # Pull words in question of length > 3
                    keywords = [w for w in question.split() if len(w) > 3 and w.lower() not in ["note", "notes", "clause", "specification"]]
                    for kw in keywords:
                        page_elements.extend(evidence_store.search_nearest_text(kw, page_num, limit=3))

                # 3. Schedule / Equipment Queries
                elif question_type in ["schedule", "equipment"]:
                    # Extract schedule page regions if they contain target tags
                    page_elements.extend(evidence_store.search_by_region("legend", page_num)) # schedules are sometimes classified as legends
                    
                    # Extract surrounding text for equipment tags
                    equip_pattern = re.compile(
                        r"\b(FCU|VAV|RTU|AHU|EF|SF|CH|B|M|ACCU|CU|HP|UH|VFD|FSD)-\d+[A-Z]?\b", 
                        re.IGNORECASE
                    )
                    tags_in_q = [m.group(0) for m in equip_pattern.finditer(question)]
                    if not tags_in_q:
                        # Try finding upper case words representing equipment
                        tags_in_q = [w for w in question.split() if w.isupper() and len(w) >= 3]

                    for tag in tags_in_q:
                        page_elements.extend(evidence_store.search_nearest_text(tag, page_num, limit=15))

                # 4. Location / Counting / Conflict Queries
                elif question_type in ["location", "counting", "conflict"]:
                    # First try keyword search
                    words = [w for w in question.split() if len(w) > 3]
                    for w in words:
                        page_elements.extend(evidence_store.search_nearest_text(w, page_num, limit=10))
                    
                    # Fallback to drawing_area region if empty
                    if not page_elements:
                        page_elements.extend(evidence_store.search_by_region("drawing_area", page_num))
                    
                    # Fallback to full page if still empty
                    if not page_elements:
                        page_elements.extend(evidence_store.search_by_page(page_num))

                # 5. Title Block / Revision Queries
                elif question_type in ["title_block", "revision"]:
                    page_elements.extend(evidence_store.search_by_region("title_block", page_num))

                # 6. General / Fallback Context
                else:
                    # Include Title block + drawing area layout regions
                    page_elements.extend(evidence_store.search_by_region("title_block", page_num))
                    page_elements.extend(evidence_store.search_by_region("notes", page_num))
                    # Add nearest terms for the question keywords
                    keywords = [w for w in question.split() if len(w) > 3]
                    for kw in keywords:
                        page_elements.extend(evidence_store.search_nearest_text(kw, page_num, limit=5))

                # Format elements for this page
                if not page_elements:
                    formatted_ocr = "No matching text evidence found in this section."
                else:
                    seen_ids = set()
                    unique_elements = []
                    for el in page_elements:
                        el_id = f"{el.text}_{el.bbox[0]:.4f}_{el.bbox[1]:.4f}"
                        if el_id not in seen_ids:
                            seen_ids.add(el_id)
                            unique_elements.append(el)
                    unique_elements.sort(key=lambda el: (el.bbox[0], el.bbox[1]))
                    lines = []
                    for el in unique_elements:
                        ymin, xmin, ymax, xmax = el.bbox
                        lines.append(f"[{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f}] {el.text}")
                    formatted_ocr = "\n".join(lines)
                
                # Only append page block if it has matching elements
                if formatted_ocr and "No matching text evidence" not in formatted_ocr:
                    packet_parts.append(f"{page_header}\n{formatted_ocr}")

            normal_context = "\n\n".join(packet_parts)

        if "No matching text evidence" in normal_context:
            normal_context = ""

        # Merge
        if def_context:
            if normal_context:
                return (def_context + normal_context).strip()
            else:
                return def_context.strip()
        else:
            if normal_context:
                return normal_context.strip()
            else:
                return "No matching text evidence found in the indexed drawing layout regions."
