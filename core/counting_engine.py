import logging
import re
from typing import Dict, Any, List, Optional
from core.evidence_store import EvidenceStore
from core.counting_handler import CountingHandler
from schemas.response_schema import InterpretationResponse

logger = logging.getLogger(__name__)

class CountingEngine:
    @staticmethod
    def audit_count(
        question: str,
        primary_response: InterpretationResponse,
        verifier_response: Optional[InterpretationResponse],
        evidence_store: EvidenceStore,
        ranked_pages: List[int]
    ) -> Dict[str, Any]:
        """
        Audits VLM counting results by analyzing:
        - Stated primary count
        - OCR word occurrences across relevant ranked sheets
        - Number of bounding boxes generated
        - Verification model stated count
        Returns a dict of counts, supports, and consensus.
        """
        # 1. Extract target object to count
        q_lower = question.lower().strip()
        strip_phrases = [
            "how many", "are there", "in the drawing", "on this sheet", "shown",
            "is there", "do we have", "can you count", "count the number of",
            "count the", "total number of", "number of", "count", "please",
            "are located on the drawing", "are located in the drawing",
            "are on the drawing", "are on this sheet", "drawing area", "drawing", "sheet"
        ]
        
        target_noun = q_lower
        for phrase in strip_phrases:
            target_noun = target_noun.replace(phrase, "")
            
        target_noun = re.sub(r"[?.,!]", "", target_noun).strip()
        
        # Tokenize and filter noise words
        words = target_noun.split()
        noise_words = {"the", "of", "on", "in", "a", "an", "at", "for", "with", "by", "are"}
        while words and words[0] in noise_words:
            words.pop(0)
        while words and words[-1] in noise_words:
            words.pop()
            
        target_obj = " ".join(words).strip()
        
        # Custom maps for engineering targets
        if "diffuser" in target_obj:
            target_obj = "diffuser"
        elif "fire damper" in target_obj:
            target_obj = "fire damper"
        elif "fcu" in target_obj:
            target_obj = "FCU"
            
        # Singularize: remove trailing 's' only if result > 3 chars
        if target_obj.endswith("s") and len(target_obj) > 3:
            target_obj = target_obj[:-1]
            
        # Casing standardizations for acronyms
        if target_obj.lower() == "fcu":
            target_obj = "FCU"
        elif target_obj.lower() == "vav":
            target_obj = "VAV"
            
        logger.info(f"CountingEngine: Extracted target object '{target_obj}' for query: {question}")

        # Resolve candidate symbols/terms to search dynamically from legend regions
        resolved_symbols = [target_obj]

        # Get all legend regions elements or fall back to pages classified as legends
        doc_intel = getattr(evidence_store.repository, "doc_intelligence", {})
        legend_elements = evidence_store.search_by_region("legend")
        if not legend_elements:
            legend_pages = doc_intel.get("legend_pages", [])
            for lp in legend_pages:
                legend_elements.extend(evidence_store.search_by_page(lp))

        # Search for query target matching phrases or terms
        target_clean = target_obj.lower().strip()
        if target_clean.endswith('s') and len(target_clean) > 3:
            target_clean = target_clean[:-1]
        target_kws = [w for w in target_clean.split() if len(w) > 2]

        for el in legend_elements:
            el_text_lower = el.text.lower()
            if target_clean in el_text_lower or (target_kws and any(kw in el_text_lower for kw in target_kws)):
                p_num = el.page_number
                p_entry = evidence_store.get_page(p_num)
                if p_entry:
                    m_ymin, m_xmin, m_ymax, m_xmax = el.bbox
                    for other in p_entry.ocr_elements:
                        e_ymin, e_xmin, e_ymax, e_xmax = other.bbox
                        if other != el and abs((e_ymin + e_ymax)/2.0 - (m_ymin + m_ymax)/2.0) < 0.02:
                            text_stripped = other.text.strip().upper()
                            if re.match(r"^[A-Z0-9\-\.]{1,6}$", text_stripped) and text_stripped.lower() not in target_clean:
                                if text_stripped not in resolved_symbols:
                                    resolved_symbols.append(text_stripped)

        # Ensure uppercase / lowercase variations are fully resolved
        resolved_symbols = list(set(resolved_symbols + [s.upper() for s in resolved_symbols] + [s.lower() for s in resolved_symbols]))

        # 2. Extract primary count
        p_digits = re.findall(r"\b\d+\b", primary_response.answer)
        primary_count = int(p_digits[0]) if p_digits else 0

        # 3. Extract verifier count
        verifier_count = primary_count
        if verifier_response:
            v_digits = re.findall(r"\b\d+\b", verifier_response.answer)
            verifier_count = int(v_digits[0]) if v_digits else primary_count

        # 4. Search OCR occurrences across ranked sheets (ignoring legend pages and legend layout areas)
        ocr_count = 0
        for p in ranked_pages:
            p_entry = evidence_store.get_page(p)
            if p_entry:
                page_type = p_entry.metadata.get("page_type")
                if not page_type:
                    for pt_dict in doc_intel.get("page_types", []):
                        if pt_dict.get("page") == p:
                            page_type = pt_dict.get("page_type")
                            break
                if page_type == "legend":
                    logger.info(f"CountingEngine: Skipping legend page {p} during OCR count.")
                    continue
                    
            page_matches = {}
            for sym in resolved_symbols:
                matches = evidence_store.search_text(sym, page_num=p)
                for m in matches:
                    is_in_legend = False
                    if p_entry:
                        for region in p_entry.layout_regions:
                            if region.region_type == "legend":
                                rymin, rxmin, rymax, rxmax = region.bbox
                                eymin, exmin, eymax, exmax = m.bbox
                                ixmin = max(exmin, rxmin)
                                iymin = max(eymin, rymin)
                                ixmax = min(exmax, rxmax)
                                iymax = min(eymax, rymax)
                                if ixmax > ixmin and iymax > iymin:
                                    is_in_legend = True
                                    break
                    if not is_in_legend:
                        bbox_key = tuple(round(c, 5) for c in m.bbox)
                        page_matches[bbox_key] = m
            ocr_count += len(page_matches)

        # 5. Extract visual bounding box count
        visual_count = len(primary_response.bbox) if primary_response.bbox else 0

        # 6. Calculate support metrics
        max_ocr = max(1, ocr_count, primary_count)
        ocr_support = 1.0 - (abs(primary_count - ocr_count) / max_ocr)

        max_vis = max(1, visual_count, primary_count)
        visual_support = 1.0 - (abs(primary_count - visual_count) / max_vis)

        max_ver = max(1, verifier_count, primary_count)
        verification_support = 1.0 - (abs(primary_count - verifier_count) / max_ver)

        # Enforce support ranges [0.0, 1.0]
        ocr_support = round(max(0.0, min(1.0, ocr_support)), 2)
        visual_support = round(max(0.0, min(1.0, visual_support)), 2)
        verification_support = round(max(0.0, min(1.0, verification_support)), 2)

        # Calculate consensus count:
        # If verification model agrees, trust primary count.
        # Otherwise, if OCR support is high, trust primary count.
        # Else fallback to OCR count.
        if verification_support >= 0.8:
            final_count = primary_count
        elif ocr_support >= 0.8:
            final_count = primary_count
        else:
            final_count = ocr_count if ocr_count > 0 else primary_count

        result = {
            "target_object": target_obj,
            "resolved_symbols": [s for s in resolved_symbols if len(s) > 1 and s.isupper()][:5],
            "count": final_count,
            "ocr_count": ocr_count,
            "visual_count": visual_count,
            "verifier_count": verifier_count,
            "ocr_support": ocr_support,
            "visual_support": visual_support,
            "verification_support": verification_support
        }
        
        logger.info(f"CountingEngine: Audit complete. Result: {result}")
        return result
