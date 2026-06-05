import logging
import re
from typing import Dict, Any, List
from core.evidence_store import EvidenceStore

logger = logging.getLogger(__name__)

class PageRanker:
    @staticmethod
    def rank_pages(
        question: str,
        question_type: str,
        evidence_store: EvidenceStore,
        doc_intelligence: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Ranks all pages in the repository based on their relevance to the question.
        Returns a sorted list of dicts: [{'page': page_num, 'score': score}]
        """
        pages = evidence_store.get_all_pages()
        if not pages:
            return []

        # 1. Parse question keywords (length > 2, not a stop word)
        stopwords = {
            "the", "and", "for", "out", "not", "yes", "but", "all", "any", "are", "was",
            "what", "where", "does", "mean", "should", "your", "that", "this", "page",
            "sheet", "drawing", "detail", "with", "from", "into", "how", "many", "count",
            "number", "total", "quantity", "about", "there", "here"
        }
        question_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
        question_words = question_words - stopwords

        # 2. Extract equipment tags from query
        tag_pattern = re.compile(
            r"\b(FCU|VAV|RTU|CD|RG|TG|FD|AHU|EF|SF|CH|B|M|ACCU|CU|HP|UH|VFD|FSD)-\d+[A-Z]?\b",
            re.IGNORECASE
        )
        query_tags = {m.group(0).upper() for m in tag_pattern.finditer(question)}

        ranked_results = []

        for p_entry in pages:
            p_num = p_entry.page_number
            
            # Get page type
            page_type = p_entry.metadata.get("page_type")
            if not page_type:
                for pt_dict in doc_intelligence.get("page_types", []):
                    if pt_dict.get("page") == p_num:
                        page_type = pt_dict.get("page_type")
                        break
            if not page_type:
                if p_num in doc_intelligence.get("legend_pages", []):
                    page_type = "legend"
                elif p_num in doc_intelligence.get("notes_pages", []):
                    page_type = "notes"
                elif p_num in doc_intelligence.get("schedule_pages", []):
                    page_type = "schedule"
                elif p_num in doc_intelligence.get("title_block_pages", []):
                    page_type = "title_block"
                else:
                    page_type = "drawing"

            # Form page OCR word set
            page_words = {el.text.lower() for el in p_entry.ocr_elements}

            # OCR word match score
            if question_words:
                match_count = sum(1 for w in question_words if w in page_words)
                ocr_match = match_count / len(question_words)
            else:
                ocr_match = 0.0

            # Equipment tag match score
            tag_match = 0.0
            if query_tags:
                page_tags = {el.text.upper() for el in p_entry.ocr_elements if tag_pattern.match(el.text)}
                if query_tags.intersection(page_tags):
                    tag_match = 1.0

            # Intent page alignment
            intent_match = 0.0
            if question_type in ["legend", "abbreviation"]:
                if p_num in doc_intelligence.get("legend_pages", []):
                    intent_match = 1.0
            elif question_type in ["notes", "specification"]:
                if p_num in doc_intelligence.get("notes_pages", []):
                    intent_match = 1.0
            elif question_type in ["schedule", "equipment"]:
                if p_num in doc_intelligence.get("schedule_pages", []):
                    intent_match = 1.0
            elif question_type in ["title_block", "revision"]:
                if p_num in doc_intelligence.get("title_block_pages", []):
                    intent_match = 1.0
            elif question_type == "cross_sheet":
                # For cross sheet, pages referencing drawing sheets score higher
                refs = sum(1 for w in page_words if re.match(r"^m-\d+$", w))
                if refs > 0:
                    intent_match = min(1.0, refs / 5.0)

            # Combined weighted score
            score = (0.40 * ocr_match) + (0.30 * tag_match) + (0.30 * intent_match)
            
            # Apply page type boosts/penalties for counting
            if question_type == "counting":
                if page_type == "drawing":
                    score += 0.40
                elif page_type == "legend":
                    score -= 0.25
                elif page_type in ["cover", "title_block"]:
                    score -= 0.50

            # Safeguard score bounds
            score = round(max(0.0, min(1.0, score)), 3)

            # Diagnostics matches
            ocr_keyword_matches = sorted(list(question_words.intersection(page_words)))
            page_tags = {el.text.upper() for el in p_entry.ocr_elements if tag_pattern.match(el.text)}
            equipment_matches = sorted(list(query_tags.intersection(page_tags)))
            
            symbols_in_query = {sym.upper() for sym in doc_intelligence.get("symbols", []) if sym.lower() in question.lower()}
            symbol_matches = sorted(list(symbols_in_query.intersection({el.text.upper() for el in p_entry.ocr_elements})))

            ranked_results.append({
                "page": p_num,
                "score": score,
                "ocr_match": round(ocr_match, 2),
                "tag_match": tag_match,
                "intent_match": round(intent_match, 2),
                "ocr_keyword_matches": ocr_keyword_matches,
                "equipment_matches": equipment_matches,
                "symbol_matches": symbol_matches,
                "final_score": score
            })

        # Sort pages in descending order of score, then ascending order of page number
        ranked_results.sort(key=lambda item: (-item["score"], item["page"]))
        
        logger.info(f"PageRanker: Ranked sheets: {[(item['page'], item['score']) for item in ranked_results]}")
        return ranked_results
