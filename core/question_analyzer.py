import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class QuestionAnalyzer:
    @staticmethod
    def analyze_question(question: str) -> Dict[str, Any]:
        """
        Classifies the input question into one of the 12 drawing question categories
        and provides retrieval and reasoning strategies.
        """
        q_lower = question.lower().strip()

        # Regular expression patterns for intent matching
        counting_pat = r"\b(how many|count|number of|quantity|qty|how much|total number)\b"
        legend_pat = r"\b(legend|symbol table|list of symbols)\b"
        abbreviation_pat = r"\b(stand for|stands for|acronym|abbreviation|abbrev|short for|definition of|meaning of)\b"
        notes_pat = r"\b(general notes?|notes?|drawing notes?|note number|note \d+)\b"
        spec_pat = r"\b(spec|specification|testing requirement|clause|section \d+|div\b|division \d+|standard|smacna|ashrae|nfpa)\b"
        schedule_pat = r"\b(schedules?|schedule tables?|panel schedules?|list of equipment)\b"
        cross_sheet_pat = r"\b(see sheet|refer to sheet|shown on sheet|drawing reference|cross reference|cross-sheet|detail \d+ on sheet)\b"
        revision_pat = r"\b(revision|revisions|revised|rev date|revision history|revision table)\b"
        title_block_pat = r"\b(sheet name|project name|sheet number|scale|drawing dates?|author|drawing number|sheet title|client|drawn by|checked by)\b"
        location_pat = r"\b(where|location|room|floor|area|zone|coordinate|placed|installed|situated)\b"
        conflict_pat = r"\b(conflict|clash|interfere|overlap|override|clashing|interference|issue)\b"
        equipment_pat = r"\b(fcu|vav|rtu|ahu|diffuser|grille|damper|fan|unit|device|compressor|chiller|boiler|capacity|cfm|gpm|voltage|tag|airflow|chw|hw)\b"

        if re.search(counting_pat, q_lower):
            q_type = "counting"
            retrieval_strategy = "Retrieve legend definitions, layout boundaries, and OCR elements matching the target item across all pages."
            reasoning_strategy = "Verify symbol types against legend and perform spatial OCR counting comparison."
        elif re.search(abbreviation_pat, q_lower):
            q_type = "abbreviation"
            retrieval_strategy = "Retrieve abbreviations, legends, and acronym definitions."
            reasoning_strategy = "Resolve abbreviations or unit shorthand codes using project legends and schedules."
        elif re.search(legend_pat, q_lower):
            q_type = "legend"
            retrieval_strategy = "Retrieve entries from the Legend/Symbol layout region."
            reasoning_strategy = "Search for keyword matches inside the Legend text block to find symbol or tag meanings."
        elif re.search(notes_pat, q_lower):
            q_type = "notes"
            retrieval_strategy = "Retrieve entries from the Notes layout region."
            reasoning_strategy = "Scan note numbers and bullet points, extracting clauses matching notes queries."
        elif re.search(spec_pat, q_lower):
            q_type = "specification"
            retrieval_strategy = "Retrieve specification sections, codes, standards, and notes clauses."
            reasoning_strategy = "Compare text references against engineering specifications (SMACNA, NFPA, ASHRAE)."
        elif re.search(revision_pat, q_lower):
            q_type = "revision"
            retrieval_strategy = "Retrieve revision columns, history blocks, and dates in the title block area."
            reasoning_strategy = "Extract revision numbers, dates, modifications description, and approval stamps."
        elif re.search(title_block_pat, q_lower):
            q_type = "title_block"
            retrieval_strategy = "Retrieve details inside the Title Block region (sheet number, sheet name, project info)."
            reasoning_strategy = "Extract metadata about sheet naming, page counts, scale, and author."
        elif re.search(cross_sheet_pat, q_lower):
            q_type = "cross_sheet"
            retrieval_strategy = "Retrieve title blocks, page links, drawing references, and cross-references across sheets."
            reasoning_strategy = "Analyze drawing cross-references to trace details or items to other sheets."
        elif re.search(location_pat, q_lower):
            q_type = "location"
            retrieval_strategy = "Retrieve drawing area OCR coordinates and spatial room references."
            reasoning_strategy = "Analyze coordinates of the target item relative to column lines or room tags."
        elif re.search(conflict_pat, q_lower):
            q_type = "conflict"
            retrieval_strategy = "Retrieve clash annotations, overlap keywords, and multi-sheet conflict points."
            reasoning_strategy = "Assess physical clashes between layouts (e.g. duct-to-wall overlaps, sprinkler clearances)."
        elif re.search(schedule_pat, q_lower):
            q_type = "schedule"
            retrieval_strategy = "Retrieve equipment tables, schedules, and electrical matrices."
            reasoning_strategy = "Extract capacities, flow rates, electrical requirements, and notes from schedule grids."
        elif re.search(equipment_pat, q_lower):
            q_type = "equipment"
            retrieval_strategy = "Retrieve OCR elements corresponding to equipment tags and adjacent scheduling entries."
            reasoning_strategy = "Extract flow rates (CFM), electrical specifications, elevations, and design ratings."
        else:
            q_type = "general"
            retrieval_strategy = "Retrieve title block entries and general OCR text."
            reasoning_strategy = "General drawing reading QA (scale, date, author, sheet number)."

        logger.info(f"QuestionAnalyzer: Classified question as '{q_type}'. Strategy: {reasoning_strategy}")

        return {
            "question_type": q_type,
            "retrieval_strategy": retrieval_strategy,
            "reasoning_strategy": reasoning_strategy
        }
