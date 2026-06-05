import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SessionMemory:
    def __init__(self):
        # We store memory in a local dictionary. 
        # Inside Streamlit, app.py will synchronize this with st.session_state.
        self._memory = {
            "resolved_symbols": {},       # e.g., {"CD-1": "CEILING DIFFUSER"}
            "resolved_abbreviations": {}, # e.g., {"CFM": "CUBIC FEET PER MINUTE"}
            "resolved_equipment": {},     # e.g., {"FCU-1": {"AIRFLOW": "400 CFM"}}
            "qa_history": []              # list of {"question": str, "answer": str}
        }

    def clear(self) -> None:
        """Reset the session memory."""
        self._memory = {
            "resolved_symbols": {},
            "resolved_abbreviations": {},
            "resolved_equipment": {},
            "qa_history": []
        }
        logger.info("Session Memory cleared.")

    def add_qa(self, question: str, answer: str) -> None:
        """Log a Q&A pair and attempt to extract key knowledge parameters."""
        self._memory["qa_history"].append({"question": question, "answer": answer})
        
        # 1. Look for symbol/abbrev translation patterns
        # e.g., "What does CD-1 mean?" -> "CD-1 stands for ceiling diffuser"
        # Extract tag and translation
        tag_match = re.search(
            r"\b(FCU|VAV|RTU|CD|RG|TG|FD|AHU|EF|SF|CH|B|M|ACCU|CU|HP|UH|VFD|FSD)-\d+[A-Z]?\b", 
            question, 
            re.IGNORECASE
        )
        if tag_match:
            tag = tag_match.group(0).upper()
            # Clean answer of common prefixes to get definition
            clean_ans = re.sub(
                r"^(" + tag + r"|stands for|means|is|represents|a|an|the|\s)+", 
                "", 
                answer, 
                flags=re.IGNORECASE
            )
            # Take the first sentence/clause
            clean_ans = clean_ans.split(".")[0].split(",")[0].strip()
            if len(clean_ans) > 2 and len(clean_ans) < 60:
                self._memory["resolved_symbols"][tag] = clean_ans
                logger.info(f"SessionMemory: Discovered symbol translation: {tag} = {clean_ans}")

        # 2. Look for equipment tag properties
        # e.g., "What is the airflow for FCU-1?" -> "The airflow is 400 CFM"
        # Extract airflow, capacity, etc.
        cfm_match = re.search(r"\b\d+\s*CFM\b", answer, re.IGNORECASE)
        gpm_match = re.search(r"\b\d+(?:\.\d+)?\s*GPM\b", answer, re.IGNORECASE)
        if tag_match:
            tag = tag_match.group(0).upper()
            if tag not in self._memory["resolved_equipment"]:
                self._memory["resolved_equipment"][tag] = {}
            if cfm_match:
                self._memory["resolved_equipment"][tag]["AIRFLOW"] = cfm_match.group(0).upper()
            if gpm_match:
                self._memory["resolved_equipment"][tag]["FLOW RATE"] = gpm_match.group(0).upper()

    def get_injected_memory_context(self, question: str) -> str:
        """
        Scans the query for known tags or words and outputs a formatted string 
        representing relevant discovered knowledge.
        """
        injected = []
        q_upper = question.upper()

        # 1. Inject resolved symbols/translations
        for tag, definition in self._memory["resolved_symbols"].items():
            if tag in q_upper:
                injected.append(f"- Discovered Symbol: {tag} represents '{definition}' (learned from legend/QA).")

        # 2. Inject resolved equipment properties
        for tag, specs in self._memory["resolved_equipment"].items():
            if tag in q_upper and specs:
                spec_str = ", ".join(f"{k}: {v}" for k, v in specs.items())
                injected.append(f"- Discovered Equipment Specs: {tag} -> {spec_str} (learned from schedules/QA).")

        # 3. Inject past Q&A references (if they match the keywords of the question)
        matched_qa = []
        words = set(re.findall(r"\b\w{4,}\b", question.lower()))
        for item in self._memory["qa_history"][-3:]: # check last 3 QA entries
            q_words = set(re.findall(r"\b\w{4,}\b", item["question"].lower()))
            if words.intersection(q_words):
                matched_qa.append(f"  Q: {item['question']}\n  A: {item['answer']}")

        if matched_qa:
            injected.append("- Past Relevant QA History:\n" + "\n".join(matched_qa))

        if injected:
            return "--- SESSION MEMORY (Discovered Project Context) ---\n" + "\n".join(injected) + "\n\n"
        return ""
