import logging
import json
from pathlib import Path
from typing import List, Optional, Union, Tuple
from PIL import Image
from google import genai
from google.genai import types

import config
from schemas.response_schema import InterpretationResponse
from schemas.document_schema import OCRElement
from prompts.system_prompts import SYSTEM_INSTRUCTION
from prompts.reasoning_templates import BASE_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set in the environment or config.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("GeminiClient initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Client: {e}")
                self.client = None

    def query(
        self, 
        image_path: Union[Path, List[Tuple[int, Path]]], 
        ocr_evidence: Union[List[OCRElement], str], 
        question: str, 
        page_number: int,
        prompt_template: Optional[str] = None
    ) -> InterpretationResponse:
        """
        Sends the page image and OCR text evidence to Gemini 2.5 Flash
        and retrieves a structured InterpretationResponse.
        """
        if not self.client:
            raise ValueError("Gemini Client is not initialized. Check GEMINI_API_KEY.")

        # Handle both list of (page_num, Path) tuples and single Path
        if isinstance(image_path, list):
            logger.info(f"Gemini call: sending {len(image_path)} images for pages {[p for p,_ in image_path]}")
            contents = []
            for p, path in image_path:
                if not path.exists():
                    raise FileNotFoundError(f"Page image not found: {path}")
                contents.append(Image.open(path))
        else:
            if not image_path.exists():
                raise FileNotFoundError(f"Page image not found: {image_path}")
            contents = [Image.open(image_path)]

        # Handle both raw OCRElements list and pre-filtered context strings
        if isinstance(ocr_evidence, str):
            ocr_text_context = ocr_evidence
        else:
            unique_words = [el.text for el in ocr_evidence]
            ocr_text_context = " ".join(unique_words)[:5000]

        # Decide which template and instruction to use
        if prompt_template:
            prompt = prompt_template.format(
                context=ocr_text_context,
                question=question,
                page_number=page_number
            )
            sys_inst = BASE_SYSTEM_INSTRUCTION
        else:
            prompt = f"""
Analyzed Page Number: {page_number}
Extracted OCR Context:
\"\"\"
{ocr_text_context}
\"\"\"

User Question: {question}

Remember the rules:
- Rely only on visible evidence.
- Return output strictly formatted according to the requested JSON schema.
- If uncertainty exists, answer 'HUMAN REVIEW REQUIRED', set evidence_found = false, and set review_required = true.
"""
            sys_inst = SYSTEM_INSTRUCTION

        try:
            logger.info(f"Querying Gemini 2.5 Flash for page {page_number}...")
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents + [prompt],
                config=types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    response_mime_type="application/json",
                    response_schema=InterpretationResponse,
                    temperature=0.0,
                )
            )

            # Response text should be structured JSON matching the schema
            resp_text = response.text.strip()
            logger.debug(f"Gemini raw response: {resp_text}")
            
            # Load and parse using Pydantic
            data = json.loads(resp_text)
            
            # Map reasoning to reasoning_summary
            reasoning = data.get("reasoning_summary")
            if not reasoning:
                reasoning = data.get("reasoning", "Successfully parsed JSON response.")
                
            # Parse confidence
            raw_confidence = data.get("confidence")
            if raw_confidence is None:
                confidence = 0.0
            elif isinstance(raw_confidence, str):
                cleaned_conf = raw_confidence.strip().lower()
                if cleaned_conf in ["n/a", "null", "none", "nan", ""]:
                    confidence = 0.0
                else:
                    try:
                        cleaned_conf = cleaned_conf.replace("%", "").strip()
                        val = float(cleaned_conf)
                        if val > 1.0:
                            confidence = val / 100.0
                        else:
                            confidence = val
                    except ValueError:
                        confidence = 0.0
            elif isinstance(raw_confidence, (int, float)):
                confidence = float(raw_confidence)
                if confidence > 1.0:
                    confidence = confidence / 100.0
            else:
                confidence = 0.0

            # Normalize fields to fit Pydantic schema
            normalized_data = {
                "answer": data.get("answer", resp_text),
                "confidence": confidence,
                "page_number": page_number,
                "bbox": data.get("bbox", []),
                "evidence_found": bool(data.get("evidence_found", True)) if data.get("evidence_found") is not None else True,
                "review_required": bool(data.get("review_required", False)),
                "reasoning_summary": reasoning
            }

            # Clean/validate bbox format
            if not isinstance(normalized_data["bbox"], list):
                normalized_data["bbox"] = []
            else:
                cleaned_bbox = []
                for box in normalized_data["bbox"]:
                    if isinstance(box, list) and len(box) == 4:
                        try:
                            cleaned_bbox.append([float(coord) for coord in box])
                        except (ValueError, TypeError):
                            pass
                normalized_data["bbox"] = cleaned_bbox

            return InterpretationResponse(**normalized_data)

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise e
