import logging
import json
import base64
import re
from pathlib import Path
from typing import List, Optional, Union
import requests

import config
from schemas.response_schema import InterpretationResponse
from schemas.document_schema import OCRElement
from prompts.system_prompts import SYSTEM_INSTRUCTION
from prompts.reasoning_templates import BASE_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

class QwenClient:
    def __init__(self):
        self.api_base = config.QWEN_API_BASE
        self.model = config.QWEN_MODEL
        logger.info(f"QwenClient configured with API base: {self.api_base}, primary Model: {self.model}")

    def _image_to_base64(self, image_path: Path) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _extract_json(self, text: str) -> str:
        """Extract JSON substring from text in case model wraps it in markdown code blocks."""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    def query(
        self, 
        image_path: Path, 
        ocr_evidence: Union[List[OCRElement], str], 
        question: str, 
        page_number: int,
        prompt_template: Optional[str] = None
    ) -> InterpretationResponse:
        """
        Sends the page image and OCR text evidence to Qwen2.5-VL,
        falling back to local SmolVLM/SmolVLM2 models if Qwen is not running.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Page image not found: {image_path}")

        # Encode image to base64
        base64_image = self._image_to_base64(image_path)
        image_data_url = f"data:image/png;base64,{base64_image}"

        # Compile context
        if isinstance(ocr_evidence, str):
            ocr_text_context = ocr_evidence
        else:
            unique_words = [el.text for el in ocr_evidence]
            ocr_text_context = " ".join(unique_words)[:4000]

        # Determine prompt and instructions
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

        # Fallback chain for local models: config.QWEN_MODEL -> smolvlm -> smolvlm2
        models_to_try = [self.model, "smolvlm", "smolvlm2"]
        last_err = None

        for idx, model_name in enumerate(models_to_try):
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": sys_inst
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }

            url = f"{self.api_base.rstrip('/')}/chat/completions"
            try:
                logger.info(f"Querying local model ({model_name}) at {url} (Attempt {idx + 1}/{len(models_to_try)})...")
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()

                result = response.json()
                choices = result.get("choices", [])
                if not choices:
                    raise ValueError(f"No response choices returned from Local API for model {model_name}.")

                resp_text = choices[0]["message"]["content"].strip()
                logger.debug(f"Local model raw response: {resp_text}")

                clean_json = self._extract_json(resp_text)
                data = json.loads(clean_json)

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
                logger.warning(f"Local model {model_name} failed: {e}")
                last_err = e
                # Continue loop to try fallback local models

        # If all fallback models failed
        logger.error("All local VLM model execution attempts failed.")
        raise last_err
