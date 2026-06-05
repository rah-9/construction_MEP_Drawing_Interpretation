import logging
import json
import base64
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse
import requests

import config
from schemas.response_schema import InterpretationResponse
from prompts.reasoning_templates import BASE_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)

class NVIDIAClient:
    def __init__(self):
        self.api_key = config.NVIDIA_API_KEY
        self.endpoint = config.NVIDIA_ENDPOINT
        self.model = config.NVIDIA_PRIMARY_MODEL
        self.payload_format = config.NVIDIA_PAYLOAD_FORMAT.lower() # "array" or "string"
        
        if not self.api_key or "your_nvidia_api_key" in self.api_key:
            logger.warning("NVIDIA_API_KEY is not set or holds placeholder value.")
            self.client_active = False
        else:
            self.client_active = True
            logger.info(f"NVIDIAClient initialized. Endpoint: {self.endpoint}, Model: {self.model}, Format: {self.payload_format}")

    def _image_to_base64(self, image_path: Path) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _extract_json(self, text: str) -> str:
        """Extract JSON substring from text in case the model wraps it in markdown tags."""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    def _normalize_response(self, result_json: Dict[str, Any], page_number: int) -> InterpretationResponse:
        """
        Extracts content from completions and normalizes it into InterpretationResponse
        before Pydantic validation.
        """
        choices = result_json.get("choices", [])
        if not choices:
            raise ValueError(f"No response choices returned from NVIDIA API. Response: {result_json}")

        resp_text = choices[0]["message"]["content"].strip()
        logger.debug(f"NVIDIA raw content extracted: {resp_text}")

        # Attempt to extract and parse JSON
        try:
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
                "page_number": int(data.get("page_number", page_number)),
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

        except Exception as json_err:
            logger.warning(f"Failed to parse NVIDIA response as JSON: {json_err}. Normalizing plain text.")
            # Normalization path for plain text response
            normalized_data = {
                "answer": resp_text,
                "confidence": 0.5,
                "page_number": page_number,
                "bbox": [],
                "evidence_found": True,
                "review_required": True,
                "reasoning_summary": f"Response was returned as plain text instead of structured JSON. Safe defaults applied. Error: {str(json_err)}"
            }

        # Standardize page number
        if normalized_data["page_number"] != page_number:
            normalized_data["page_number"] = page_number

        return InterpretationResponse(**normalized_data)

    def health_check(self) -> Tuple[bool, str]:
        """
        Runs a health check on the NVIDIA API endpoint.
        Returns (is_healthy, status_message).
        """
        if not self.client_active:
            return False, "NVIDIA API Key not configured."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "ping"
                }
            ],
            "max_tokens": 5
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                return True, "Endpoint accessible and active."
            else:
                return False, f"HTTP Error {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"Failed to connect: {str(e)}"

    def query(
        self,
        image_path: Path,
        retrieved_context: str,
        question: str,
        page_number: int,
        prompt_template: str,
        custom_format: Optional[str] = None
    ) -> InterpretationResponse:
        """
        Queries any NVIDIA-hosted multimodal model with page image and filtered context.
        Supports both 'array' (Content Array) and 'string' (HTML img tag) payload formats.
        Includes asynchronous polling for 202 Accepted status codes.
        """
        if not self.client_active:
            raise ValueError("NVIDIA Client is inactive: NVIDIA_API_KEY is not configured.")

        if not image_path.exists():
            raise FileNotFoundError(f"Page image not found: {image_path}")

        # Encode image to base64
        base64_image = self._image_to_base64(image_path)

        # Populate prompt template
        prompt = prompt_template.format(
            context=retrieved_context,
            question=question,
            page_number=page_number
        )

        fmt = custom_format.lower() if custom_format else self.payload_format

        # Construct message content depending on format choice
        if fmt == "string":
            # Image URL/HTML tag format
            user_content = f'{prompt} <img src="data:image/png;base64,{base64_image}" />'
        else:
            # Content Array format (default)
            user_content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": BASE_SYSTEM_INSTRUCTION
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        max_retries = 3
        backoff_factor = 2.0
        timeout = 45

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Querying NVIDIA endpoint ({self.model}) at {self.endpoint} (attempt {attempt}/{max_retries}, format: {fmt})...")
                response = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout)
                
                # Log raw initial response
                logger.info(f"Raw NVIDIA API Response: {response.text}")

                # Check for rate limiting (429)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = int(retry_after) if retry_after and retry_after.isdigit() else (backoff_factor ** attempt)
                    logger.warning(f"Rate limited (429). Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue

                # Handle 202 Accepted status (asynchronous polling path)
                if response.status_code == 202:
                    response_data = response.json()
                    request_id = response_data.get("requestId") or response_data.get("id")
                    if not request_id:
                        raise ValueError(f"Received 202 status but no requestId found in response: {response_data}")

                    # Construct status URL dynamically
                    parsed = urlparse(self.endpoint)
                    base_url = f"{parsed.scheme}://{parsed.netloc}/v1"
                    status_url = f"{base_url}/status/{request_id}"

                    poll_headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "accept": "application/json"
                    }
                    max_poll_attempts = 15
                    poll_delay = 2.0
                    final_result = None

                    for poll_attempt in range(1, max_poll_attempts + 1):
                        logger.info(f"Polling NVIDIA status (attempt {poll_attempt}/{max_poll_attempts}) at {status_url}...")
                        poll_response = requests.get(status_url, headers=poll_headers, timeout=10)
                        
                        # Log raw poll responses
                        logger.info(f"Raw NVIDIA API Poll Response: {poll_response.text}")
                        
                        if poll_response.status_code == 200:
                            final_result = poll_response.json()
                            break
                        elif poll_response.status_code == 202:
                            time.sleep(poll_delay)
                            poll_delay = min(5.0, poll_delay * 1.5)
                        else:
                            poll_response.raise_for_status()
                            raise ValueError(f"Unexpected status code {poll_response.status_code} during polling: {poll_response.text}")
                    
                    if not final_result:
                        raise TimeoutError(f"NVIDIA API request timed out during polling for requestId: {request_id}")

                    return self._normalize_response(final_result, page_number)

                response.raise_for_status()

                # Directly parse 200 response
                result = response.json()
                return self._normalize_response(result, page_number)

            except requests.exceptions.HTTPError as http_err:
                if response.status_code not in [429, 202, 500, 502, 503, 504]:
                    logger.error(f"NVIDIA API client error: {http_err}")
                    raise http_err
                
                if attempt == max_retries:
                    raise http_err
                
                sleep_time = backoff_factor ** attempt
                logger.warning(f"NVIDIA API transient HTTP error {response.status_code}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as conn_err:
                if attempt == max_retries:
                    logger.error(f"NVIDIA API request failed after {max_retries} attempts: {conn_err}")
                    raise conn_err
                
                sleep_time = backoff_factor ** attempt
                logger.warning(f"NVIDIA connection/timeout error. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"NVIDIA query failed with error: {e}")
                raise e
