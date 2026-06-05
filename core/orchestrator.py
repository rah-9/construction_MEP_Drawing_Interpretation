import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import config
from schemas.response_schema import InterpretationResponse
from schemas.document_schema import PageEntry, OCRElement
from models.nvidia_client import NVIDIAClient
from models.gemini_client import GeminiClient
from models.qwen_client import QwenClient

from core.evidence_store import EvidenceStore
from core.question_analyzer import QuestionAnalyzer
from core.model_verifier import ModelVerifier
from core.validator import Validator
from core.confidence_engine import ConfidenceEngine
from core.grounding_engine import GroundingEngine
from prompts.reasoning_templates import TEMPLATES

logger = logging.getLogger(__name__)

class RequestOrchestrator:
    def __init__(self):
        self.nvidia_client = NVIDIAClient()
        self.gemini_client = GeminiClient()
        self.qwen_client = QwenClient()
        self.validator = Validator()
        self.confidence_engine = ConfidenceEngine()
        self.grounding_engine = GroundingEngine()

    def interpret_question(
        self, 
        evidence_store: EvidenceStore, 
        page_num: int, 
        question: str,
        reasoning_mode: str = "smart",
        settings: Optional[Dict[str, bool]] = None,
        session_memory: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Coordinates the document-aware question interpretation workflow:
        1. Analyzes the question intent (12 intents).
        2. Ranks pages using Page Ranker based on intent, tags, and keywords.
        3. Retrieves and builds context packets using Context Builder.
        4. Injects Session Memory (Discovered symbol translations & tags).
        5. Invokes Model Orchestration (NVIDIA -> Gemini -> Local).
        6. Verifies response using Model Agreement comparisons.
        7. Audits counting questions using Counting Engine.
        8. Computes intent-aware confidence scores.
        9. Highlights grounding annotations on the current page.
        10. Updates Session Memory with discovered details.
        """
        # Load settings defaults if not present
        if settings is None:
            settings = {
                "enable_smart_retrieval": True,
                "enable_full_document": True,
                "enable_cross_sheet": True,
                "enable_gemini_verification": True,
                "enable_local_verification": True,
                "enable_counting_verification": True
            }

        page_entry = evidence_store.get_page(page_num)
        if not page_entry:
            return {
                "answer": "HUMAN REVIEW REQUIRED",
                "confidence_score": 0.0,
                "confidence_label": "Human Review Required",
                "page_number": page_num,
                "bbox": [],
                "evidence_found": False,
                "review_required": True,
                "reasoning_summary": f"Target page {page_num} not found in repository.",
                "annotated_image_path": "",
                "question_analysis": {"question_type": "general", "retrieval_strategy": "", "reasoning_strategy": ""},
                "retrieved_context": "",
                "model_verifier_results": {"agreement_score": 0.0, "disagreement_reason": "Page missing", "verification_summary": ""},
                "confidence_breakdown": {},
                "audit_trail": {"model_used": "System Error", "fallback_triggered": False, "verified": False, "validation_msg": "Page missing"}
            }

        image_path = Path(page_entry.image_path)
        doc_intel = getattr(evidence_store.repository, "doc_intelligence", {})
        
        # Identify definition pages from doc_intelligence
        definition_pages = set()
        definition_pages.update(doc_intel.get("legend_pages", []))
        definition_pages.update(doc_intel.get("notes_pages", []))
        doc_intel = dict(doc_intel)
        doc_intel["definition_pages"] = sorted(list(definition_pages))

        # ----------------------------------------------------
        # Step 1: Question Analysis (12 intents)
        # ----------------------------------------------------
        analysis = QuestionAnalyzer.analyze_question(question)
        q_type = analysis["question_type"]
        template = TEMPLATES.get(q_type, TEMPLATES["general"])

        # ----------------------------------------------------
        # Step 2: Page Ranking
        # ----------------------------------------------------
        from core.page_ranker import PageRanker
        page_rankings = PageRanker.rank_pages(question, q_type, evidence_store, doc_intel)
        
        # Select pages based on Reasoning Mode
        if reasoning_mode == "sheet" or not settings.get("enable_smart_retrieval", True):
            ranked_pages = [page_num]
        elif reasoning_mode == "full" and settings.get("enable_full_document", True):
            ranked_pages = [p.page_number for p in evidence_store.get_all_pages()]
        else: # "smart" retrieval mode
            # Select pages with score >= 0.30
            ranked_pages = [p["page"] for p in page_rankings if p["score"] >= 0.30]
            # Ensure current page is included
            if page_num not in ranked_pages:
                ranked_pages.append(page_num)
            # Ensure legend pages are included for legend/abbrev intents
            if q_type in ["legend", "abbreviation"] and settings.get("enable_cross_sheet", True):
                for lp in doc_intel.get("legend_pages", []):
                    if lp not in ranked_pages:
                        ranked_pages.append(lp)
            # Ensure notes pages are included for notes/spec intents
            if q_type in ["notes", "specification"] and settings.get("enable_cross_sheet", True):
                for np in doc_intel.get("notes_pages", []):
                    if np not in ranked_pages:
                        ranked_pages.append(np)
            
            # For counting queries, automatically exclude legend, cover, title block, and notes pages
            if q_type == "counting":
                keywords_lower = question.lower()
                allow_legend = "legend" in keywords_lower or "symbol" in keywords_lower
                allow_notes = "note" in keywords_lower or "specification" in keywords_lower or "spec" in keywords_lower
                allow_schedule = "schedule" in keywords_lower
                allow_cover = "cover" in keywords_lower or "title" in keywords_lower
                
                filtered_pages = []
                for p in ranked_pages:
                    p_entry = evidence_store.get_page(p)
                    page_type = "drawing"
                    if p_entry:
                        page_type = p_entry.metadata.get("page_type")
                        if not page_type:
                            for pt_dict in doc_intel.get("page_types", []):
                                if pt_dict.get("page") == p:
                                    page_type = pt_dict.get("page_type")
                                    break
                    
                    if page_type == "legend" and not allow_legend:
                        continue
                    if page_type == "notes" and not allow_notes:
                        continue
                    if page_type == "schedule" and not allow_schedule:
                        continue
                    if page_type in ["cover", "title_block"] and not allow_cover:
                        continue
                    filtered_pages.append(p)
                ranked_pages = filtered_pages
                if not ranked_pages:
                    ranked_pages = [page_num]
            
            # Sort pages numerically to keep output consistent
            ranked_pages = sorted(list(set(ranked_pages)))

        # Select the highest-ranked page whose page_type == "drawing" as the image to send for single-image VLMs (NVIDIA / Qwen)
        best_drawing_page = page_num # default fallback
        for ranking in page_rankings:
            p_num = ranking["page"]
            if p_num in ranked_pages:
                p_entry = evidence_store.get_page(p_num)
                if p_entry:
                    p_type = p_entry.metadata.get("page_type")
                    if not p_type:
                        for pt_dict in doc_intel.get("page_types", []):
                            if pt_dict.get("page") == p_num:
                                p_type = pt_dict.get("page_type")
                                break
                    if not p_type:
                        if p_num in doc_intel.get("legend_pages", []):
                            p_type = "legend"
                        elif p_num in doc_intel.get("notes_pages", []):
                            p_type = "notes"
                        elif p_num in doc_intel.get("schedule_pages", []):
                            p_type = "schedule"
                        elif p_num in doc_intel.get("title_block_pages", []):
                            p_type = "title_block"
                        else:
                            p_type = "drawing"
                    
                    if p_type == "drawing":
                        best_drawing_page = p_num
                        break

        best_drawing_page_entry = evidence_store.get_page(best_drawing_page) or page_entry
        best_drawing_image_path = Path(best_drawing_page_entry.image_path)
        
        # Construct ranked_images for Gemini
        ranked_images = []
        for p in ranked_pages:
            pe = evidence_store.get_page(p)
            if pe:
                ranked_images.append((p, Path(pe.image_path)))

        primary_image_page = best_drawing_page
        logger.info(f"Q: {question[:60]} | intent: {q_type} | ranked_pages: {ranked_pages} | primary_image_page: {primary_image_page}")

        # ----------------------------------------------------
        # Step 3: Context Building & Session Memory Lookup
        # ----------------------------------------------------
        from core.context_builder import ContextBuilder
        retrieved_context = ContextBuilder.build_context(
            evidence_store, ranked_pages, question, q_type, doc_intel
        )
        
        # Inject memory context
        memory_context = ""
        if session_memory:
            memory_context = session_memory.get_injected_memory_context(question)
            if memory_context:
                retrieved_context = memory_context + retrieved_context

        ocr_elements_count = len(retrieved_context.splitlines())

        # ----------------------------------------------------
        # Step 4: Model Orchestration & Verification
        # ----------------------------------------------------
        primary_response: Optional[InterpretationResponse] = None
        verifier_response: Optional[InterpretationResponse] = None

        model_used = "NVIDIA Model"
        fallback_triggered = False
        verified = False
        validation_msg = "Passed initial check"

        # TIER 1: NVIDIA Model
        if self.nvidia_client.client_active:
            try:
                primary_response = self.nvidia_client.query(
                    best_drawing_image_path, retrieved_context, question, page_num, template
                )
                
                # Run Gemini as verification if enabled
                if settings.get("enable_gemini_verification", True):
                    try:
                        logger.info("NVIDIA Model succeeded. Querying Gemini for verification...")
                        verifier_response = self.gemini_client.query(
                            ranked_images, retrieved_context, question, page_num, template
                        )
                        verified = True
                    except Exception as gemini_ver_err:
                        logger.warning(f"Verification model (Gemini) query failed: {gemini_ver_err}")
            except Exception as nvidia_err:
                logger.warning(f"Tier 1 model (NVIDIA Model) failed: {nvidia_err}. Triggering Tier 2 (Gemini)...")
                fallback_triggered = True
                model_used = "Gemini"
        else:
            logger.info("NVIDIA Client is inactive. Defaulting to Gemini.")
            model_used = "Gemini"

        # TIER 2: Gemini 2.5 Flash (as primary if NVIDIA failed or was inactive)
        if primary_response is None:
            try:
                primary_response = self.gemini_client.query(
                    ranked_images, retrieved_context, question, page_num, template
                )
                
                # Run Qwen local model as verification if enabled
                if settings.get("enable_local_verification", True):
                    try:
                        logger.info("Gemini succeeded. Querying Local Model for verification...")
                        verifier_response = self.qwen_client.query(
                            best_drawing_image_path, retrieved_context, question, page_num, template
                        )
                        verified = True
                    except Exception as qwen_ver_err:
                        logger.warning(f"Verification model (Local Qwen) query failed: {qwen_ver_err}")
            except Exception as gemini_err:
                logger.warning(f"Tier 2 model (Gemini) failed: {gemini_err}. Triggering Tier 3 (Local Qwen)...")
                fallback_triggered = True
                model_used = "Local Model"

        # TIER 3: Local VLM Fallback (Qwen / SmolVLM)
        if primary_response is None:
            try:
                primary_response = self.qwen_client.query(
                    best_drawing_image_path, retrieved_context, question, page_num, template
                )
            except Exception as local_err:
                logger.error(f"Tier 3 Local model fallback failed: {local_err}")
                primary_response = InterpretationResponse(
                    answer="HUMAN REVIEW REQUIRED",
                    confidence=0.0,
                    page_number=page_num,
                    bbox=[],
                    evidence_found=False,
                    review_required=True,
                    reasoning_summary=f"All model orchestration tiers (NVIDIA, Gemini, Qwen/SmolVLM) failed. Local error: {local_err}"
                )
                model_used = "None (Failure)"

        # Determine primary_image_page based on model used
        if model_used == "Gemini":
            if primary_response and primary_response.page_number in ranked_pages:
                primary_image_page = primary_response.page_number
            else:
                primary_image_page = page_num
        else:
            primary_image_page = best_drawing_page

        # ----------------------------------------------------
        # Step 5: Model Comparison & Verification
        # ----------------------------------------------------
        agreement_score = 1.0
        verifier_results = {
            "agreement_score": 1.0,
            "disagreement_reason": "No model comparison performed (verification disabled or single model executed).",
            "verification_summary": "Single model response resolved."
        }

        if primary_response and verifier_response:
            verifier_results = ModelVerifier.verify(primary_response, verifier_response)
            agreement_score = verifier_results["agreement_score"]

        # ----------------------------------------------------
        # Step 6: Counting Engine (V3 upgrades)
        # ----------------------------------------------------
        counting_results = None
        if q_type == "counting" and primary_response.evidence_found and settings.get("enable_counting_verification", True):
            from core.counting_engine import CountingEngine
            counting_results = CountingEngine.audit_count(
                question, primary_response, verifier_response, evidence_store, ranked_pages
            )
            
            # Format count in answer if audit found discrepancies
            if counting_results["verification_support"] < 0.8:
                validation_msg = f"Counting Audit Warning: Disagreement between visual items count ({counting_results['visual_count']}) and OCR items ({counting_results['ocr_count']})."

        # ----------------------------------------------------
        # Step 7: OCR Cross-Verification (Validator)
        # ----------------------------------------------------
        is_valid, validation_msg_val, ocr_matches = self.validator.validate(primary_response, page_entry, evidence_store)
        if "OCR Cross-Verification Warning" in validation_msg_val:
            validation_msg = validation_msg_val
        
        # Calculate OCR match score
        keywords = self.validator._extract_keywords(primary_response.answer)
        if keywords:
            matched_keywords = {kw for kw in keywords if any(kw.lower() in m.text.lower() for m in ocr_matches)}
            ocr_match_score = len(matched_keywords) / len(keywords)
        else:
            ocr_match_score = 1.0

        # ----------------------------------------------------
        # Step 8: Confidence Engine V3 (Intent-Aware)
        # ----------------------------------------------------
        confidence_result = self.confidence_engine.calculate_final_confidence(
            model_confidence=primary_response.confidence,
            ocr_match_score=ocr_match_score,
            agreement_score=agreement_score,
            has_bbox=len(primary_response.bbox) > 0,
            ocr_elements_count=ocr_elements_count,
            intent=q_type,
            counting_results=counting_results,
            review_required=primary_response.review_required,
            evidence_found=primary_response.evidence_found,
            visual_evidence_only=((not primary_response.evidence_found) and len(primary_response.bbox) > 0)
        )

        final_conf = confidence_result["final_score"]
        conf_label = confidence_result["confidence_label"]

        # ----------------------------------------------------
        # Step 9: Apply V3 Review Mode (Never suppress answers)
        # ----------------------------------------------------
        review_required = primary_response.review_required
        if final_conf < config.CONFIDENCE_THRESHOLD or not is_valid or agreement_score < 0.5:
            review_required = True

        # Enforce consistency capping on the final packaged result
        if review_required:
            final_conf = min(final_conf, 0.69)
        visual_evidence_only = (not primary_response.evidence_found) and len(primary_response.bbox) > 0
        if not primary_response.evidence_found:
            if not visual_evidence_only:
                final_conf = min(final_conf, 0.59)
        if review_required and not primary_response.evidence_found:
            if not visual_evidence_only:
                final_conf = min(final_conf, 0.49)

        # Recalculate final label
        if final_conf >= 0.90:
            conf_label = "High"
        elif final_conf >= 0.70:
            conf_label = "Medium"
        elif final_conf >= 0.50:
            conf_label = "Low"
        else:
            conf_label = "Human Review Required"

        # ----------------------------------------------------
        # Step 10: Update Session Memory
        # ----------------------------------------------------
        if session_memory and not review_required and final_conf >= 0.70:
            session_memory.add_qa(question, primary_response.answer)

        # ----------------------------------------------------
        # Step 11: Visual Grounding Highlights
        # ----------------------------------------------------
        annotated_image_path = ""
        primary_page_entry = evidence_store.get_page(primary_image_page) or page_entry
        primary_image_path = Path(primary_page_entry.image_path)
        output_name = f"{evidence_store.repository.file_hash}_p{primary_image_page}_annotated.png"
        target_annotated_path = config.ANNOTATIONS_DIR / output_name

        if primary_response.bbox:
            try:
                self.grounding_engine.annotate_image(
                    image_path=primary_image_path,
                    bboxes=primary_response.bbox,
                    output_path=target_annotated_path
                )
                annotated_image_path = str(target_annotated_path.resolve())
            except Exception as ground_err:
                logger.error(f"Failed to generate visual grounding highlights: {ground_err}")
                shutil.copy(primary_image_path, target_annotated_path)
                annotated_image_path = str(target_annotated_path.resolve())
        else:
            shutil.copy(primary_image_path, target_annotated_path)
            annotated_image_path = str(target_annotated_path.resolve())

        # Build comprehensive audit logs
        return {
            "answer": primary_response.answer,
            "confidence_score": final_conf,
            "confidence_label": conf_label,
            "page_number": primary_response.page_number,
            "bbox": primary_response.bbox,
            "evidence_found": primary_response.evidence_found,
            "review_required": review_required,
            "reasoning_summary": primary_response.reasoning_summary,
            "annotated_image_path": annotated_image_path,
            "primary_image_page": primary_image_page,
            
            # Metadata properties for Debug Panels
            "question_analysis": {
                "question_type": q_type,
                "retrieval_strategy": analysis["retrieval_strategy"],
                "reasoning_strategy": analysis["reasoning_strategy"]
            },
            "retrieved_context": retrieved_context,
            "model_verifier_results": {
                "agreement_score": agreement_score,
                "disagreement_reason": verifier_results["disagreement_reason"],
                "verification_summary": verifier_results["verification_summary"],
                "nvidia_output": primary_response.answer if model_used == "NVIDIA Model" else ("N/A" if not self.nvidia_client.client_active else "Error/Fallback"),
                "gemini_output": verifier_response.answer if (model_used == "NVIDIA Model" and verifier_response) else (primary_response.answer if model_used == "Gemini" else "N/A"),
                "qwen_output": verifier_response.answer if (model_used == "Gemini" and verifier_response) else (primary_response.answer if model_used == "Local Model" else "N/A")
            },
            "confidence_breakdown": confidence_result["breakdown"],
            "confidence_audit": confidence_result.get("confidence_audit", {}),
            "audit_trail": {
                "model_used": model_used,
                "fallback_triggered": fallback_triggered,
                "verified": verified,
                "validation_msg": validation_msg + f" | OCR Match score: {ocr_match_score:.2f} | BBox valid: {len(primary_response.bbox) > 0}"
            },
            # V3 Upgrade Metrics
            "reasoning_mode": reasoning_mode,
            "ranked_pages_scores": page_rankings,
            "retrieved_pages": ranked_pages,
            "counting_results": counting_results,
            "page_classifications": doc_intel.get("page_types", [])
        }
