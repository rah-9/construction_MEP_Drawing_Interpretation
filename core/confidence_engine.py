import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ConfidenceEngine:
    def calculate_final_confidence(
        self,
        model_confidence: float,
        ocr_match_score: float,
        agreement_score: float,
        has_bbox: bool,
        ocr_elements_count: int,
        intent: str = "general",
        counting_results: Optional[Dict[str, Any]] = None,
        review_required: bool = False,
        evidence_found: bool = True,
        visual_evidence_only: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates evidence-based confidence using intent-specific formulas:
        - Legend/Abbreviation: 40% OCR Match, 30% Model Agreement, 20% Model Confidence, 10% BBox
        - Notes/Specification: 40% OCR Match, 30% Model Agreement, 20% Model Confidence, 10% BBox
        - Schedule/Equipment: 40% OCR Match, 30% Model Agreement, 20% Model Confidence, 10% BBox
        - Counting: 35% OCR Support, 25% Verification Support, 20% Model Agreement, 20% Visual Support
        - Cross-Sheet: 35% OCR Match, 30% Density, 25% Model Agreement, 10% Model Confidence
        - Default: 30% OCR Match, 25% Model Agreement, 20% BBox, 15% Density, 10% Model Confidence
        """
        mc = max(0.0, min(1.0, model_confidence))
        oms = max(0.0, min(1.0, ocr_match_score))
        ag = max(0.0, min(1.0, agreement_score))
        bbox_val = 1.0 if has_bbox else 0.0
        
        # Calculate evidence density score
        density_score = min(1.0, ocr_elements_count / 5.0) if ocr_elements_count > 0 else 0.0

        # Custom Weights by Intent
        if intent in ["legend", "abbreviation", "notes", "specification", "schedule", "equipment"]:
            ocr_match_contrib = 0.40 * oms
            agreement_contrib = 0.30 * ag
            bbox_contrib = 0.10 * bbox_val
            density_contrib = 0.0
            model_conf_contrib = 0.20 * mc
        elif intent == "counting" and counting_results:
            s_ocr = counting_results.get("ocr_support", 0.0)
            s_vis = counting_results.get("visual_support", 0.0)
            s_ver = counting_results.get("verification_support", 0.0)
            
            ocr_match_contrib = 0.35 * s_ocr
            agreement_contrib = 0.20 * ag
            bbox_contrib = 0.20 * s_vis
            density_contrib = 0.0
            model_conf_contrib = 0.25 * s_ver
            
            # Update inputs for UI visualization mapping
            oms = s_ocr
            bbox_val = s_vis
            density_score = 0.0
            mc = s_ver
        elif intent == "cross_sheet":
            ocr_match_contrib = 0.35 * oms
            agreement_contrib = 0.25 * ag
            bbox_contrib = 0.0
            density_contrib = 0.30 * density_score
            model_conf_contrib = 0.10 * mc
        else:
            # Default / general fallback
            ocr_match_contrib = 0.30 * oms
            agreement_contrib = 0.25 * ag
            bbox_contrib = 0.20 * bbox_val
            density_contrib = 0.15 * density_score
            model_conf_contrib = 0.10 * mc

        final_score = (
            ocr_match_contrib + 
            agreement_contrib + 
            bbox_contrib + 
            density_contrib + 
            model_conf_contrib
        )

        final_score = max(0.0, min(1.0, final_score))

        # Enforce consistency rules
        if review_required:
            final_score = min(final_score, 0.69)
        if not evidence_found:
            if not visual_evidence_only:
                final_score = min(final_score, 0.59)
            elif not has_bbox:
                final_score = min(final_score, 0.59)
        if not has_bbox:
            bbox_val = 0.0
        if review_required and not evidence_found:
            if not visual_evidence_only:
                final_score = min(final_score, 0.49)
            elif not has_bbox:
                final_score = min(final_score, 0.49)

        # Determine label
        if final_score >= 0.90:
            label = "High"
        elif final_score >= 0.70:
            label = "Medium"
        elif final_score >= 0.50:
            label = "Low"
        else:
            label = "Human Review Required"

        # Construct audit dict
        audit_log = {
            "intent": intent,
            "ocr_score": round(oms, 3),
            "bbox_score": round(bbox_val, 3),
            "agreement_score": round(ag, 3),
            "model_confidence": round(mc, 3),
            "review_required": review_required,
            "evidence_found": evidence_found,
            "final_confidence": round(final_score, 3)
        }
        logger.info(f"Confidence Audit: {audit_log}")

        return {
            "final_score": round(final_score, 2),
            "confidence_label": label,
            "confidence_audit": audit_log,
            "breakdown": {
                "ocr_contribution": round(ocr_match_contrib, 3),
                "agreement_contribution": round(agreement_contrib, 3),
                "bbox_contribution": round(bbox_contrib, 3),
                "density_contribution": round(density_contrib, 3),
                "model_confidence_contribution": round(model_conf_contrib, 3),
                "ocr_match_score": round(oms, 2),
                "agreement_score": round(ag, 2),
                "has_bbox": bbox_val > 0.0,
                "density_score": round(density_score, 2),
                "model_confidence": round(mc, 2),
                "ocr_elements_count": ocr_elements_count,
                "visual_evidence_only": (not evidence_found) and has_bbox
            }
        }
