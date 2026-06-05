import logging
import re
from typing import Dict, Any, List
from schemas.response_schema import InterpretationResponse

logger = logging.getLogger(__name__)

class ModelVerifier:
    @staticmethod
    def _calculate_jaccard_similarity(str1: str, str2: str) -> float:
        """Calculate the token-based Jaccard similarity between two strings."""
        tokens1 = set(re.findall(r"\w+", str1.lower()))
        tokens2 = set(re.findall(r"\w+", str2.lower()))
        
        if not tokens1 and not tokens2:
            return 1.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)

    @staticmethod
    def _calculate_box_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection-over-Union (IoU) of two normalized bounding boxes."""
        ymin1, xmin1, ymax1, xmax1 = box1
        ymin2, xmin2, ymax2, xmax2 = box2

        # Intersection bounds
        iymin = max(ymin1, ymin2)
        ixmin = max(xmin1, xmin2)
        iymax = min(ymax1, ymax2)
        ixmax = min(xmax1, xmax2)

        if iymax > iymin and ixmax > ixmin:
            inter_area = (iymax - iymin) * (ixmax - ixmin)
        else:
            inter_area = 0.0

        # Areas of individual boxes
        area1 = (ymax1 - ymin1) * (xmax1 - xmin1)
        area2 = (ymax2 - ymin2) * (xmax2 - xmin2)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def _calculate_bbox_similarity(bboxes1: List[List[float]], bboxes2: List[List[float]]) -> float:
        """Calculate similarity between two lists of bounding boxes."""
        if not bboxes1 and not bboxes2:
            return 1.0
        if not bboxes1 or not bboxes2:
            return 0.0

        # Calculate max IoU for each box in bboxes1 relative to bboxes2
        ious = []
        for b1 in bboxes1:
            max_iou = 0.0
            for b2 in bboxes2:
                iou = ModelVerifier._calculate_box_iou(b1, b2)
                if iou > max_iou:
                    max_iou = iou
            ious.append(max_iou)

        return sum(ious) / len(ious)

    @classmethod
    def verify(
        cls, 
        primary_response: InterpretationResponse, 
        verifier_response: InterpretationResponse
    ) -> Dict[str, Any]:
        """
        Compares two model responses for agreement:
        1. Compares textual answer content (Jaccard similarity).
        2. Compares bounding box overlap (IoU).
        3. Checks page number consistency.
        """
        # 1. Page similarity
        page_similarity = 1.0 if primary_response.page_number == verifier_response.page_number else 0.0

        # 2. Text similarity
        text_similarity = cls._calculate_jaccard_similarity(
            primary_response.answer, 
            verifier_response.answer
        )

        # 3. Coordinate similarity
        bbox_similarity = cls._calculate_bbox_similarity(
            primary_response.bbox, 
            verifier_response.bbox
        )

        # Weighted agreement score: 50% Text, 35% BBox coordinates, 15% Page
        agreement_score = (0.50 * text_similarity) + (0.35 * bbox_similarity) + (0.15 * page_similarity)

        # Determine reasons if disagreement exists
        disagreement_reasons = []
        if text_similarity < 0.6:
            disagreement_reasons.append("Models resolved different text descriptions.")
        if bbox_similarity < 0.4:
            disagreement_reasons.append("Visual evidence locations (bounding boxes) do not overlap.")
        if page_similarity == 0.0:
            disagreement_reasons.append("Models referenced different drawing pages.")

        reason_str = " ".join(disagreement_reasons) if disagreement_reasons else "Models are in high agreement."
        
        summary = f"Text Similarity: {text_similarity:.2f}, BBox IoU: {bbox_similarity:.2f}. {reason_str}"
        logger.info(f"ModelVerifier: Agreement Score: {agreement_score:.2f}. {summary}")

        return {
            "agreement_score": round(agreement_score, 2),
            "disagreement_reason": reason_str,
            "verification_summary": summary
        }
