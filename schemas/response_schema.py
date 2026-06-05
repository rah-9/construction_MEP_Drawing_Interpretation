from typing import List, Optional
from pydantic import BaseModel, Field

class InterpretationResponse(BaseModel):
    answer: str = Field(
        description="The detailed answer to the user's question, based strictly on the drawing evidence."
    )
    confidence: float = Field(
        description="The raw model confidence score from 0.0 to 1.0. Low values suggest human review."
    )
    page_number: int = Field(
        description="The 1-based page number where the evidence was located."
    )
    bbox: List[List[float]] = Field(
        default_factory=list,
        description="A list of bounding boxes highlighting the evidence. Each box is [ymin, xmin, ymax, xmax] in normalized (0.0 to 1.0) coordinates."
    )
    evidence_found: bool = Field(
        description="True if direct, visible evidence was found to answer the question, False otherwise."
    )
    review_required: bool = Field(
        description="True if the model is uncertain, evidence is missing, or human validation is needed."
    )
    reasoning_summary: str = Field(
        description="A short explanation of how the model found the answer and determined the confidence level."
    )
