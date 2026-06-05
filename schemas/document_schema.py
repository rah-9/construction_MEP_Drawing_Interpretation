from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class PageDimension(BaseModel):
    width: float = Field(description="Width of the page in PDF points (1/72 inch).")
    height: float = Field(description="Height of the page in PDF points (1/72 inch).")

class OCRElement(BaseModel):
    text: str = Field(description="The extracted text string.")
    bbox: List[float] = Field(
        description="Bounding box coordinates [ymin, xmin, ymax, xmax] in normalized (0.0 to 1.0) coordinates."
    )
    confidence: float = Field(description="OCR engine confidence score from 0.0 to 1.0.")
    page_number: int = Field(description="1-based page number.")
    evidence_type: str = Field(
        default="ocr_text",
        description="The type of evidence (e.g. 'legend', 'note', 'title_block', 'ocr_text')."
    )

class LayoutRegion(BaseModel):
    region_type: str = Field(description="Type of region: 'legend', 'notes', 'drawing_area', 'title_block'.")
    bbox: List[float] = Field(
        description="Bounding box [ymin, xmin, ymax, xmax] in normalized (0.0 to 1.0) coordinates."
    )
    bbox_pts: List[float] = Field(
        description="Bounding box [xmin, ymin, xmax, ymax] in original PDF point coordinates."
    )

class PageEntry(BaseModel):
    page_number: int = Field(description="1-based page index.")
    dimensions: PageDimension = Field(description="Original dimensions of the page.")
    embedded_text: str = Field(default="", description="Any raw embedded text extracted directly from the PDF.")
    image_path: str = Field(description="Absolute path to the high-resolution rendered PNG image of the page.")
    preview_path: str = Field(description="Absolute path to the thumbnail/preview image of the page.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary for arbitrary annotations.")
    ocr_elements: List[OCRElement] = Field(default_factory=list, description="Extracted OCR words and bounding boxes.")
    layout_regions: List[LayoutRegion] = Field(default_factory=list, description="Identified layout zones on the page.")
    page_hash: str = Field(description="SHA-256 hash of the rendered page image to detect single page changes.")

class DocumentRepository(BaseModel):
    version: str = Field(default="2.0", description="Schema version of the repository.")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="Timestamp of when the repository was created."
    )
    document_name: str = Field(description="Original name of the uploaded PDF file.")
    file_hash: str = Field(description="SHA-256 hash of the entire PDF file to serve as a cache key.")
    pages: List[PageEntry] = Field(default_factory=list, description="List of processed page entries.")
    doc_intelligence: Dict[str, Any] = Field(
        default_factory=lambda: {
            "legend_pages": [],
            "notes_pages": [],
            "schedule_pages": [],
            "title_block_pages": [],
            "equipment_tags": [],
            "symbols": [],
            "abbreviations": [],
            "specifications": []
        },
        description="Document-wide metadata extracted once upon ingestion."
    )
