import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import fitz  # PyMuPDF
from PIL import Image

import config
from schemas.document_schema import DocumentRepository, PageEntry, PageDimension
from core.layout_extractor import LayoutExtractor
from core.ocr_engine import OCREngine

logger = logging.getLogger(__name__)

def calculate_file_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class PDFProcessor:
    @staticmethod
    def get_cached_repository(file_hash: str) -> Optional[DocumentRepository]:
        """Check if a cached document repository exists for the given file hash."""
        repo_path = config.REPOSITORIES_DIR / f"{file_hash}.json"
        if repo_path.exists():
            try:
                with open(repo_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                repo = DocumentRepository(**data)
                
                # Check for schema compatibility (v1 or v2)
                major_version = repo.version.split(".")[0]
                if major_version in ["1", "2"]:
                    needs_upgrade = (major_version == "1") or ("page_types" not in repo.doc_intelligence)
                    if needs_upgrade:
                        logger.info(f"Upgrading/enriching cached repository for hash: {file_hash}")
                        from core.document_intelligence import DocumentIntelligence
                        repo.doc_intelligence = DocumentIntelligence.analyze(repo)
                        repo.version = "2.0"
                        PDFProcessor.save_repository(repo)
                    else:
                        logger.info(f"Loaded cached repository for hash: {file_hash}")
                    return repo
            except Exception as e:
                logger.error(f"Error loading cached repository: {e}")
        return None

    @staticmethod
    def save_repository(repository: DocumentRepository) -> Path:
        """Save a document repository to the local output folder."""
        repo_path = config.REPOSITORIES_DIR / f"{repository.file_hash}.json"
        with open(repo_path, "w", encoding="utf-8") as f:
            f.write(repository.model_dump_json(indent=2))
        logger.info(f"Saved repository to {repo_path}")
        return repo_path

    @staticmethod
    def process_pdf(pdf_path: Path, dpi: int = None) -> DocumentRepository:
        """
        Process a PDF file.
        1. Calculates file hash.
        2. Returns cached repository if it exists.
        3. Renders pages, extracts embedded text, and saves to repository.
        """
        if dpi is None:
            dpi = config.RENDER_DPI

        file_hash = calculate_file_sha256(pdf_path)
        cached_repo = PDFProcessor.get_cached_repository(file_hash)
        if cached_repo:
            return cached_repo

        logger.info(f"Processing new PDF: {pdf_path.name} (DPI: {dpi})")
        
        # Copy to uploads folder with hash-based name
        dest_pdf_path = config.UPLOADS_DIR / f"{file_hash}.pdf"
        if not dest_pdf_path.exists():
            import shutil
            shutil.copy(pdf_path, dest_pdf_path)

        doc = fitz.open(pdf_path)
        pages_entries = []
        ocr_engine = OCREngine()

        try:
            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                page = doc.load_page(page_idx)
                rect = page.rect
                width_pts = rect.width
                height_pts = rect.height

                # Render high-res image for drawing
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                img_name = f"{file_hash}_p{page_num}.png"
                img_path = config.RENDERED_PAGES_DIR / img_name
                pix.save(str(img_path))

                # Render low-res preview image for UI
                preview_zoom = 75.0 / 72.0
                preview_mat = fitz.Matrix(preview_zoom, preview_zoom)
                preview_pix = page.get_pixmap(matrix=preview_mat, alpha=False)
                preview_name = f"{file_hash}_p{page_num}_preview.png"
                preview_path = config.RENDERED_PAGES_DIR / preview_name
                preview_pix.save(str(preview_path))

                # Compute page hash (hash of image bytes)
                with open(img_path, "rb") as f:
                    page_hash = hashlib.sha256(f.read()).hexdigest()

                # Extract embedded text
                embedded_text = page.get_text("text")

                # Extract layout regions
                layout_regions = LayoutExtractor.extract_regions(page)

                # Extract OCR elements
                ocr_elements = ocr_engine.extract_text(page, img_path, page_num, layout_regions)

                page_entry = PageEntry(
                    page_number=page_num,
                    dimensions=PageDimension(width=width_pts, height=height_pts),
                    embedded_text=embedded_text,
                    image_path=str(img_path.resolve()),
                    preview_path=str(preview_path.resolve()),
                    page_hash=page_hash,
                    ocr_elements=ocr_elements,
                    layout_regions=layout_regions,
                    metadata={
                        "rendered_dpi": dpi,
                        "rotation": page.rotation
                    }
                )
                pages_entries.append(page_entry)

        finally:
            doc.close()

        repo = DocumentRepository(
            document_name=pdf_path.name,
            file_hash=file_hash,
            pages=pages_entries
        )
        
        # Run Document Intelligence analysis
        from core.document_intelligence import DocumentIntelligence
        repo.doc_intelligence = DocumentIntelligence.analyze(repo)
        
        PDFProcessor.save_repository(repo)
        return repo
