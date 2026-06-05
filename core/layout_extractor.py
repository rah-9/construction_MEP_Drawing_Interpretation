import logging
from typing import List
import fitz  # PyMuPDF
from schemas.document_schema import LayoutRegion

logger = logging.getLogger(__name__)

class LayoutExtractor:
    @staticmethod
    def extract_regions(page: fitz.Page) -> List[LayoutRegion]:
        """
        Identify major regions of engineering drawings:
        - Legend
        - Notes
        - Drawing Area
        - Title Block
        
        Uses geometric templates and keyword searches to refine bounding boxes.
        """
        rect = page.rect
        width = rect.width
        height = rect.height

        regions = []

        # ----------------------------------------------------
        # 1. Heuristics for Title Block (Bottom-Right or Right Margin)
        # ----------------------------------------------------
        title_block_found = False
        title_rects = page.search_for("TITLE") + page.search_for("SHEET") + page.search_for("DRAWING NO")
        
        if title_rects:
            # Find the right-most or bottom-most title indicators
            best_rect = max(title_rects, key=lambda r: (r.x0 + r.y0))
            # If it's in the lower section, assume a bottom-right title block
            if best_rect.y0 > height * 0.7 and best_rect.x0 > width * 0.6:
                xmin_pts = max(width * 0.6, best_rect.x0 - 50)
                ymin_pts = max(height * 0.7, best_rect.y0 - 20)
                xmax_pts = width
                ymax_pts = height
                title_block_found = True
            # If it's on the far right, assume a right vertical column title block
            elif best_rect.x0 > width * 0.8:
                xmin_pts = max(width * 0.8, best_rect.x0 - 20)
                ymin_pts = 0.0
                xmax_pts = width
                ymax_pts = height
                title_block_found = True

        if not title_block_found:
            # Default to standard bottom-right title block (typical for drawings)
            xmin_pts = width * 0.70
            ymin_pts = height * 0.75
            xmax_pts = width
            ymax_pts = height

        regions.append(LayoutRegion(
            region_type="title_block",
            bbox=[ymin_pts/height, xmin_pts/width, ymax_pts/height, xmax_pts/width],
            bbox_pts=[xmin_pts, ymin_pts, xmax_pts, ymax_pts]
        ))

        # Save title block coordinates to exclude them from other regions
        tb_xmin, tb_ymin, tb_xmax, tb_ymax = xmin_pts, ymin_pts, xmax_pts, ymax_pts

        # ----------------------------------------------------
        # 2. Heuristics for General Notes
        # ----------------------------------------------------
        notes_found = False
        notes_rects = page.search_for("GENERAL NOTES") + page.search_for("NOTES")
        if notes_rects:
            # Take the first match
            first_note = notes_rects[0]
            # General notes are usually a column
            n_xmin = max(0.0, first_note.x0 - 10)
            n_ymin = max(0.0, first_note.y0 - 10)
            n_xmax = min(width, first_note.x1 + 250)
            # Extends down, but stops before title block if it overlaps
            n_ymax = height * 0.7
            if n_xmin >= tb_xmin and n_ymin <= tb_ymax:
                # notes are in right column, stop at title block top
                n_ymax = tb_ymin
            notes_found = True
        
        if not notes_found:
            # Default notes region (right margin, top half)
            n_xmin = width * 0.75
            n_ymin = height * 0.05
            n_xmax = width * 0.98
            n_ymax = height * 0.45

        regions.append(LayoutRegion(
            region_type="notes",
            bbox=[n_ymin/height, n_xmin/width, n_ymax/height, n_xmax/width],
            bbox_pts=[n_xmin, n_ymin, n_xmax, n_ymax]
        ))

        # ----------------------------------------------------
        # 3. Heuristics for Legend
        # ----------------------------------------------------
        legend_found = False
        legend_rects = page.search_for("LEGEND") + page.search_for("SYMBOLS")
        if legend_rects:
            first_leg = legend_rects[0]
            l_xmin = max(0.0, first_leg.x0 - 20)
            l_ymin = max(0.0, first_leg.y0 - 10)
            l_xmax = min(width, first_leg.x1 + 300)
            l_ymax = min(height, l_ymin + 400)
            
            # Adjust if it overlaps with title block
            if l_xmin >= tb_xmin and l_ymax > tb_ymin:
                l_ymax = tb_ymin
            legend_found = True

        if not legend_found:
            # Default legend region (right margin, middle/lower half)
            l_xmin = width * 0.75
            l_ymin = height * 0.45
            l_xmax = width * 0.98
            l_ymax = height * 0.75
            if l_ymax > tb_ymin:
                l_ymax = tb_ymin

        regions.append(LayoutRegion(
            region_type="legend",
            bbox=[l_ymin/height, l_xmin/width, l_ymax/height, l_xmax/width],
            bbox_pts=[l_xmin, l_ymin, l_xmax, l_ymax]
        ))

        # ----------------------------------------------------
        # 4. Drawing Area (Main section on the left/center)
        # ----------------------------------------------------
        # The main drawing canvas excludes the right columns (Legend, Notes, Title Block)
        # Typically occupies the left 75% of the page
        da_xmin = width * 0.02
        da_ymin = height * 0.02
        da_xmax = min(tb_xmin, n_xmin, l_xmin) - 10
        if da_xmax < width * 0.5:
            # Safeguard in case margins are very wide
            da_xmax = width * 0.70
        da_ymax = height * 0.95

        regions.append(LayoutRegion(
            region_type="drawing_area",
            bbox=[da_ymin/height, da_xmin/width, da_ymax/height, da_xmax/width],
            bbox_pts=[da_xmin, da_ymin, da_xmax, da_ymax]
        ))

        logger.info(f"Extracted {len(regions)} layout regions for page.")
        return regions
