import logging
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw

from schemas.document_schema import LayoutRegion

logger = logging.getLogger(__name__)

class AnnotationEngine:
    @staticmethod
    def annotate_layout_regions(
        image_path: Path,
        layout_regions: List[LayoutRegion],
        output_path: Path
    ) -> None:
        """
        Draws semi-transparent color overlays outlining the extracted layout regions:
        - title_block: Purple
        - legend: Green
        - notes: Blue
        - drawing_area: Orange (subtle)
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Source image for layout annotation not found: {image_path}")

        logger.info(f"AnnotationEngine: Drawing layout boundaries on {image_path.name}...")

        # Color schemes mapping: (fill_rgba, outline_rgba)
        region_styles = {
            "title_block": ((156, 39, 176, 30), (156, 39, 176, 255), "Title Block"),
            "legend": ((76, 175, 80, 30), (76, 175, 80, 255), "Legend"),
            "notes": ((33, 150, 243, 30), (33, 150, 243, 255), "Notes"),
            "drawing_area": ((255, 152, 0, 15), (255, 152, 0, 180), "Drawing Area")
        }

        with Image.open(image_path) as img:
            base_img = img.convert("RGBA")
            width, height = base_img.size

            overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for r in layout_regions:
                style = region_styles.get(r.region_type)
                if not style:
                    continue
                
                fill_color, outline_color, label = style
                ymin, xmin, ymax, xmax = r.bbox

                # Scale coordinates
                x0 = xmin * width
                y0 = ymin * height
                x1 = xmax * width
                y1 = ymax * height

                # Draw filled transparent rectangle and outline
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=fill_color,
                    outline=outline_color,
                    width=4
                )

                # Draw text label in top left of region
                # Draw a backing rectangle for text readability
                label_y = max(y0, 10.0)
                label_x = max(x0, 10.0)
                text_w = len(label) * 8 + 10  # rough estimate for default font
                text_h = 18
                draw.rectangle(
                    [label_x, label_y, label_x + text_w, label_y + text_h],
                    fill=outline_color
                )
                
                # Write label
                draw.text(
                    (label_x + 5, label_y + 3),
                    label,
                    fill=(255, 255, 255, 255)
                )

            # Composite overlay
            final_img = Image.alpha_composite(base_img, overlay)
            final_img.convert("RGB").save(output_path, "PNG")
            logger.info(f"AnnotationEngine: Saved layout annotated image to {output_path}")
