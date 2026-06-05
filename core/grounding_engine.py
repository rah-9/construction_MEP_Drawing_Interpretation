import logging
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

class GroundingEngine:
    @staticmethod
    def annotate_image(
        image_path: Path, 
        bboxes: List[List[float]], 
        output_path: Path
    ) -> None:
        """
        Draws semi-transparent highlighted bounding boxes on the drawing page.
        Coordinates are [ymin, xmin, ymax, xmax] in range 0.0 - 1.0.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Source image for grounding not found: {image_path}")

        logger.info(f"GroundingEngine: Highlighting {len(bboxes)} bounding boxes on {image_path.name}...")
        
        # Load base image and convert to RGBA to support transparency
        with Image.open(image_path) as img:
            base_img = img.convert("RGBA")
            width, height = base_img.size

            # Create transparent overlay
            overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for box in bboxes:
                ymin, xmin, ymax, xmax = box
                
                # Scale coordinates
                x0 = xmin * width
                y0 = ymin * height
                x1 = xmax * width
                y1 = ymax * height

                # Draw filled transparent yellow box and solid red border
                # yellow fill: (255, 235, 59, 80) -> transparent yellow
                # red outline: (244, 67, 54, 255) -> solid red
                draw.rectangle(
                    [x0, y0, x1, y1], 
                    fill=(255, 235, 59, 80), 
                    outline=(244, 67, 54, 255), 
                    width=4
                )

            # Composite the overlay onto the original image
            final_img = Image.alpha_composite(base_img, overlay)
            
            # Save as RGB PNG
            final_img.convert("RGB").save(output_path, "PNG")
            logger.info(f"GroundingEngine: Saved annotated image to {output_path}")
