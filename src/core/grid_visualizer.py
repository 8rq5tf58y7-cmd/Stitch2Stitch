"""
Grid visualization for 2D alignment preview
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class GridVisualizer:
    """Visualize grid alignment without merging"""
    
    def save_grid(self, grid_layout: Dict, output_path: str):
        """
        Save grid layout visualization
        
        Args:
            grid_layout: Grid layout dictionary
            output_path: Output file path
        """
        images = grid_layout['images']
        positions = grid_layout['positions']
        grid_size = grid_layout['grid_size']
        
        if not images or not positions:
            logger.error("Empty grid layout")
            return
        
        # Calculate canvas size
        max_x = max(pos['x'] + images[pos['image_idx']]['shape'][1] 
                   for pos in positions)
        max_y = max(pos['y'] + images[pos['image_idx']]['shape'][0] 
                   for pos in positions)
        
        # Create canvas
        canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        
        # Place images on canvas
        for pos in positions:
            img_data = images[pos['image_idx']]
            img = img_data['image']
            alpha = img_data.get('alpha')
            
            x = pos['x']
            y = pos['y']
            h, w = img.shape[:2]
            
            if alpha is not None:
                # Handle transparency
                alpha_3d = alpha[:, :, np.newaxis] / 255.0
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                canvas[y:y+h, x:x+w] = (
                    canvas[y:y+h, x:x+w] * (1 - alpha_3d) +
                    img_rgb * alpha_3d
                ).astype(np.uint8)
            else:
                canvas[y:y+h, x:x+w] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() in ['.tif', '.tiff']:
            import tifffile
            tifffile.imwrite(str(output_path), canvas, compression='lzw')
        else:
            cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Grid visualization saved to {output_path}")

