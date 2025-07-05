# src/compositor.py
import cv2
import numpy as np
from typing import List, Tuple, Optional


class DynamicCompositor:
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        
    def calculate_homography(self, 
                           src_corners: List[Tuple[int, int]], 
                           dst_corners: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """Calculate perspective transformation matrix"""
        if len(src_corners) != 4 or len(dst_corners) != 4:
            return None
            
        src_pts = np.array(src_corners, dtype=np.float32)
        dst_pts = np.array(dst_corners, dtype=np.float32)
        
        try:
            homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
            return homography
        except Exception as e:
            print(f"Error calculating homography: {e}")
            return None
    
    def warp_artwork(self, 
                     artwork_frame: np.ndarray, 
                     homography: np.ndarray, 
                     output_shape: Tuple[int, int]) -> np.ndarray:
        """Apply perspective warp to artwork"""
        warped = cv2.warpPerspective(artwork_frame, homography, output_shape)
        return warped
    
    def create_mask(self, 
                    corners: List[Tuple[int, int]], 
                    frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create mask for marquee area"""
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        
        # Create polygon mask
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Optional: feather edges for smoother blend
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def blend_frames(self, 
                     background: np.ndarray, 
                     warped_artwork: np.ndarray, 
                     mask: np.ndarray, 
                     blend_mode='normal') -> np.ndarray:
        """Blend warped artwork into background"""
        # Normalize mask to 0-1 range
        mask_norm = mask.astype(float) / 255.0
        mask_3d = np.stack([mask_norm] * 3, axis=2)
        
        if blend_mode == 'normal':
            # Simple alpha blending
            result = background * (1 - mask_3d) + warped_artwork * mask_3d
        elif blend_mode == 'multiply':
            # Multiply blend mode
            blended = (background.astype(float) * warped_artwork.astype(float)) / 255.0
            result = background * (1 - mask_3d) + blended * mask_3d
        elif blend_mode == 'screen':
            # Screen blend mode
            inv_bg = 255 - background.astype(float)
            inv_art = 255 - warped_artwork.astype(float)
            blended = 255 - (inv_bg * inv_art) / 255.0
            result = background * (1 - mask_3d) + blended * mask_3d
        else:
            result = background * (1 - mask_3d) + warped_artwork * mask_3d
            
        return result.astype(np.uint8)
    
    def color_match(self, 
                    artwork: np.ndarray, 
                    background: np.ndarray, 
                    mask: np.ndarray) -> np.ndarray:
        """Match artwork colors to background lighting"""
        # Extract background colors in marquee area
        masked_bg = cv2.bitwise_and(background, background, mask=mask)
        
        # Calculate average color in mask area
        mask_area = np.sum(mask > 0)
        if mask_area == 0:
            return artwork
            
        avg_bg_color = np.sum(masked_bg, axis=(0, 1)) / mask_area
        
        # Adjust artwork to match background tone
        artwork_float = artwork.astype(float)
        
        # Simple color temperature adjustment
        color_factor = avg_bg_color / np.mean(avg_bg_color)
        adjusted_artwork = artwork_float * color_factor
        
        # Clamp values
        adjusted_artwork = np.clip(adjusted_artwork, 0, 255)
        
        return adjusted_artwork.astype(np.uint8)
