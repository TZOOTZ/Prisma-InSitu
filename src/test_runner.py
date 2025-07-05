# src/test_runner.py
import cv2
import numpy as np
import os
import sys
from motion_tracker import MarqueeTracker
from compositor import DynamicCompositor


class PrismaMotionTest:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.tracker = MarqueeTracker(debug_mode=True)
        self.compositor = DynamicCompositor(debug_mode=True)
        
        # Create output directories
        os.makedirs(f"{output_dir}/tracking_debug", exist_ok=True)
        os.makedirs(f"{output_dir}/composition_tests", exist_ok=True)
        os.makedirs(f"{output_dir}/final_results", exist_ok=True)
    
    def test_tracking_only(self, background_video_path: str):
        """Test motion tracking without composition"""
        print("=== TESTING MOTION TRACKING ===")
        
        cap = cv2.VideoCapture(background_video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {background_video_path}")
            return False
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        debug_out = cv2.VideoWriter(
            f"{self.output_dir}/tracking_debug/tracking_test.mp4",
            fourcc, fps, (width, height)
        )
        
        # Read first frame for corner selection
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Cannot read first frame")
            return False
            
        # Manual corner selection
        corners = self.tracker.setup_manual_tracking(first_frame)
        if not corners:
            print("Corner selection cancelled")
            return False
            
        # Initialize trackers
        self.tracker.initialize_trackers(first_frame, corners)
        
        # Process video
        frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Track corners
            current_corners = self.tracker.track_frame(frame.copy())
            
            # Draw tracking visualization
            if current_corners:
                # Draw corners
                for i, corner in enumerate(current_corners):
                    cv2.circle(frame, corner, 8, (0, 255, 0), -1)
                    cv2.putText(frame, f'{i+1}', 
                               (corner[0]+10, corner[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw marquee outline
                pts = np.array(current_corners, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (255, 0, 0), 3)
                
                # Draw tracking info
                cv2.putText(frame, f"Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Corners: {len(current_corners)}/4", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "TRACKING LOST", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            debug_out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        debug_out.release()
        
        # Save tracking data
        self.tracker.save_tracking_data(
            f"{self.output_dir}/tracking_debug/tracking_data.json"
        )
        
        print(f"Tracking test completed. {frame_count} frames processed.")
        print(f"Tracking success rate: {len(self.tracker.tracking_history)}/{frame_count} frames")
        
        return True
    
    def test_composition(self, 
                        background_video_path: str, 
                        artwork_video_path: str, 
                        tracking_data_path: str = None):
        """Test full composition pipeline"""
        print("=== TESTING COMPOSITION ===")
        
        # Load videos
        bg_cap = cv2.VideoCapture(background_video_path)
        art_cap = cv2.VideoCapture(artwork_video_path)
        
        if not bg_cap.isOpened() or not art_cap.isOpened():
            print("Error: Cannot open video files")
            return False
        
        # Get properties
        fps = int(bg_cap.get(cv2.CAP_PROP_FPS))
        width = int(bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        comp_out = cv2.VideoWriter(
            f"{self.output_dir}/composition_tests/composition_test.mp4",
            fourcc, fps, (width, height)
        )
        
        # Load or generate tracking data
        if tracking_data_path and os.path.exists(tracking_data_path):
            import json
            with open(tracking_data_path, 'r') as f:
                tracking_data = json.load(f)
            corner_history = tracking_data['smoothed_history']
        else:
            print("No tracking data provided, running tracking first...")
            if not self.test_tracking_only(background_video_path):
                return False
            corner_history = self.tracker.smooth_tracking()
        
        # Get artwork frame for homography calculation
        ret, art_frame = art_cap.read()
        if not ret:
            print("Error: Cannot read artwork frame")
            return False
            
        art_height, art_width = art_frame.shape[:2]
        
        # Define source corners (artwork corners)
        art_corners = [(0, 0), (art_width, 0), (art_width, art_height), (0, art_height)]
        
        # Process composition
        frame_count = 0
        
        while frame_count < len(corner_history):
            ret_bg, bg_frame = bg_cap.read()
            ret_art, art_frame = art_cap.read()
            
            if not ret_bg:
                break
                
            # Loop artwork if shorter than background
            if not ret_art:
                art_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_art, art_frame = art_cap.read()
                
            if not ret_art:
                break
            
            # Get tracking corners for this frame
            dst_corners = corner_history[frame_count]
            
            # Calculate homography
            homography = self.compositor.calculate_homography(art_corners, dst_corners)
            
            if homography is not None:
                # Warp artwork
                warped_art = self.compositor.warp_artwork(
                    art_frame, homography, (width, height)
                )
                
                # Create mask
                mask = self.compositor.create_mask(dst_corners, bg_frame.shape)
                
                # Color matching
                color_matched_art = self.compositor.color_match(
                    warped_art, bg_frame, mask
                )
                
                # Blend frames
                result = self.compositor.blend_frames(
                    bg_frame, color_matched_art, mask, blend_mode='normal'
                )
                
                # Add debug info
                cv2.putText(result, f"Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            else:
                result = bg_frame
                cv2.putText(result, "NO HOMOGRAPHY", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            comp_out.write(result)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Composed {frame_count} frames")
        
        bg_cap.release()
        art_cap.release()
        comp_out.release()
        
        print(f"Composition test completed. {frame_count} frames processed.")
        return True
    
    def run_full_test(self, background_video: str, artwork_video: str):
        """Run complete test pipeline"""
        print("=== RUNNING FULL PRISMA MOTION TEST ===")
        
        # Step 1: Test tracking
        if not self.test_tracking_only(background_video):
            print("Tracking test failed")
            return False
            
        # Step 2: Test composition
        tracking_data_path = f"{self.output_dir}/tracking_debug/tracking_data.json"
        if not self.test_composition(background_video, artwork_video, tracking_data_path):
            print("Composition test failed")
            return False
            
        print("=== ALL TESTS COMPLETED SUCCESSFULLY ===")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_runner.py <background_video> <artwork_video>")
        sys.exit(1)
        
    background_video = sys.argv[1]
    artwork_video = sys.argv[2]
    
    tester = PrismaMotionTest()
    tester.run_full_test(background_video, artwork_video)
