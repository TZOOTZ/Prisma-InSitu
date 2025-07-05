# src/motion_tracker.py
import cv2
import numpy as np
import json
from typing import List, Tuple, Optional


class MarqueeTracker:
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.corner_trackers = []
        self.tracking_history = []
        self.initial_corners = None
        
    def setup_manual_tracking(self, first_frame: np.ndarray) -> List[Tuple[int, int]]:
        """Setup manual corner selection for marquee"""
        print("Click 4 corners of the marquee in clockwise order")
        print("Press 'r' to reset, 'c' to confirm")
        
        self.corners = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
                self.corners.append((x, y))
                print(f"Corner {len(self.corners)}: ({x}, {y})")
        
        # Clone frame for marking
        frame_copy = first_frame.copy()
        cv2.namedWindow('Marquee Selection', cv2.WINDOW_RESIZABLE)
        cv2.setMouseCallback('Marquee Selection', mouse_callback)
        
        while True:
            display_frame = frame_copy.copy()
            
            # Draw selected corners
            for i, corner in enumerate(self.corners):
                cv2.circle(display_frame, corner, 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f'{i+1}', 
                           (corner[0]+10, corner[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw lines between corners
            if len(self.corners) > 1:
                for i in range(len(self.corners)):
                    cv2.line(display_frame, self.corners[i], 
                            self.corners[(i+1) % len(self.corners)], 
                            (255, 0, 0), 2)
            
            cv2.imshow('Marquee Selection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Reset
                self.corners = []
                print("Reset corners")
            elif key == ord('c') and len(self.corners) == 4:  # Confirm
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
                
        cv2.destroyAllWindows()
        
        self.initial_corners = self.corners
        return self.corners
    
    def initialize_trackers(self, frame: np.ndarray, corners: List[Tuple[int, int]]):
        """Initialize individual corner trackers"""
        self.corner_trackers = []
        
        for corner in corners:
            # Create bounding box around corner
            x, y = corner
            bbox = (x-15, y-15, 30, 30)
            
            # Initialize tracker
            tracker = cv2.TrackerCSRT_create()
            success = tracker.init(frame, bbox)
            
            if success:
                self.corner_trackers.append(tracker)
            else:
                print(f"Failed to initialize tracker for corner {corner}")
                
        print(f"Initialized {len(self.corner_trackers)} corner trackers")
    
    def track_frame(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Track corners in current frame"""
        if not self.corner_trackers:
            return None
            
        current_corners = []
        valid_trackers = []
        
        for i, tracker in enumerate(self.corner_trackers):
            success, bbox = tracker.update(frame)
            
            if success:
                # Calculate center of bounding box
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                current_corners.append((int(center_x), int(center_y)))
                valid_trackers.append(tracker)
                
                if self.debug_mode:
                    # Draw tracking box
                    cv2.rectangle(frame, 
                                (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                                (255, 0, 0), 2)
                    cv2.putText(frame, f'C{i}', 
                               (int(center_x), int(center_y)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print(f"Lost tracking for corner {i}")
                
        self.corner_trackers = valid_trackers
        
        if len(current_corners) == 4:
            self.tracking_history.append(current_corners)
            return current_corners
        else:
            return None
    
    def smooth_tracking(self, history_window=5) -> List[List[Tuple[int, int]]]:
        """Apply smoothing to tracking history"""
        if len(self.tracking_history) < history_window:
            return self.tracking_history
            
        smoothed_history = []
        
        for i in range(len(self.tracking_history)):
            start_idx = max(0, i - history_window // 2)
            end_idx = min(len(self.tracking_history), i + history_window // 2 + 1)
            
            # Average corners in window
            smoothed_corners = []
            for corner_idx in range(4):
                avg_x = np.mean([self.tracking_history[j][corner_idx][0] 
                               for j in range(start_idx, end_idx)])
                avg_y = np.mean([self.tracking_history[j][corner_idx][1] 
                               for j in range(start_idx, end_idx)])
                smoothed_corners.append((int(avg_x), int(avg_y)))
                
            smoothed_history.append(smoothed_corners)
            
        return smoothed_history
    
    def save_tracking_data(self, filepath: str):
        """Save tracking data to JSON"""
        data = {
            'initial_corners': self.initial_corners,
            'tracking_history': self.tracking_history,
            'smoothed_history': self.smooth_tracking()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Tracking data saved to {filepath}")
