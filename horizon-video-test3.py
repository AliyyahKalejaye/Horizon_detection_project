import cv2
import numpy as np
import os
import csv

def process_video_to_dataset(video_path, output_folder, sample_rate=10):
    # Setup folders
    os.makedirs(os.path.join(output_folder, 'canny'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'final'), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    csv_path = os.path.join(output_folder, 'horizon_labels.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'x1', 'y1', 'x2', 'y2'])
        
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            # Only process every 'sample_rate' frame to ensure data diversity
            if frame_count % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                
                # Detect lines
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                        minLineLength=frame.shape[1]//4, maxLineGap=20)
                
                if lines is not None:
                    # Logic to find the most "horizontal" line
                    best_line = None
                    max_len = 0
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if length > max_len:
                            max_len = length
                            best_line = (x1, y1, x2, y2)
                    
                    if best_line:
                        x1, y1, x2, y2 = best_line
                        img_name = f"frame_{saved_count:04d}.jpg"
                        
                        # 1. Save Label
                        writer.writerow([img_name, x1, y1, x2, y2])
                        
                        # 2. Save Canny Edge (Training Feature)
                        cv2.imwrite(os.path.join(output_folder, 'canny', img_name), edges)
                        
                        # 3. Save Final Visualization (Verification)
                        vis = frame.copy()
                        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(vis, (x1, y1), 5, (0, 0, 255), -1)
                        cv2.circle(vis, (x2, y2), 5, (0, 0, 255), -1)
                        cv2.imwrite(os.path.join(output_folder, 'final', img_name), vis)
                        
                        saved_count += 1
            
            frame_count += 1
            
    cap.release()
    print(f"Processed {frame_count} frames. Saved {saved_count} training samples.")

# Usage: process 'drone_flight.mp4', sampling every 15th frame
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'Horizon_video_3.mp4')
process_video_to_dataset(video_path, script_dir, sample_rate=15)
