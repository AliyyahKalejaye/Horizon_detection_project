import cv2
import numpy as np
import math

# --- GLOBAL STATE FOR SMOOTHING ---
# alpha: 0.2 (very smooth/slow) to 0.9 (fast/jittery). 0.2 is best for drones.
ALPHA = 0.7
smoothed_coords = None # Stores [x1, y1, x2, y2]

def find_single_horizon(frame):
    global smoothed_coords
    
    # 1. Image Processing (Same as before)
    smooth = cv2.bilateralFilter(frame, 9, 75, 75)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 2. Detection
    lines = cv2.HoughLinesP(closed_edges, 1, np.pi/180, threshold=50, 
                            minLineLength=100, maxLineGap=60)
    
    if lines is not None:
        # Pick the longest raw line
        new_coords = max(lines, key=lambda l: np.linalg.norm([l[0][0]-l[0][2], l[0][1]-l[0][3]]))[0]
        
        # 3. ANTI-JITTER LOGIC (Exponential Moving Average)
        if smoothed_coords is None:
            smoothed_coords = new_coords.astype(float)
        else:
            # Formula: Smooth = (Alpha * New) + ((1 - Alpha) * Old)
            smoothed_coords = (ALPHA * new_coords) + ((1 - ALPHA) * smoothed_coords)
            
        return smoothed_coords.astype(int)
    
    # If horizon is lost for one frame, return the last known smoothed position 
    # to prevent the line from disappearing instantly
    return smoothed_coords.astype(int) if smoothed_coords is not None else None

def main():
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pi_url = "http://192.168.1.172:8080/?action=stream"
    cap = cv2.VideoCapture(pi_url)
    
    if not cap.isOpened(): return

    while True:
        ret, frame = cap.read()
        if not ret: break

        coords = find_single_horizon(frame)

        if coords is not None:
            x1, y1, x2, y2 = coords
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Smoothed Green Line
            
            roll = math.degrees(math.atan2(y2 - y1, x2 - x1))
            cv2.putText(frame, f"STABLE ROLL: {roll:.2f} deg", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Jitter-Free Horizon', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
