import cv2
import numpy as np
import math

def find_single_horizon(frame):
    if frame is None:
        return None

    # STEP 1: Bilateral Filter (Removes texture noise like waves/grass while keeping horizon sharp)
    # Params: d=9, sigmaColor=75, sigmaSpace=75
    smooth = cv2.bilateralFilter(frame, 9, 75, 75)
    
    # STEP 2: Grayscale and Canny Edge Detection
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # STEP 3: Morphological Closing (Connects multiple line segments into one)
    # A 5x5 kernel bridges larger gaps than a 3x3
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # STEP 4: HoughLinesP with high 'maxLineGap' to force line merging
    # minLineLength=100 ensures we ignore small distractions (clouds/ships)
    lines = cv2.HoughLinesP(closed_edges, 1, np.pi/180, 
                            threshold=50, 
                            minLineLength=100, 
                            maxLineGap=60) 
    
    if lines is not None:
        # STEP 5: Mathematical Filter - Pick the single longest line in the image
        best_line = max(lines, key=lambda l: np.linalg.norm([l[0][0]-l[0][2], l[0][1]-l[0][3]]))
        return best_line[0] # Returns [x1, y1, x2, y2]
    
    return None

def main():
    # Use index 0 for built-in, 1 for external USB, or your Pi-Cam URL
    # cap = cv2.VideoCapturehtt(0, cv2.CAP_DSHOW) 
    pi_url = "http://192.168.1.172:8080/?action=stream"
    cap = cv2.VideoCapture(pi_url)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        coords = find_single_horizon(frame)

        if coords is not None:
            x1, y1, x2, y2 = coords
            
            # Draw the SINGLE horizon line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Calculate and display telemetry
            roll = math.degrees(math.atan2(y2 - y1, x2 - x1))
            cv2.putText(frame, f"SINGLE HORIZON - ROLL: {roll:.2f} deg", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "HORIZON LOST", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show detection result
        cv2.imshow('Clean Horizon Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
