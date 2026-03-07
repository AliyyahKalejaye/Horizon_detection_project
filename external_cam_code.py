import cv2
import numpy as np
import math

def find_horizon(frame):
    # Same logic as before: Canny Edge + Hough Transform
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=20)
    
    if lines is not None:
        return max(lines, key=lambda l: np.linalg.norm([l-l, l-l]))
    return None

def visualize_stability():
    # 0 indicates the default webcam (change to 1, 2, etc. if you have multiple cameras)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Run horizon detection on the live frame
        coords = find_horizon(frame)

        if coords is not None:
            # Reshape/Flatten to ensure 4 values
            coords = np.array(coords).flatten()
            
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
            
            # Draw the horizon line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate and display the roll angle (tilt)
            roll_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            cv2.putText(frame, f"Tilt: {roll_angle:.2f} deg", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Highlight center for pitch reference
            height, width = frame.shape[:2]
            cv2.circle(frame, (width // 2, height // 2), 5, (255, 0, 0), -1)

        # Display the resulting frame
        cv2.imshow('Camera Stability Visualization', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_stability()
