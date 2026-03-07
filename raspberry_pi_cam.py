import cv2
import numpy as np
import math

def find_horizon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=80, maxLineGap=20)
    if lines is not None:
        return max(lines, key=lambda l: np.linalg.norm([l[0][0]-l[0][2], l[0][1]-l[0][3]]))
    return None

def main():
    # Replace <PI_IP_ADDRESS> with your Pi's actual IP (e.g., 192.168.1.15)
    pi_url = "http://192.168.1.172:8080/?action=stream"
    
    cap = cv2.VideoCapture(pi_url)
    
    # Critical for Drones: Set buffer to 1 so we don't process "old" laggy frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not connect to Raspberry Pi stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret: continue

        coords = find_horizon(frame)
        if coords is not None:
            # Correctly unpacking the nested HoughLinesP result
            x1, y1, x2, y2 = coords[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate Tilt for stabilization
            tilt = math.degrees(math.atan2(y2 - y1, x2 - x1))
            cv2.putText(frame, f"PI-CAM TILT: {tilt:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Remote Raspberry Pi Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
