import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_horizon_steps(image_path):
    # Load and prepare image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # STEP 1: Gaussian Blur (Noise Reduction)
    # The (5,5) kernel blurs the image to prevent false edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # STEP 2: Canny Edge Detection
    # Identifies sharp intensity changes
    edges = cv2.Canny(blurred, 50, 250)

    # STEP 3: Final Detection & Points Visualization
    # Using HoughLinesP to extract mathematical line coordinates
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=img.shape[1]//4, maxLineGap=20)
    
    final_output = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw the horizon line in green
            cv2.line(final_output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Draw the calculation points (endpoints) in red
            cv2.circle(final_output, (x1, y1), 10, (0, 0, 255), -1)
            cv2.circle(final_output, (x2, y2), 10, (0, 0, 255), -1)

    # --- Displaying the results side-by-side ---
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(blurred, cmap='gray')
    plt.title('1. Gaussian Blur Output')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('2. Canny Edge Map')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # Convert BGR to RGB for correct display in Matplotlib
    plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
    plt.title('3. Final Detection (Red Points = Calc Basis)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Replace with your filename
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'ocean_view.jpg')
visualize_horizon_steps(image_path)
