import cv2
import numpy as np
import os
import pandas as pd

def extract_features(img_path, label):
    img = cv2.imread(img_path)
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Feature 1: Edge Density (Noise check)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges / 255) / (height * width)
    
    # Feature 2: Line "Straightness" (Hough confidence)
    # We re-run a quick Hough to get the 'score' of the best line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=10)
    
    if lines is not None:
        best_line = lines[0][0]
        length = np.sqrt((best_line[2]-best_line[0])**2 + (best_line[3]-best_line[1])**2)
        relative_length = length / width
    else:
        relative_length = 0
        
    return {
        'edge_density': edge_density,
        'relative_length': relative_length,
        'label': label  # 1 for correct, 0 for failed
    }

# Process folders into a CSV
dataset = []
for label_dist in [('1_correct', 1), ('0_failed', 0)]:
    folder = os.path.join('./horizon_videos/horizon_videos_3/labeled_training_data', label_dist[0])
    for filename in os.listdir(folder):
        feat = extract_features(os.path.join(folder, filename), label_dist[1])
        if feat: dataset.append(feat)

df = pd.DataFrame(dataset)
df.to_csv('horizon_training_data.csv', index=False)
print("Features extracted to CSV.")
