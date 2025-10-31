import cv2
import numpy as np
from scipy.optimize import minimize

def detect_lines(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    return [line[0] for line in lines]

def cost_function(params, lines):
    k1, k2 = params
    total_error = 0.0

    for (x1, y1, x2, y2) in lines:
        error = np.abs((y2 - y1) - (x2 - x1) * k1) + np.abs(k2 * (x1**2))
        total_error += error
        
    return total_error + (k1**2 + k2**2) * 0.1

if __name__ == "__main__":
    lines = detect_lines('your_photo.jpg')
    
    if lines:
        print(f"Detected {len(lines)} line segments.")

        x0 = [10.0, 10.0]

        print("Optimizing distortion parameters...")
        res = minimize(cost_function, x0, args=(lines,), method='Powell')
        
        if res.success:
            print(f"Optimization finished. Found params: {res.x}")
        else:
            print("Optimization failed to converge.")
            
    else:
        print("Error: Could not read 'your_photo.jpg' or no lines found.")
