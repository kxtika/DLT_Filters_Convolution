import cv2
import sys
import time
import numpy as np


def main():
    if len(sys.argv) < 4:
        print("Usage: python opencv_filters.py image.jpg width filter_type [sigma] [t1 t2]")
        print("filter_type: gaussian, median, mean, gabor")
        print("Example: python opencv_filters.py image.jpg 5 gaussian 1.0 50 100")
        return
    
    filename = sys.argv[1]
    width = int(sys.argv[2])
    filter_type = sys.argv[3].lower()
    
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image")
        return

    # Downscale the image by 0.5 (50% of its original width and height).
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

    # Apply smoothing filter
    start_time = time.time()
    
    if filter_type == "gaussian":
        sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 0
        smoothed = cv2.GaussianBlur(img, (width, width), sigma)
    elif filter_type == "median":
        smoothed = cv2.medianBlur(img, width)
    elif filter_type == "mean":
        smoothed = cv2.blur(img, (width, width))
    elif filter_type == "gabor":
        sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
        lambd = 10.0
        gamma = 0.5
        theta = np.pi/4
        kernel = cv2.getGaborKernel((width, width), sigma, theta, lambd, gamma)
        smoothed = cv2.filter2D(img, -1, kernel)
    else:
        print("Invalid filter type")
        return
    
    filter_time = time.time() - start_time
    
    # Edge detection
    t_low = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    t_high = int(sys.argv[6]) if len(sys.argv) > 6 else None
    
    start_time = time.time()
    if t_high is None:
        edges = cv2.Canny(smoothed, t_low, t_low*3)
    else:
        edges = cv2.Canny(smoothed, t_low, t_high)
    edge_time = time.time() - start_time
    
    # Display results
    cv2.imshow("Original", img)
    cv2.imshow("Smoothed", smoothed)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Filter time: {filter_time:.4f}s")
    print(f"Edge detection time: {edge_time:.4f}s")


if __name__ == "__main__":
    main()