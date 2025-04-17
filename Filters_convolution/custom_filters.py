import numpy as np
import cv2
import sys
import time
import math


def create_gaussian_kernel(size, sigma):
    """
    Creates a 2D Gaussian blur kernel of given size and standard deviation.
    The kernel values are calculated using the Gaussian function:
        G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
    The loop shifts (i, j) from the kernel's centre to calculate x and y offsets.
    Each kernel entry is computed using this formula and normalised by dividing by the total sum.
    Used for smoothing the image while preserving edge transitions.
    """
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def apply_mean_filter(image, size):
    """
    Applies a mean (box) filter using a custom convolution.
    A kernel of ones is created and normalized so that all elements are equal:
        kernel = 1 / (size * size)
    The kernel is passed to custom_convolution, which averages neighboring pixels.
    This filter smooths the image but can blur edges.
    """
    kernel = np.ones((size, size)) / (size ** 2)
    return custom_convolution(image, kernel)


def apply_median_filter_slow(image, size):
    """
    Applies a median filter using nested loops (slow version for comparison).
    - Pads the image using cv2.copyMakeBorder with reflection mode.
    - Iterates over each pixel (i, j), extracts a size×size neighborhood,
      and replaces the pixel with the median of the neighborhood.
    This method is easy to understand but computationally expensive.
    """
    pad = size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded[i:i + size, j:j + size]
            result[i, j] = np.median(neighborhood)
    return result


def apply_median_filter(image, size):     # Faster version
    """
    Applies a fast median filter using NumPy's stride tricks for vectorisation.
    - First pads the image using OpenCV’s BORDER_REFLECT.
    - Constructs a 4D view of all size×size neighborhoods using np.lib.stride_tricks.as_strided.
    - Computes the median for each window efficiently across all pixels without loops.
    This version greatly improves performance.
    """
    pad = size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # Shape of the sliding window view: (number_of_rows, number_of_cols, size, size)
    out_shape = (image.shape[0], image.shape[1], size, size)

    # Compute the strides: each step in the sliding window along rows/columns.
    # padded.strides gives (stride_row, stride_col)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])

    # Create the sliding window view of the padded image.
    windows = np.lib.stride_tricks.as_strided(padded, shape=out_shape, strides=strides)

    # Compute the median for each window along the last two axes.
    median = np.median(windows, axis=(2, 3))

    # Return the result converted back to 8-bit unsigned integer.
    return median.astype(np.uint8)


def create_gabor_kernel(size, theta, sigma, lambd, gamma):
    """
     Generates a 2D Gabor filter kernel for texture and directional edge detection.
    - For each position (x, y), coordinates are rotated by theta.
    - The envelope is a Gaussian function controlled by sigma and gamma.
    - The carrier is a cosine wave based on wavelength (lambda) and x_theta.
    - Final value is the product of envelope and wave.
    The kernel is normalised by the sum of absolute values.
    Used to highlight edges in a specific orientation or frequency.
    """
    kernel = np.zeros((size, size))
    center = size // 2

    for x in range(size):
        for y in range(size):
            x_theta = (x - center) * np.cos(theta) + (y - center) * np.sin(theta)
            y_theta = -(x - center) * np.sin(theta) + (y - center) * np.cos(theta)

            envelope = np.exp(-(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2))
            wave = np.cos(2 * np.pi * x_theta / lambd)

            kernel[x, y] = envelope * wave

    return kernel / np.sum(np.abs(kernel))


def custom_convolution(image, kernel):
    """
     Applies a manually implemented 2D convolution between the image and kernel.
    - Pads the input image with BORDER_REFLECT to handle edges.
    - For each pixel, extracts a neighborhood of the same size as the kernel.
    - Performs element-wise multiplication and sums the result.
    The output is clipped to [0, 255] and cast to uint8 for valid image display.
    """
    ksize = kernel.shape[0]
    pad = ksize // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + ksize, j:j + ksize]
            result[i, j] = np.sum(region * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


def detect_edges(image, t_low, t_high=None):
    """
    Detects edges using simple gradient filters and optional hysteresis thresholding.
    - Applies custom_convolution using predefined horizontal (kernel_x) and vertical (kernel_y) Sobel-like kernels.
    - Computes gradient magnitude: sqrt(Gx^2 + Gy^2), then normalizes it.
    - If only t_low is provided: performs binary thresholding.
    - If t_low and t_high are provided:
        - Marks strong edges where magnitude >= t_high.
        - Marks weak edges where t_low <= magnitude < t_high.
        - Promotes weak edges to strong if they are 8-connected to a strong edge.
    Returns a binary edge map.
    """
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = kernel_x.T

    grad_x = custom_convolution(image, kernel_x)
    grad_y = custom_convolution(image, kernel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

    if t_high is None:
        return (magnitude > t_low).astype(np.uint8) * 255

    # Hysteresis thresholding
    strong = magnitude >= t_high
    weak = (magnitude >= t_low) & (magnitude < t_high)

    # Find connected weak pixels adjacent to strong ones
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if weak[i, j] and np.any(strong[i - 1:i + 2, j - 1:j + 2]):
                strong[i, j] = True

    return (strong * 255).astype(np.uint8)


def main():
    if len(sys.argv) < 4:
        print("Usage: python custom_filters.py image.jpg width filter_type [sigma] [t1 t2]")
        print("filter_type: gaussian, median, mean, gabor")
        print("Example: python custom_filters.py image.jpg 5 gaussian 1.0 50 100")
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
        sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        kernel = create_gaussian_kernel(width, sigma)
        smoothed = custom_convolution(img, kernel)
    elif filter_type == "median":
        smoothed = apply_median_filter(img, width)
    elif filter_type == "mean":
        smoothed = apply_mean_filter(img, width)
    elif filter_type == "gabor":
        theta = np.pi / 4  # 45 degrees
        sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
        lambd = 10.0
        gamma = 0.5
        kernel = create_gabor_kernel(width, theta, sigma, lambd, gamma)
        smoothed = custom_convolution(img, kernel)
    else:
        print("Invalid filter type")
        return

    filter_time = time.time() - start_time

    # Edge detection
    t_low = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    t_high = int(sys.argv[6]) if len(sys.argv) > 6 else None

    start_time = time.time()
    edges = detect_edges(smoothed, t_low, t_high)
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
