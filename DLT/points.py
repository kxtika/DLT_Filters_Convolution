"""
Interactive 2D Point Collector for Camera Calibration

This script loads an image of a checkerboard cube and allows the user to manually click
on the 2D locations of known 3D control points. These clicked points are saved for use in
camera calibration using the DLT method.
"""

import cv2
import numpy as np
import os

# Global dictionary to store the collected 2D points and image copy for drawing.
params = {'points': [], 'img': None}


def click_event(event, x, y, param):
    """
    Mouse callback function to record 2D points from mouse clicks on the image.

    When the left mouse button is clicked, the (x, y) coordinates are appended
    to param['points'] and a small circle is drawn for feedback.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at: (x={}, y={})".format(x, y))
        param['points'].append([x, y])
        # Draw a red circle (radius 5) at the clicked point
        cv2.circle(param['img'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Cube Image", param['img'])


def get_2d_points(image_path):
    """
    Loads an image, displays it, and uses a mouse callback to collect 2D points.

    Args:
      image_path (str): Path to the image file.

    Returns:
      pts (np.ndarray): Collected 2D points as a float32 NumPy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found at", image_path)
        return None

    # Make a copy of the image for drawing.
    params['img'] = img.copy()

    # Display the image and attach the mouse callback.
    cv2.imshow("Cube Image", params['img'])
    cv2.setMouseCallback("Cube Image", click_event, params)

    print("\nInstructions:")
    print("  1. Click on the image to select the corresponding 2D image points.")
    print("  2. The order of clicks should match the order of the defined 3D control points:")
    print("       [ [0,0,0], [7,0,0], [7,0,7], [0,0,7], [7,9,7], [0,9,7], [0,9,0] ]")
    print("  3. Press any key once you have finished clicking.\n")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pts = np.array(params['points'], dtype=np.float32)
    return pts


def points():
    # Define the path to the image (adjust the folder path as needed).
    image_path = os.path.join("checkerboard_cube", "cube0.jpg")

    # Define the 3D control points on the cube.
    # The origin is assumed to be at the nearest corner as in cube_origin.png
    # and the points are given in the same order as the user clicks.
    X = np.array([
        [0, 0, 0],
        [7, 0, 0],
        [7, 0, 7],
        [0, 0, 7],
        [7, 9, 7],
        [0, 9, 7],
        [0, 9, 0]
    ], dtype=np.float32)

    print("3D Control Points (X):")
    print(X)

    # Load the image and get 2D correspondences by clicking.
    points_file = "image_points.npy"

    # Check if saved 2D points exist; if yes, load them.
    if os.path.exists(points_file):
        print("Loading previously saved 2D image points from {}...".format(points_file))
        x = np.load(points_file)
        print("Loaded 2D points:\n", x)
    else:
        # Otherwise, collect them interactively.
        x = get_2d_points(image_path)
        if x is None:
            return
        # Save the 2D points to file for future runs.
        np.save(points_file, x)
        print("2D points saved to", points_file)

    print("\nFinal 2D Image Points (x):")
    print(x)
    print(
        "\nMake sure that the ordering of these 2D points corresponds exactly to the ordering of the 3D control points in X.")


if __name__ == '__main__':
    points()
