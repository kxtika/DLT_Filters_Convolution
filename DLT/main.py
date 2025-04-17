"""
Camera Calibration Using the DLT Method

This script implements the Direct Linear Transformation (DLT) method to compute the camera
projection matrix from 3D–2D correspondences obtained from an image of a checkerboard cube.
It then decomposes the projection matrix into the intrinsic matrix, rotation matrix, and
camera center, and assesses calibration quality by computing the reprojection error.
A visual overlay of the original (observed) 2D points and the reprojected points is also created.
"""

import numpy as np
import cv2


def dlt(x, X):
    """
    Compute the camera projection matrix P using the Direct Linear Transformation (DLT) method.

    Parameters:
        x (ndarray): An (N x 2) array of 2D image points (observed), where each row is [u, v].
        X (ndarray): An (N x 3) array of corresponding 3D control points, where each row is [X, Y, Z].

    Returns:
        P (ndarray): A (3 x 4) projection matrix that maps 3D points in homogeneous coordinates to 2D points.

    The following steps are performed:
    1. Convert 3D points to homogeneous coordinates by appending a 1 to each point.
    2. Build a linear system Ap = 0 from the correspondences (each correspondence contributes two rows).
    3. Solve the linear system using Singular Value Decomposition (SVD) and reshape the resulting
       12-element vector into a 3x4 projection matrix.
    """
    N = X.shape[0]
    # Convert 3D points to homogeneous coordinates
    if X.shape[1] == 3:
        X_h = np.hstack([X, np.ones((N, 1))])
    else:
        X_h = X.copy()

    A = []
    for i in range(N):
        X_i = X_h[i]  # (4,) homogeneous coordinate of 3D point
        u, v = x[i]
        # Each correspondence yields two equations:
        # Row 1:  -X_i   , 0s      , u * X_i
        # Row 2:  0s     , -X_i    , v * X_i
        row1 = np.hstack([-X_i, np.zeros(4), u * X_i])
        row2 = np.hstack([np.zeros(4), -X_i, v * X_i])
        A.append(row1)
        A.append(row2)
    A = np.array(A)

    # Solve using SVD. p is the singular vector corresponding to the smallest singular value.
    U, S, Vt = np.linalg.svd(A)
    p = Vt[-1]
    P = p.reshape(3, 4)
    return P


def rq_decomposition(A):
    """
    Compute the RQ decomposition of a 3x3 matrix.

    Parameters:
        A (ndarray): A (3 x 3) matrix, representing the first three columns of the projection matrix.

    Returns:
        R (ndarray): An upper-triangular matrix corresponding to the intrinsic parameters.
        Q (ndarray): An orthogonal matrix corresponding to the rotation.

    The decomposition is performed by reversing the rows of A, applying QR decomposition on the
    transposed matrix, and then flipping the factors appropriately.
    """
    # Reverse the order of rows
    A_flip = np.flipud(A)
    Q, R = np.linalg.qr(A_flip.T)
    R = np.flipud(R.T)
    Q = Q.T
    Q = np.fliplr(Q)
    return R, Q


def decompose_P(P):
    """
    Decompose the camera projection matrix P into intrinsic and extrinsic parameters.

    Parameters:
        P (ndarray): A (3 x 4) projection matrix.

    Returns:
        K (ndarray): A (3 x 3) intrinsic matrix.
        R (ndarray): A (3 x 3) rotation matrix representing the camera's orientation.
        X0 (ndarray): A (3,) vector representing the camera center in world coordinates.

    The steps are as follows:
    1. Extract M, the left 3x3 submatrix of P, and m4, the last column.
    2. Compute the camera center as X0 = -inv(M)*m4.
    3. Use RQ decomposition on M to obtain K and R.
    4. Normalize K so that its (3,3) entry is 1. A warning is printed if K[2,2] is zero.
    """
    M = P[:, :3]
    m4 = P[:, 3]
    # Compute camera center
    X0 = -np.linalg.inv(M) @ m4

    # Decompose M into K and R
    K, R = rq_decomposition(M)

    # Normalise K so that K[2,2] == 1
    if abs(K[2, 2]) > 1e-6:
        K = K / K[2, 2]
    else:
        print("Warning: K[2,2] is zero — can't normalize K.")
    return K, R, X0


def compute_reprojection_error(P, X, x_observed):
    """
    Compute the reprojection error by reprojecting the 3D points onto the image.

    Parameters:
        P (ndarray): A (3 x 4) projection matrix.
        X (ndarray): An (N x 3) array of 3D points.
        x_observed (ndarray): An (N x 2) array of observed 2D points.

    Returns:
        x_proj (ndarray): An (N x 2) array of reprojected 2D points.
        error (ndarray): A 1D array where each element is the Euclidean distance error for each point.
        mean_error (float): The mean of the reprojection errors.

    The 3D points are first converted to homogeneous coordinates, projected using P,
    and then converted back to 2D coordinates for comparison with the observed points.
    """
    N = X.shape[0]
    X_h = np.hstack([X, np.ones((N, 1))])
    x_proj_h = (P @ X_h.T).T  # resulting in homogeneous 2D coordinates
    x_proj = x_proj_h[:, :2] / x_proj_h[:, [2]]
    error = np.linalg.norm(x_observed - x_proj, axis=1)
    mean_error = np.mean(error)
    return x_proj, error, mean_error


def main():
    """
    Main function to execute the camera calibration pipeline.

    This function performs the following steps:
    1. Loads the 2D image points from file and defines the corresponding 3D control points.
    2. Computes the projection matrix P using the DLT method.
    3. Decomposes P to extract the intrinsic matrix K, rotation matrix R, and camera center X0.
    4. Rebuilds the projection matrix from the extracted calibration parameters.
    5. Reprojects the 3D points onto the image and computes the reprojection error.
    6. Visualizes the observed vs. reprojected points by overlaying them on the image.
    """
    # Define the 3D control points (based on the checkerboard cube dimensions)
    X = np.array([
        [0, 0, 0],
        [7, 0, 0],
        [7, 0, 7],
        [0, 0, 7],
        [7, 9, 7],
        [0, 9, 7],
        [0, 9, 0]
    ], dtype=np.float32)

    # Load the 2D image points from file (ensure the file 'image_points.npy' exists)
    x = np.load("image_points.npy")
    print("2D image points:\n", x)

    # Compute the projection matrix using DLT
    P = dlt(x, X)
    print("\nProjection Matrix P:\n", P)

    # Decompose the projection matrix to extract calibration parameters
    K, R, X0 = decompose_P(P)
    print("\nIntrinsic Matrix K:\n", K)
    print("\nRotation Matrix R:\n", R)
    print("\nCamera Center X0:\n", X0)

    # Rebuild the projection matrix from the calibration parameters:
    # Compute translation t = -R * X0, then reconstruct P' = K * [R | t]
    t = -R @ X0.reshape(-1, 1)
    P_rebuilt = K @ np.hstack((R, t))

    # Reproject 3D points using the rebuilt projection matrix
    N = X.shape[0]
    X_h = np.hstack([X, np.ones((N, 1))])
    x_hat_h = (P_rebuilt @ X_h.T).T
    x_hat = x_hat_h[:, :2] / x_hat_h[:, [2]]
    print("\nReprojected image coordinates (x_hat):\n", x_hat)

    # Compute reprojection error
    err_reproj, _, mean_error = compute_reprojection_error(P_rebuilt, X, x)
    print("\nThe reprojection error (per point):\n", err_reproj)
    print("\nMean reprojection error: {:.2f} pixels".format(mean_error))

    # Visualise observed and reprojected points on the image
    img = cv2.imread("checkerboard_cube/data/cube0.jpg")
    for pt in x:
        cv2.circle(img, tuple(np.round(pt).astype(int)), 5, (0, 0, 255), -1)  # Red circles (observed)
    for pt in x_hat:
        cv2.circle(img, tuple(np.round(pt).astype(int)), 5, (0, 255, 0), 2)  # Green circles (reprojected)

    cv2.imwrite("reprojection_result.jpg", img)
    cv2.imshow("Observed (Red) vs Reprojected (Green)", img)
    print("\nPress any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
