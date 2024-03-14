import cv2
import os
from datetime import datetime

# Create a folder for snapshots if it doesn't exist
snapshots_folder = "snapshots"
if not os.path.exists(snapshots_folder):
    os.makedirs(snapshots_folder)

# Initialize the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the captured image
    cv2.imshow('Camera Feed', frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    if key == ord('s'):  # Save on 's' key
        # Create a unique filename for each snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(snapshots_folder, f"snapshot_{timestamp}.jpg")
        
        # Save the captured image to the snapshots folder
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved as {filename}")
    
    elif key == ord('q'):  # Quit on 'q' key
        break

# When everything done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()












# import cv2
# import numpy as np
# import glob
# import os
# from datetime import datetime
# from tqdm import tqdm

# # Checkerboard dimensions (inner corners, not squares)
# CHECKERBOARD = (7, 10)
# subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
# calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# # Arrays to store object points and image points from all images.
# objpoints = []  # 3d point in real world space
# imgpoints = []  # 2d points in image plane.

# # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,9,0)
# objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * 2.5  # Adjust this if your square size is different

# # Path to the images
# images = glob.glob('snapshots/*.jpg')

# # Process each image with tqdm for a progress bar
# for fname in tqdm(images, desc="Processing Images"):
#     img = cv2.imread(fname)
#     if img is None:
#         continue
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
#     # If found, add object points, image points
#     if ret:
#         objpoints.append(objp)
#         cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
#         imgpoints.append(corners)

# # Calibration
# N_OK = len(objpoints)
# K = np.zeros((3, 3))
# D = np.zeros((4, 1))
# rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
#     objpoints,
#     imgpoints,
#     gray.shape[::-1],
#     K,
#     D,
#     rvecs,
#     tvecs,
#     calibration_flags,
#     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
# )

# print("Found " + str(N_OK) + " valid images for calibration")
# print("Camera matrix:\n", K)
# print("Distortion coefficients:\n", D)
