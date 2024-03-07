from cscore import CameraServer
import ntcore

import cv2
import json
import numpy as np
import time

# Set your team number
TEAM_NUMBER = 4829
# Specify the camera index (usually 0 for built-in webcam)
CAMERA_INDEX = 0
# Define lower and upper bounds for orange color in HSV
LOWER_ORANGE_HSV = np.array([0, 80, 80])
UPPER_ORANGE_HSV = np.array([6, 255, 255])
# The minimum contour area to detect a note
MINIMUM_CONTOUR_AREA = 400
# The threshold for a contour to be considered a disk
CONTOUR_DISK_THRESHOLD = 0.9


def find_largest_orange_contour(hsv_image: np.ndarray) -> np.ndarray:
    """
    Finds the largest orange contour in an HSV image
    :param hsv_image: the image to find the contour in
    :return: the largest orange contour
    """
    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv_image, LOWER_ORANGE_HSV, UPPER_ORANGE_HSV)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return max(contours, key=cv2.contourArea)


def contour_is_note(contour: np.ndarray) -> bool:
    """
    Checks if the contour is shaped like a note
    :param contour: the contour to check
    :return: True if the contour is shaped like a note
    """
    # Makes sure the contour isn't some random small spec of noise
    if cv2.contourArea(contour) < MINIMUM_CONTOUR_AREA:
        return False

    # Gets the smallest convex polygon that can fit around the contour
    contour_hull = cv2.convexHull(contour)
    # Fits an ellipse to the hull, and gets its area
    ellipse = cv2.fitEllipse(contour_hull)
    best_fit_ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
    # Returns True if the hull is almost as big as the ellipse
    return cv2.contourArea(contour_hull) / best_fit_ellipse_area > CONTOUR_DISK_THRESHOLD


def main():
   with open('/boot/frc.json') as f:
      config = json.load(f)
   camera = config['cameras'][0]

   width = 640
   height = 480

   nt = ntcore.NetworkTableInstance.getDefault()

   # Initialize NetworkTables
   nt.initialize(server='roborio-4829-frc.local')
   visionTable = nt.getTable('Note Detection OpenCV')

   CameraServer.startAutomaticCapture()

   input_stream = CameraServer.getVideo()
   output_stream = CameraServer.putVideo('Processed', width, height)

   # Allocating new images is very expensive, always try to preallocate
   img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

   # Wait for NetworkTables to start
   time.sleep(0.5)

   while True:
      start_time = time.time()

      frame_time, input_img = input_stream.grabFrame(img)
      output_img = np.copy(input_img)

      # Notify output of error and skip iteration
      if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue

      # Convert to HSV and threshold image
      hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
      binary_img = cv2.inRange(hsv_img, (0, 0, 100), (85, 255, 255))

      contour = find_largest_orange_contour(hsv_img)
      if contour is not None and contour_is_note(contour):
         cv2.ellipse(output_img, cv2.fitEllipse(contour), (255, 0, 255), 2)
         # print("Yo Note found w")

         # Extracting the center, width, and height of the ellipse
         ellipse = cv2.fitEllipse(contour)
         (x_center, y_center), (minor_axis, major_axis), angle = ellipse
         
         # Writing the extracted values to the NetworkTables
         visionTable.putNumber('EllipseCenterX', x_center)
         visionTable.putNumber('EllipseCenterY', y_center)
         visionTable.putNumber('EllipseWidth', minor_axis)
         visionTable.putNumber('EllipseHeight', major_axis)
         visionTable.putNumber('EllipseAngle', angle)
         

      # else:
      #     print('no note found ;c;c;c;cc;')

      output_stream.putFrame(output_img)

      # if cv2.waitKey(1) & 0xFF == ord("q"):
      #    break


if __name__ == "__main__":
    main()

