from cscore import CameraServer
import ntcore

import cv2
import json
import numpy as np
import time


# Define lower and upper bounds for orange color in HSV
# LOWER_ORANGE_HSV = np.array([4, 85, 162])
# UPPER_ORANGE_HSV = np.array([17, 255, 255])
LOWER_ORANGE_HSV = np.array([13, 41, 58])
UPPER_ORANGE_HSV = np.array([25, 255, 255])
# The minimum contour area to detect a note
MINIMUM_CONTOUR_AREA = 150
MAXIMUM_CONTOUR_AREA_FOR_CAMERA_UPPER_HALF = 1500
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


def contour_is_note(contour: np.ndarray) -> list:
    """
    Checks if the contour is shaped like a note
    :param contour: the contour to check
    :return: True if the contour is shaped like a note
    """

    # Makes sure the contour isn't some random small spec of noise
    if cv2.contourArea(contour) < MINIMUM_CONTOUR_AREA:
        return [False, -2]

    # Gets the smallest convex polygon that can fit around the contour
    contour_hull = cv2.convexHull(contour)
    # Fits an ellipse to the hull, and gets its area
    ellipse = cv2.fitEllipse(contour_hull)
    best_fit_ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
    # Returns True if the hull is almost as big as the ellipse
    print(cv2.contourArea(contour_hull))
    return [cv2.contourArea(contour_hull) / best_fit_ellipse_area > CONTOUR_DISK_THRESHOLD, cv2.contourArea(contour_hull)]


def main():
   #define camera settings
   with open('/boot/frc.json') as f:
      config = json.load(f)
   camera = config['cameras'][0]

   width = 320
   height = 240

   #instantiate network tables
   nt = ntcore.NetworkTableInstance.getDefault()
   nt.startClient4("coprocessor")
   nt.setServerTeam(4829)


#    # Initialize NetworkTables
   visionTable = nt.getTable('SmartDashboard')

#start capture
   CameraServer.startAutomaticCapture()

   input_stream = CameraServer.getVideo()
   #mirror input stream to output with ellipse
   output_stream = CameraServer.putVideo('Processed', width, height)

   # Allocating new images is very expensive, always try to preallocate
   img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

   # Wait for NetworkTables to start
   time.sleep(1)


   while True:
      start_time = time.time()
      #capture frame
      frame_time, input_img = input_stream.grabFrame(img)
      output_img = np.copy(input_img)

      # Notify output of error and skip iteration
      if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue

      # Convert to HSV and threshold image
      hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

      contour = find_largest_orange_contour(hsv_img)

      if contour is not None and contour_is_note(contour)[0]:
         cv2.ellipse(output_img, cv2.fitEllipse(contour), (255, 0, 255), 2)

         # Extracting the center, width, and height of the ellipse
         ellipse = cv2.fitEllipse(contour)
         (x_center, y_center), (minor_axis, major_axis), angle = ellipse
        #  print(x_center, ", ", y_center, ", ")
         
         # Writing the extracted values to the NetworkTables
         if (contour_is_note(contour)[1] > MAXIMUM_CONTOUR_AREA_FOR_CAMERA_UPPER_HALF and y_center < 75):
            visionTable.putNumber('EllipseCenterX', -2.0)
            visionTable.putNumber('EllipseCenterY', -2.0)    
         else:
            visionTable.putNumber('EllipseCenterX', x_center)
            visionTable.putNumber('EllipseCenterY', y_center)
        

        
      #view result on wpilibpi.local/1182
      output_stream.putFrame(output_img)


if __name__ == "__main__":
    main()

