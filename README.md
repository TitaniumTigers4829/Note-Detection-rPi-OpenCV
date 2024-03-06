# Note-Object-Detection-OpenCV-Rasp-Pi-Code
Note Detection

Please see the example on WPILib for setting up the rPi: https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/using-the-raspberry-pi-for-frc.html
Also, here is an example for setting up a USB camera for the rPi: https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/basic-vision-example.html

For the USB Camera, ensure you adjust the resolution in main.py, and you can play around with the saturation, brightness, and contrast for the camera settings using the WPILib imager (wpilibpi.local once plug in pi)

Lower bounds for the HSV value can also be adjusted for easier detection. Be careful though, as too low values could result in high detection of other objects as well. Lighting conditions may vary from testing, comp, and practice environments.

Good luck!
