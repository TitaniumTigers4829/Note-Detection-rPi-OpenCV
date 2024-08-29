# Note-Object-Detection-OpenCV-Rasp-Pi-Code
Note Detection

Uses our custom made [note detection](https://github.com/TitaniumTigers4829/camera-note-detection) code, with enhancements to interact with WPILib/networktables with a raspberry pi.

Please see the example on WPILib to [set up](https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/using-the-raspberry-pi-for-frc.html) the raspberry pi.
Also, here is an example to [set up](https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/basic-vision-example.html) a USB camera for the raspberry pi.

For the USB Camera, ensure you adjust the resolution in main.py, and you can play around with the saturation, brightness, and contrast for the camera settings using the WPILib imager (wpilibpi.local once plug in pi)

Lower bounds for the HSV value can also be adjusted for easier detection. Be careful though, as too low values could result in high detection of other objects as well. Lighting conditions may vary from testing, comp, and practice environments.

Good luck!
