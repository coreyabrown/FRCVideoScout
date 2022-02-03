import argparse
from cv2 import contourArea
import imutils
import cv2

# Construct argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# Define color ranges of bumpers ((H-low, S-low, V-low), (H-high, S-high, V-high), "color")
# Can be found by running range-detection.py
colorRanges = [
    ((126, 121, 97), (179, 243, 255), "red"),
    ((44, 122, 0), (147, 255, 255), "blue")
]

# if no video path was supplied grab webcam, otherwise get video reference
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

# Loop through frames
while True:
    # Grab current frame
    (grabbed, frame) = camera.read()

    # Break at end of video
    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width = 600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Go over the color ranges
    for (lower, upper, colorName) in colorRanges:
        # Mask colors
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = imutils.grab_contours(cnts)

        # Only process frame if at least one contour was found
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.putText(frame, colorName, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'esc' key is pressed, stop the loop
    if key ==  27:
        break

# Close windows and clean camera
camera.release()
cv2.destroyAllWindows()