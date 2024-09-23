# Import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2

# Update with the correct paths to your model files
prototxt = "C:/Users/maddo/OneDrive/Desktop/PROJECTS/Real Time Object Analysis Using Python & ML/deploy.prototxt"
model = "C:/Users/maddo/OneDrive/Desktop/PROJECTS/Real Time Object Analysis Using Python & ML/model.caffemodel"

confidence_threshold = 0.2

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "pen"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Initialize the video stream, allow the camera sensor to warm up,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Allow camera to warm up
fps = FPS().start()

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (1028, 1028)),
                                  0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.stop()
