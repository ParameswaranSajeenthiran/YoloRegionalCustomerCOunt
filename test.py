import cv2
import cvzone
import math
import time
import numpy as np
from ultralytics import YOLO
from sort import *

# tracker =

# Initialize video capture
cap = cv2.VideoCapture("test2.mp4")  # For Video

# Load the YOLO model
model = YOLO("yolov8l.pt")

# List to store all polygons
polygons = []
# Temporary list to store points for the current polygon
current_polygon = []

# Define class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

prev_frame_time = 0
new_frame_time = 0


# Mouse callback function to get polygon points
def mouse_callback(event, x, y, flags, param):
    global current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the point to the current polygon
        current_polygon.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(current_polygon) > 2:
        # Complete the polygon on right-click
        polygons.append({"tracker":Sort(max_age=20, min_hits=3, iou_threshold=0.3),"current_person_detections": np.empty((0, 5)), "cummulative_detections": [], "points": current_polygon.copy()})
        current_polygon = []
        print(f"Polygon completed. Total polygons: {len(polygons)}")


# Set up the mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Capture the first frame
success, img = cap.read()
if success:
    height, width, _ = img.shape

    # Resize the frame
    scale_percent = 100
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    dim = (new_width, new_height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Display the first frame and allow the user to draw polygons
    while True:
        # Draw all existing polygons
        for polygon in polygons:
            cv2.polylines(img, [np.array(polygon['points'])], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the current polygon as it's being created
        if len(current_polygon) > 1:
            cv2.polylines(img, [np.array(current_polygon)], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        # Press 's' to start the video after drawing polygons
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Process the video
while True:
    new_frame_time = time.time()

    success, img = cap.read()
    if not success:
        break

    # Resize the frame
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Detect objects
    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = classNames[cls]
            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            if label == "person":
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    for polygon in polygons:
        cv2.polylines(img, [np.array(polygon['points'])], isClosed=True, color=(0, 255, 0), thickness=2)
        temp_detections = np.empty((0, 5))

        for person in detections:
            center = ((person[0] + person[2]) / 2, person[1] + person[3] / 2)
            # detect wheather the cordinates of the center of the person is inside the polygon
            if cv2.pointPolygonTest(np.array(polygon['points']), center, False) == 1:
                temp_detections = np.vstack((temp_detections, person))



        trackResults =polygon['tracker'].update(temp_detections)
        for results_ in trackResults:
            x1, y1, x2, y2, id_ = results_
            if id_ not in polygon['cummulative_detections']:
                polygon['cummulative_detections'].append(id_)

        if polygon['points']:
            cv2.putText(img, f'People: {len(polygon["cummulative_detections"])}', polygon['points'][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
