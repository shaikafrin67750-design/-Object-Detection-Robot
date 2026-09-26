# Object Detection Robot using Python

import cv2

# Start camera

camera = cv2.VideoCapture(0)

if not camera.isOpened():
print("Camera could not be opened!")
exit()

# Load pre-trained object detection model

config = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
model = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(model, config)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# COCO class names

class_names = [
"background", "person", "bicycle", "car", "motorcycle",
"airplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign",
"parking meter", "bench", "bird", "cat", "dog",
"horse", "sheep", "cow", "elephant", "bear",
"zebra", "giraffe", "backpack", "umbrella", "handbag",
"tie", "suitcase", "frisbee", "skis", "snowboard",
"sports ball", "kite", "baseball bat", "baseball glove",
"skateboard", "surfboard", "tennis racket", "bottle",
"wine glass", "cup", "fork", "knife", "spoon", "bowl",
"banana", "apple", "sandwich", "orange", "broccoli",
"carrot", "hot dog", "pizza", "donut", "cake", "chair",
"couch", "potted plant", "bed", "dining table", "toilet",
"TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator",
"book", "clock", "vase", "scissors", "teddy bear",
"hair drier", "toothbrush"
]

print("Object Detection Robot Started")
print("Press 'q' to exit.")

while True:
ret, frame = camera.read()

```
if not ret:
    print("Camera error!")
    break

# Detect objects
class_ids, confidence, boxes = net.detect(
    frame,
    confThreshold=0.5
)

if len(class_ids) != 0:

    for class_id, conf, box in zip(
        class_ids.flatten(),
        confidence.flatten(),
        boxes
    ):

        label = class_names[class_id]

        # Draw detection box
        x, y, w, h = box

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label}: {conf * 100:.1f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Robot response
        if label == "person":
            print("Person detected -> Robot STOPPED")

        elif label == "car":
            print("Car detected -> Robot STOPPED")

        else:
            print(f"{label} detected -> Robot monitoring")

else:
    print("No object detected -> Robot MOVING")

# Show camera
cv2.imshow("Object Detection Robot", frame)

# Press q to exit
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

camera.release()
cv2.destroyAllWindows()

print("Object Detection Robot Closed.")
