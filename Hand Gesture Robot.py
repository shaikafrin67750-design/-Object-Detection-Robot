# Hand Gesture Robot using Python

# Install required libraries:

# pip install opencv-python mediapipe

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
max_num_hands=1,
min_detection_confidence=0.7,
min_tracking_confidence=0.7
)

# Start webcam

camera = cv2.VideoCapture(0)

print("Hand Gesture Robot Started")
print("Press 'q' to exit.")

while True:
success, frame = camera.read()

```
if not success:
    print("Camera error!")
    break

# Flip image
frame = cv2.flip(frame, 1)

# Convert BGR to RGB
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Detect hand
result = hands.process(rgb_frame)

command = "STOP"

if result.multi_hand_landmarks:

    for hand_landmarks in result.multi_hand_landmarks:

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Get finger positions
        thumb = hand_landmarks.landmark[
            mp_hands.HandLandmark.THUMB_TIP
        ]

        index = hand_landmarks.landmark[
            mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]

        middle = hand_landmarks.landmark[
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]

        ring = hand_landmarks.landmark[
            mp_hands.HandLandmark.RING_FINGER_TIP
        ]

        pinky = hand_landmarks.landmark[
            mp_hands.HandLandmark.PINKY_TIP
        ]

        # Simple gesture detection
        if (
            index.y < hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_PIP
            ].y
            and middle.y < hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            ].y
            and ring.y < hand_landmarks.landmark[
                mp_hands.HandLandmark.RING_FINGER_PIP
            ].y
            and pinky.y < hand_landmarks.landmark[
                mp_hands.HandLandmark.PINKY_PIP
            ].y
        ):
            command = "FORWARD"

        elif (
            index.y < hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_PIP
            ].y
            and middle.y >= hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            ].y
        ):
            command = "LEFT"

        elif (
            index.y >= hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_PIP
            ].y
            and middle.y < hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            ].y
        ):
            command = "RIGHT"

        else:
            command = "STOP"

# Display robot command
cv2.putText(
    frame,
    "Robot: " + command,
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

print("Robot:", command)

# Show camera
cv2.imshow("Hand Gesture Robot", frame)

# Press q to exit
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

camera.release()
cv2.destroyAllWindows()

print("Hand Gesture Robot Closed.")
