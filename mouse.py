import cv2
import mediapipe as mp
import numpy as np
import pyautogui as p

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
p.FAILSAFE = False


def distance(d1, d2):
    d = (d2[0]-d1[0])**2 + (d2[1]-d1[1])**2 + (d2[2]-d1[2])**2
    d = d**(1/2)
    return d

def angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
hand_landmarks = None
# speed = 50

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.2) as hands:
    while True:
        success, image = cap.read()
        image = image[150:420, 100:440]
#         cv2.imshow("initial image", image)
#         image = cv2.resize(image, (320, 240))
        if not success:
              print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
              continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
              break

        if hand_landmarks:
            (x,y,z) = (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z)
#             d2 = (hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[8].z)
#             x = (d1[0]+d2[0])/2
#             y = (d1[1]+d2[1])/2

            x, y = round(x,6), round(y,6)

            x = tuple(np.multiply(x, [1920]).astype(int))[0]
            y = tuple(np.multiply(y, [1080]).astype(int))[0]
            if x>1920:
                x = 1920
            if y>1080:
                y = 1080
#             print(x, y)

            p.moveTo(x,y)

            d1 = (hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z)
            d2 = (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z)
            d3 = (hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y, hand_landmarks.landmark[3].z)
            d = angle(d1, d3, d2)
#             d = round(distance(d1, d2), 3)
            print(d)
            dist = distance(d1, d2)
            print(dist)
            if d < 41 and dist<0.05:
                print("clicking!")
                p.click(x,y)
#             print(d)

        hand_landmarks = None

