import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import socket

# Communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    detection_result = detector.detect(img_mp)
    # print(detection_result)
    # print(len(detection_result.multi_hand_landmarks))
    # print('Landmarks:', detection_result.handedness)
    # print('labels:', detection_result.hand_landmarks)

    h,w,c = img.shape

    data = []

    for one_hand in detection_result.hand_landmarks:
        for lm in one_hand:
            x, y, z = int(w*lm.x), int(h-h*lm.y), int(w*lm.z) #mediapipe和unity Y轴方向不一样
            data.extend([x, y, z])

        sock.sendto(str.encode(str(data)), serverAddressPort)

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in one_hand])

        mpDraw.draw_landmarks(img, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()