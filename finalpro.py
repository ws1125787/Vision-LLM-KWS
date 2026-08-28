
import numpy as np
from numpy.linalg import inv
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import skimage.data

def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# Morphological operation용 타원형 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 객체 최초 검출 여부 확인용 boolean
found = False

# 객체 좌표 및 반지름 변수 정의
x_bel, y_bel = 0, 0
radius = 0

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    start = time.time()
    ret, frame = cap.read()

    if not ret:
            break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame = cv2.flip(frame, 0)

    # 가우시안 블러
    blr = cv2.GaussianBlur(frame, (11, 11), 0)

    # LAB 색공간 변환
    lab = cv2.cvtColor(blr, cv2.COLOR_BGR2LAB)

    # LAB color segmentation
    mask = cv2.inRange(lab, green_lower, green_upper)

    # Opening 2회, Dilation 2회
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Contour detection
    contour_lst, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contour_lst) > 0:
        # 가장 큰 contour 선택
        contour = max(contour_lst, key=cv2.contourArea)
        # 최소 외접원 반지름
        _, radius = cv2.minEnclosingCircle(contour)
        # 무게중심
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (0, 0)

        # 검출된 객체에 파란 원 overlay
        cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

        # 객체 최초 검출
        if not found:
            found = True

        x, y, w, h = cv2.boundingRect(contour)
        

        def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
            _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(img, coord1, coord2, color, thickness)
            cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+13), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    


    # TODO: cv2.rectangle() 함수에 사용할 수 있는 좌표로 변환
        x1, y1 = (x , y)
        x2, y2 = (x + w, y + h)

        draw_bbox(center, (x1, y1), (x2, y2), (0, 255, 0), txt="Apple, 0.98")


cap.release()
cv2.destroyAllWindows()
