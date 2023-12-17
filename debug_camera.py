import cv2
from utils.constants import *

cap = cv2.VideoCapture(0)
ip = 'http://192.168.223.121:8080/video'
cap.open(ip)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.rectangle(img=frame, pt1=(x_player, y), pt2=(x_player + distance_x, y + distance_y), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img=frame, text="Player", org=(x_player, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img=frame, pt1=(x_opponent, y), pt2=(x_opponent + distance_x, y + distance_y), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img=frame, text="Opponent", org=(x_opponent, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()