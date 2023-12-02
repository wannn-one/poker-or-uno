import cv2

cap = cv2.VideoCapture(0)
ip = 'http://192.168.0.101:8080/video'
cap.open(ip)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.rectangle(img=frame, pt1=(56,45), pt2=(56+342, 45+630), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img=frame, pt1=(469,45), pt2=(469+342, 45+630), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img=frame, pt1=(882,45), pt2=(882+342, 45+630), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()