import numpy as np
import cv2
import time

font = cv2.FONT_HERSHEY_TRIPLEX
def faceDetect():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
    info = ''

    try:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
    except:
        print('카메라 로딩 실패')
        return
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Detected Face', (x-5, y-5), font, 0.5, (255, 255, 0), 2)

        cv2.imshow('attendence-using-face-detection', frame)

        if cv2.waitKey(1) != -1:  # 아무 키나 누르면
            cv2.imwrite("file_%d.jpg" %i, frame, params=[cv2.IMWRITE_JPEG_QUALITY,100])  # 프레임을 'photo.jpg'에 저장
            i += 1
        k = cv2.waitKey(30)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

faceDetect()