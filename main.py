import numpy as np
import cv2

font = cv2.FONT_HERSHEY_TRIPLEX
def faceDetect():
    eye_detect = False
    face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    info = ''

    try:
        cap = cv2.VideoCapture(0)
    except:
        print('카메라 로딩 실패')
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if eye_detect:
            info = 'Eye Detection On'
        else:
            info = 'Eye Detection Off'

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Detected Face', (x-5, y-5), font, 0.5, (255, 255, 0), 2)
            if eye_detect:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(30)
        if k == ord('i'):
            eye_detect = not eye_detect

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

faceDetect()



# import cv2
#
# # Load the cascade
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # Read the input image
# img = cv2.imread('test7.jpg')
# # Convert into grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Detect faces
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# # Draw rectangle around the faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# # Display the output
# cv2.imshow('img', img)
# cv2.waitKey()