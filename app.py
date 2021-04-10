import cv2
from flask import Flask, render_template, Response


# font = cv2.FONT_HERSHEY_TRIPLEX
# face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
# info = ''
cap = cv2.VideoCapture(0)


def detect():
    # while True:
    #     ret, frame = cap.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    #     cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)
    #
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #         cv2.putText(frame, 'Detected Face', (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     cv2.imshow('face-detection', frame)
    #
    #     k = cv2.waitKey(1) & 0xff
    #     if k == 27:
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
    # print("end.")
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__)


@app.route('/face')
def face():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run('0.0.0.0', port=8000, debug=True)