import cv2
import threading
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui

font = cv2.FONT_HERSHEY_TRIPLEX
running = False

def run():
    global frame
    global running
    face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
    info = ''
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, 'Detected Face', (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)
            h,w,c = frame.shape
            qImg = QtGui.QImage(frame.data, w, h, w*c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            label.setPixmap(pixmap)
        else:
            QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
            print("cannot read frame.")
            break
    cap.release()
    print("Thread end.")


def stop():
    global running
    running = False
    print("stoped..")


global i
i = 0
def capture():
    global i
    fr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite("file_%d.jpg" %i, fr, params=[cv2.IMWRITE_JPEG_QUALITY,100])
    i += 1
    print("caprure..")


def start():
    global running
    running = True
    th = threading.Thread(target=run)
    th.start()
    print("started..")


def onExit():
    print("exit")
    stop()


app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
vbox = QtWidgets.QVBoxLayout()
label = QtWidgets.QLabel()
btn_start = QtWidgets.QPushButton("Camera On")
btn_capture = QtWidgets.QPushButton("Capture")
vbox.addWidget(label)
vbox.addWidget(btn_start)
vbox.addWidget(btn_capture)
win.setLayout(vbox)
win.show()

btn_start.clicked.connect(start)
btn_capture.clicked.connect(capture)
app.aboutToQuit.connect(onExit)

sys.exit(app.exec_())