import cv2
import threading
import sys
import os

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel


font = cv2.FONT_HERSHEY_TRIPLEX
running = False
data_count = 0

def run():
    global frame
    global cropped
    global running
    global data_count
    face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
    info = ''
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret:
            height, width = frame.shape[:2]
            
            top_left_x = width / 3
            top_left_y = (height / 2) + (height / 4)
            bottom_right_x = (width / 3) * 2
            bottom_right_y = (height / 2) - (height /4)

            cv2.rectangle(frame, (int(top_left_x) - 3, int(top_left_y) + 3), (int(bottom_right_x) + 3, int(bottom_right_y) - 3), 255, 3)
            
            cropped = frame[int(bottom_right_y):int(top_left_y), int(top_left_x):int(bottom_right_x)]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
        
            faces = face_cascade.detectMultiScale(cropped, 1.1, 5)
            print('faces:', faces)

            for (x, y, w, h) in faces:
                cv2.putText(frame, 'Detected Face', (int(top_left_x) + 9, int(bottom_right_y) + 12), font, 0.5, (255, 255, 0), 2)
                cv2.imwrite(f'data/train_{data_count}.jpg', cropped, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                print(f'capture complete: data\\train_{data_count}.jpg')
            
                if data_count < len(os.listdir('data')):
                    data_count += 1 
                
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
    fr = cropped
    cv2.imwrite("file_%d.jpg" %i, fr, params=[cv2.IMWRITE_JPEG_QUALITY,100])
    i += 1
    print("capture..")


def start():
    global running
    running = True
    th = threading.Thread(target=run)
    th.start()
    print("started..")


def onExit():
    print("exit")
    stop()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout()
    label: QLabel = QtWidgets.QLabel()
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