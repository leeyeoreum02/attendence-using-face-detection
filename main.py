import cv2
import threading
import sys
import os
import time
from PIL import Image
import numpy as np

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel

from src.models.python.models import Efficientnet

import torch
import torch.optim as optim
from torchvision import transforms


font = cv2.FONT_HERSHEY_TRIPLEX
running = False
data_count = 0
pTime = 0
cTime = 0

if os.path.exists('data') == False:
    os.makedirs('data', exist_ok=True)

def run():
    global frame
    global cropped
    global running
    global data_count
    global pTime, cTime, tm

    face_cascade = cv2.CascadeClassifier('src/models/xml/haarcascade_frontface.xml')

    # load model
    PATH = 'src/models/pretrained/face_detection.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Efficientnet(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 320))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret:
            height, width, _ = frame.shape
            
            top_left_x = width / 3
            top_left_y = (height / 2) + (height / 4)
            bottom_right_x = (width / 3) * 2
            bottom_right_y = (height / 2) - (height /4)

            cv2.rectangle(frame, (int(top_left_x) - 3, int(top_left_y) + 3), (int(bottom_right_x) + 3, int(bottom_right_y) - 3), 255, 2)
            cropped = frame[int(bottom_right_y):int(top_left_y), int(top_left_x):int(bottom_right_x)]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, "ROI", (int(top_left_x), int(top_left_y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cTime = time.time()
            fps = str(int(1 / (cTime - pTime)))
            pTime = cTime
            cv2.putText(img=frame, text=f'fps: {fps}', org=(0, 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                        color=(255, 255, 255), thickness=1)

            tm = time.localtime(cTime)
            timetxt = f'{tm.tm_year}/{tm.tm_mon}/{tm.tm_mday} {tm.tm_hour}:{tm.tm_min}:{tm.tm_sec}'
            cv2.putText(img=frame, text=timetxt, org=(0, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(0, 255, 0), thickness=1)

            faces = face_cascade.detectMultiScale(cropped, 1.1, 5)
            
            # predict
            if type(faces) is np.ndarray:
                face = frame[int(bottom_right_y):int(top_left_y), int(top_left_x):int(bottom_right_x)]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = np.float32(face)
                face = transforms.ToTensor()(face)
                face = transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )(face)
                model.eval()  
                face = face.to(device)
                target = model(face[None, ...])
                target = target > 0.5
                print(target)

            for (x, y, w, h) in faces:
                cv2.putText(frame, 'Detected Face', (int(top_left_x) + 9, int(bottom_right_y) + 12), font, 0.5, (255, 255, 0), 2)
                # cv2.imwrite(f'data/train_{data_count}.jpg', cropped, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                # print(f'capture complete: data\\train_{data_count}.jpg')
            
                # if data_count < len(os.listdir('data')):
                #     data_count += 1 
                
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


def capture():
    fr = cropped
    cv2.imwrite(f"captured_{tm.tm_year}_{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{tm.tm_sec}.jpg", fr, params=[cv2.IMWRITE_JPEG_QUALITY,100])
    print(fr.shape)
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
    btn_start = QtWidgets.QPushButton("Camera On") #TODO on off로 만들기
    btn_capture = QtWidgets.QPushButton("Capture")
    # TODO 이름 선택 할 수 있게
    vbox.addWidget(label)
    vbox.addWidget(btn_start)
    vbox.addWidget(btn_capture)
    win.setLayout(vbox)
    win.show()

    btn_start.clicked.connect(start)
    btn_capture.clicked.connect(capture)
    app.aboutToQuit.connect(onExit)

    sys.exit(app.exec_())