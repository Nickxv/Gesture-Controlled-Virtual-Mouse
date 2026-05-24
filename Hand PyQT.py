import sys
import cv2
import time
import math
import pyautogui
import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QComboBox
)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

pyautogui.FAILSAFE = False


class HandMouseApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Gesture Mouse Controller")
        self.setGeometry(100,100,1100,700)

        self.video_label = QLabel()
        self.video_label.setFixedSize(720,520)

        # Start/Stop Button (smaller width)
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.setFixedSize(150,40)

        self.camera_btn.setStyleSheet("""
        QPushButton{
            background-color:#2ecc71;
            color:white;
            font-size:14px;
            font-weight:bold;
            border-radius:6px;
        }
        QPushButton:hover{
            background-color:#27ae60;
        }
        """)

        self.camera_running=False

        self.status=QLabel("Camera Stopped")

        self.speed_slider=QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)

        gestures=[
            "Pinch",
            "Thumb + Pinky",
            "Open Hand",
            "Closed Fist",
            "Thumbs Up",
            "Four Fingers"
        ]

        self.left_click_combo=QComboBox()
        self.right_click_combo=QComboBox()
        self.scroll_combo=QComboBox()
        self.pause_combo=QComboBox()
        self.screenshot_combo=QComboBox()

        for g in gestures:
            self.left_click_combo.addItem(g)
            self.right_click_combo.addItem(g)
            self.scroll_combo.addItem(g)
            self.pause_combo.addItem(g)
            self.screenshot_combo.addItem(g)

        self.left_click_combo.setCurrentText("Pinch")
        self.right_click_combo.setCurrentText("Thumb + Pinky")
        self.scroll_combo.setCurrentText("Open Hand")
        self.pause_combo.setCurrentText("Closed Fist")
        self.screenshot_combo.setCurrentText("Thumbs Up")

        # Camera Layout
        camera_layout=QVBoxLayout()
        camera_layout.addWidget(self.video_label)
        camera_layout.addWidget(self.camera_btn, alignment=Qt.AlignLeft)
        camera_layout.addWidget(QLabel("Cursor Speed"))
        camera_layout.addWidget(self.speed_slider)
        camera_layout.addWidget(self.status)

        # Settings Layout
        settings_layout=QVBoxLayout()

        settings_layout.addWidget(QLabel("Left Click Gesture"))
        settings_layout.addWidget(self.left_click_combo)

        settings_layout.addWidget(QLabel("Right Click Gesture"))
        settings_layout.addWidget(self.right_click_combo)

        settings_layout.addWidget(QLabel("Scroll Gesture"))
        settings_layout.addWidget(self.scroll_combo)

        settings_layout.addWidget(QLabel("Pause Gesture"))
        settings_layout.addWidget(self.pause_combo)

        settings_layout.addWidget(QLabel("Screenshot Gesture"))
        settings_layout.addWidget(self.screenshot_combo)

        settings_layout.addStretch()

        main_layout=QHBoxLayout()
        main_layout.addLayout(camera_layout,3)
        main_layout.addLayout(settings_layout,1)

        self.setLayout(main_layout)

        self.camera_btn.clicked.connect(self.toggle_camera)

        self.timer=QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap=None

        self.mp_hands=mp.solutions.hands
        self.hands=self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.mp_draw=mp.solutions.drawing_utils

        self.screen_w,self.screen_h=pyautogui.size()

        self.last_click=0
        self.last_screenshot=0

        self.click_delay=0.6
        self.screenshot_delay=2

        self.prev_scroll_y=None


    def toggle_camera(self):

        if not self.camera_running:

            self.cap=cv2.VideoCapture(0)
            self.timer.start(20)

            self.camera_btn.setText("Stop Camera")

            self.camera_btn.setStyleSheet("""
            QPushButton{
                background-color:#e74c3c;
                color:white;
                font-size:14px;
                font-weight:bold;
                border-radius:6px;
            }
            """)

            self.status.setText("Camera Running")
            self.camera_running=True

        else:

            self.timer.stop()

            if self.cap:
                self.cap.release()

            self.video_label.clear()

            self.camera_btn.setText("Start Camera")

            self.camera_btn.setStyleSheet("""
            QPushButton{
                background-color:#2ecc71;
                color:white;
                font-size:14px;
                font-weight:bold;
                border-radius:6px;
            }
            """)

            self.status.setText("Camera Stopped")
            self.camera_running=False


    def take_screenshot(self):

        screenshot=pyautogui.screenshot()

        filename=f"screenshot_{int(time.time())}.png"

        screenshot.save(filename)

        self.status.setText(f"Screenshot saved: {filename}")


    def distance(self,p1,p2):

        return math.hypot(p1.x-p2.x,p1.y-p2.y)


    def count_fingers(self,hand):

        tips=[4,8,12,16,20]
        fingers=[]
        landmarks=hand.landmark

        if landmarks[tips[0]].x<landmarks[tips[0]-1].x:
            fingers.append(1)
        else:
            fingers.append(0)

        for i in range(1,5):

            if landmarks[tips[i]].y<landmarks[tips[i]-2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers.count(1)


    def thumbs_up(self,hand):

        lm=hand.landmark

        thumb_up = lm[4].y < lm[3].y < lm[2].y
        other_down = lm[8].y > lm[6].y and lm[12].y > lm[10].y and lm[16].y > lm[14].y and lm[20].y > lm[18].y

        return thumb_up and other_down


    def update_frame(self):

        ret,frame=self.cap.read()

        if not ret:
            return

        frame=cv2.flip(frame,1)

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(frame,hand_landmarks,self.mp_hands.HAND_CONNECTIONS)

                landmarks=hand_landmarks.landmark

                index_tip=landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip=landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip=landmarks[self.mp_hands.HandLandmark.PINKY_TIP]

                x=int(index_tip.x*self.screen_w)
                y=int(index_tip.y*self.screen_h)

                speed=self.speed_slider.value()

                pyautogui.moveTo(x,y,duration=0.01/speed)

                current_time=time.time()

                fingers=self.count_fingers(hand_landmarks)

                # Screenshot
                if self.screenshot_combo.currentText()=="Thumbs Up":

                    if self.thumbs_up(hand_landmarks):

                        if current_time-self.last_screenshot>self.screenshot_delay:

                            self.take_screenshot()
                            self.last_screenshot=current_time

                if self.screenshot_combo.currentText()=="Four Fingers":

                    if fingers==4:

                        if current_time-self.last_screenshot>self.screenshot_delay:

                            self.take_screenshot()
                            self.last_screenshot=current_time


                # Left Click
                if self.left_click_combo.currentText()=="Pinch":

                    if self.distance(index_tip,thumb_tip)<0.05:

                        if current_time-self.last_click>self.click_delay:

                            pyautogui.click()
                            self.last_click=current_time


                # Right Click
                if self.right_click_combo.currentText()=="Thumb + Pinky":

                    if self.distance(thumb_tip,pinky_tip)<0.06:

                        if current_time-self.last_click>self.click_delay:

                            pyautogui.rightClick()
                            self.last_click=current_time


                # Scroll
                if self.scroll_combo.currentText()=="Open Hand":

                    if fingers==5:

                        current_y=index_tip.y

                        if self.prev_scroll_y is not None:

                            dy=current_y-self.prev_scroll_y

                            if abs(dy)>0.002:

                                scroll_amount=int(dy*-3000)

                                pyautogui.scroll(scroll_amount)

                        self.prev_scroll_y=current_y

                    else:

                        self.prev_scroll_y=None


                if self.pause_combo.currentText()=="Closed Fist":

                    if fingers==0:

                        self.status.setText("Paused")


        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,ch=img.shape

        qimg=QImage(img.data,w,h,ch*w,QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(qimg))


if __name__=="__main__":

    app=QApplication(sys.argv)

    window=HandMouseApp()
    window.show()

    sys.exit(app.exec_())
