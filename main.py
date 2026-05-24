import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp
import math
import tkinter as tk
from threading import Thread
from typing import List, Tuple
from dataclasses import dataclass

# ========== Configuration ==========
@dataclass
class Config:
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    # Detection settings
    MIN_DETECTION_CONFIDENCE = 0.4
    MIN_TRACKING_CONFIDENCE = 0.4
    
    # Gesture thresholds
    PINCH_THRESHOLD = 0.04
    SCROLL_THRESHOLD = 0.01
    
    # Smoothing
    SMOOTHING_FACTOR = 0.5
    
    # Colors
    CURSOR_MODE_COLOR = (0, 255, 0)
    SCROLL_MODE_COLOR = (255, 165, 0)
    KEYBOARD_MODE_COLOR = (0, 165, 255)
    
    # UI Settings
    KEYBOARD_SIZE = "800x300+100+500"
    KEYBOARD_BG = "black"
    KEY_BG = "gray20"
    KEY_FG = "white"
    KEY_FONT = ('Arial', 16)
    
class GestureModes:
    CURSOR = "Cursor"
    SCROLL = "Scroll"
    KEYBOARD = "Keyboard"

class HandController:
    def __init__(self):
        self.config = Config()
        self.mode = GestureModes.CURSOR
        self.screen_width, self.screen_height = pyautogui.size()
        self.prev_cursor_x = self.prev_cursor_y = None
        self.prev_scroll_y = None
        self.previous_tap = self.previous_pinch = self.previous_right_click = False
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize camera
        self.setup_camera()
        
        # Start virtual keyboard in separate thread
        self.start_virtual_keyboard()
    
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, self.config.CAMERA_WIDTH)
        self.cap.set(4, self.config.CAMERA_HEIGHT)
    
    def smooth_coordinates(self, new_x: float, new_y: float) -> Tuple[float, float]:
        if self.prev_cursor_x is None:
            self.prev_cursor_x, self.prev_cursor_y = new_x, new_y
            return new_x, new_y
        
        smoothed_x = self.prev_cursor_x + (new_x - self.prev_cursor_x) * self.config.SMOOTHING_FACTOR
        smoothed_y = self.prev_cursor_y + (new_y - self.prev_cursor_y) * self.config.SMOOTHING_FACTOR
        
        self.prev_cursor_x, self.prev_cursor_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y
    
    @staticmethod
    def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    @staticmethod
    def fingers_up(hand_landmarks) -> List[int]:
        finger_tips = [8, 12, 16, 20]
        finger_up = []
        
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                finger_up.append(1)
            else:
                finger_up.append(0)
        
        # Thumb check
        thumb = 1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0
        return [thumb] + finger_up
    
    def process_cursor_mode(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        index_dip = hand_landmarks.landmark[7]
        middle_tip = hand_landmarks.landmark[12]
        thumb_tip = hand_landmarks.landmark[4]
        
        # Smooth cursor movement
        screen_x, screen_y = self.smooth_coordinates(
            index_tip.x * self.screen_width,
            index_tip.y * self.screen_height
        )
        pyautogui.moveTo(int(screen_x), int(screen_y))
        
        # Process gestures
        dist_tip_dip = self.calculate_distance(index_tip.x, index_tip.y, index_dip.x, index_dip.y)
        dist_thumb_index = self.calculate_distance(thumb_tip.x, thumb_tip.y, index_tip.x, index_tip.y)
        dist_thumb_middle = self.calculate_distance(thumb_tip.x, thumb_tip.y, middle_tip.x, middle_tip.y)
        
        # Click gestures
        if dist_tip_dip < self.config.PINCH_THRESHOLD and not self.previous_tap:
            pyautogui.click()
            self.previous_tap = True
        elif dist_tip_dip >= self.config.PINCH_THRESHOLD:
            self.previous_tap = False
            
        if dist_thumb_index < self.config.PINCH_THRESHOLD and not self.previous_pinch:
            pyautogui.click()
            self.previous_pinch = True
        elif dist_thumb_index >= self.config.PINCH_THRESHOLD:
            self.previous_pinch = False
            
        if dist_thumb_middle < self.config.PINCH_THRESHOLD and not self.previous_right_click:
            pyautogui.rightClick()
            self.previous_right_click = True
        elif dist_thumb_middle >= self.config.PINCH_THRESHOLD:
            self.previous_right_click = False
    
    def process_scroll_mode(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        dist_index_middle = self.calculate_distance(
            index_tip.x, index_tip.y,
            middle_tip.x, middle_tip.y
        )
        
        if dist_index_middle < self.config.PINCH_THRESHOLD:
            current_scroll_y = (index_tip.y + middle_tip.y) / 2
            if self.prev_scroll_y is not None:
                delta_y = current_scroll_y - self.prev_scroll_y
                if abs(delta_y) > self.config.SCROLL_THRESHOLD:
                    pyautogui.scroll(-int(delta_y * 1000))
            self.prev_scroll_y = current_scroll_y
        else:
            self.prev_scroll_y = None
    
    def create_keyboard(self):
        root = tk.Tk()
        root.title("Virtual Keyboard")
        root.geometry(self.config.KEYBOARD_SIZE)
        root.configure(bg=self.config.KEYBOARD_BG)
        
        keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'Space']
        ]
        
        def press_key(key):
            if key == 'Space':
                pyautogui.press('space')
            else:
                pyautogui.write(key.lower())
        
        for i, row in enumerate(keys):
            for j, key in enumerate(row):
                width = 12 if key == 'Space' else 6
                btn = tk.Button(
                    root, text=key, width=width, height=2,
                    font=self.config.KEY_FONT,
                    bg=self.config.KEY_BG,
                    fg=self.config.KEY_FG,
                    command=lambda k=key: press_key(k)
                )
                btn.grid(row=i, column=j, padx=4, pady=4)
        
        root.attributes('-topmost', True)
        root.mainloop()
    
    def start_virtual_keyboard(self):
        keyboard_thread = Thread(target=self.create_keyboard)
        keyboard_thread.daemon = True
        keyboard_thread.start()
    
    def run(self):
        prev_time = 0
        
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    print("Failed to capture frame")
                    break
                
                img = cv2.flip(img, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(
                        img, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Detect fingers and set mode
                    finger_state = self.fingers_up(hand_landmarks)
                    total_fingers = sum(finger_state)
                    
                    # Mode switching logic
                    if total_fingers == 5:
                        self.mode = GestureModes.CURSOR
                    elif finger_state[1] == 1 and finger_state[2] == 1 and total_fingers == 2:
                        self.mode = GestureModes.SCROLL
                    elif finger_state[1] == 1 and finger_state[2] == 1 and finger_state[3] == 1 and total_fingers == 3:
                        self.mode = GestureModes.KEYBOARD
                    
                    # Process current mode
                    if self.mode == GestureModes.CURSOR:
                        self.process_cursor_mode(hand_landmarks)
                    elif self.mode == GestureModes.SCROLL:
                        self.process_scroll_mode(hand_landmarks)
                    
                    # Visual feedback
                    mode_color = {
                        GestureModes.CURSOR: self.config.CURSOR_MODE_COLOR,
                        GestureModes.SCROLL: self.config.SCROLL_MODE_COLOR,
                        GestureModes.KEYBOARD: self.config.KEYBOARD_MODE_COLOR
                    }[self.mode]
                    
                    cv2.putText(
                        img, f'MODE: {self.mode}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2
                    )
                
                # FPS calculation and display
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time
                
                cv2.putText(
                    img, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                
                cv2.imshow('Gesture Control', img)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Enable failsafe and add small pause between actions
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
    
    # Start the hand controller
    controller = HandController()
    controller.run()
