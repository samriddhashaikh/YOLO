"""
# Project - Adaptive Traffic Control with AI on Edge Device (Raspberry-Pi)
 ## Major Functional Blocks
  > 1. Object Detection with AI - YOLO Convolutional Neural Network (CNN) Model
  > 2. Program Adaptive Traffic Signaling Logic
  > 3. GPIO Programming for Controlling Traffic Lights (LEDs) and Timers for the 2x16 LCD screen
"""
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-----------------
# Import Libraries
#-----------------
import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from gpiozero import LED
from RPLCD.i2c import CharLCD

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#--------------------------------------------------------
# Define Constants - Possibility of Future Customizations 
#--------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
UPDATE_INTERVAL = 15  # seconds
MIN_TIMER = 30
MAX_TIMER = 90
BASE_TIMER = 60
DELTA_SECONDS_PER_CAR = 5
LANE_DIVIDER_X = FRAME_WIDTH // 2

#------------------------
# Define Global Variables
#------------------------
frame_lock = threading.Lock()
current_frame = None
running = True

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#---------------------------------------------
# Class: LEDController
# Purpose: Control Red, Yellow, and Green LEDs 
#---------------------------------------------
class LEDController:
    def __init__(self):
        # Lane 1 LEDs
        self.lane1_green = LED(17)   # BOARD 11
        self.lane1_red = LED(27)     # BOARD 13
        self.lane1_yellow = LED(24)  # BOARD 18
        # Lane 2 LEDs
        self.lane2_green = LED(22)   # BOARD 15
        self.lane2_red = LED(23)     # BOARD 16
        self.lane2_yellow = LED(25)  # BOARD 22

    def update(self, active_lane, remaining_time):
        # Turn on green/yellow/red appropriately
        if active_lane == 1:
            self._set_lane(self.lane1_green, self.lane1_red, self.lane1_yellow, remaining_time)
            self._set_lane(self.lane2_green, self.lane2_red, self.lane2_yellow, 0)
        else:
            self._set_lane(self.lane2_green, self.lane2_red, self.lane2_yellow, remaining_time)
            self._set_lane(self.lane1_green, self.lane1_red, self.lane1_yellow, 0)

    def _set_lane(self, green, red, yellow, remaining):
        if remaining > 5:
            green.on()
            red.off()
            yellow.off()
        elif 0 < remaining <= 5:
            green.off()
            red.off()
            yellow.on()
        else:
            green.off()
            red.on()
            yellow.off()

    def all_off(self):
        for led in [self.lane1_green, self.lane1_red, self.lane1_yellow,
                    self.lane2_green, self.lane2_red, self.lane2_yellow]:
            led.off()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-------------------------------------------------
# Class: I2C2x16LCD
# Purpose: Display Timer Info on 2x16 I2C LCD Screen
#-------------------------------------------------
class I2C2x16LCD:
    def __init__(self):
        # Initialize I2C LCD display
        # I2C-1 bus on Raspberry Pi: SDA = GPIO2 (Pin 3), SCL = GPIO3 (Pin 5)
        # Initialize LCD (using PCF8574 I2C backpack, common address is 0x27)
        self.lcd = CharLCD('PCF8574', address=0x27, port=1, cols=16, rows=2)
        
    def display_timers(self, t1, t2):
        self.lcd.clear()
        self.lcd.cursor_pos = (0,0)
        self.lcd.write_string(f"Lane-1: {t1:>3}s")
        self.lcd.cursor_pos = (1,0)
        self.lcd.write_string(f"Lane-2: {t2:>3}s")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#---------------------------------------
# Object Detection Class with YOLO model
#---------------------------------------
class YOLOModel:
    def __init__(self, model_path="yolo11n_ncnn_model", input_size=640, target_class="car"):
        self.model = YOLO(model_path, task='detect')
        self.input_size = input_size
        self.labels = self.model.names
        self.target_class = target_class
        self.class_index = [i for i, name in self.labels.items() if name.lower() == target_class][0]

    def detect_lane_objects(self, frame, x_start, x_end):
        results = self.model(frame, verbose=False)[0]
        count = 0
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = box.conf.item()
            if cls != self.class_index or conf < 0.5:
                continue
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            xc = (xmin + xmax) / 2
            if x_start <= xc <= x_end:
                count += 1
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        return count

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------------------
# Traffic Timer and LED Coordination
# --------------------------------- 
class TrafficTimer:
    def __init__(self):
        self.timer1 = BASE_TIMER
        self.timer2 = BASE_TIMER
        self.last_update_time = time.time()
        self.last_counts = (0, 0)
        self.green_lane = 1
        self.green_end_time = time.time() + self.timer1

    def update(self, count1, count2):
        now = time.time()
        count_changed = (count1, count2) != self.last_counts
        interval_passed = (now - self.last_update_time) > UPDATE_INTERVAL

        if count_changed or interval_passed:
            delta = count1 - count2
            t1 = BASE_TIMER - DELTA_SECONDS_PER_CAR * delta if delta < 0 else BASE_TIMER
            t2 = BASE_TIMER + DELTA_SECONDS_PER_CAR * delta if delta > 0 else BASE_TIMER
            self.timer1 = min(max(t1, MIN_TIMER), MAX_TIMER)
            self.timer2 = min(max(t2, MIN_TIMER), MAX_TIMER)
            self.last_update_time = now
            self.last_counts = (count1, count2)

            # Switch to the more congested lane
            self.green_lane = 1 if count1 >= count2 else 2
            self.green_end_time = now + (self.timer1 if self.green_lane == 1 else self.timer2)

    def get_timers_and_active_lane(self):
        now = time.time()
        remaining = int(self.green_end_time - now)
        return int(self.timer1), int(self.timer2), self.green_lane, max(remaining, 0)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------
# Camera Capture Thread
# ---------------------
def capture_frames(cap):
    global current_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            current_frame = frame.copy()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------------------------------------------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# --------------------------------------------
# Custom Function - Main Execution Block Logic 
# --------------------------------------------
def main():
    global running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    detector = YOLOModel()
    traffic_control = TrafficTimer()
    led_control = LEDController()
    lcd_display = I2C2x16LCD()

    capture_thread = threading.Thread(target=capture_frames, args=(cap,))
    capture_thread.start()

    try:
        while True:
            with frame_lock:
                if current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = current_frame.copy()

            count1 = detector.detect_lane_objects(frame, 0, LANE_DIVIDER_X - 1)
            count2 = detector.detect_lane_objects(frame, LANE_DIVIDER_X, FRAME_WIDTH)

            traffic_control.update(count1, count2)
            t1, t2, green_lane, remaining = traffic_control.get_timers_and_active_lane()

            led_control.update(green_lane, remaining)
            lcd_display.display_timers(t1, t2)

            # Draw Visuals
            cv2.line(frame, (LANE_DIVIDER_X, 0), (LANE_DIVIDER_X, FRAME_HEIGHT), (0, 255, 255), 2)
            cv2.putText(frame, f"Lane-1 Cars: {count1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Lane-2 Cars: {count2}", (LANE_DIVIDER_X + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Timer-1: {t1}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f"Timer-2: {t2}s", (LANE_DIVIDER_X + 10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f"Green Lane: {green_lane} | Ends in: {remaining}s", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

            cv2.imshow("Traffic Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resource and cleanup
        running = False
        capture_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        led_control.all_off()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------
# Execute Main Function
# -----------------------
if __name__ == "__main__":
    main()
