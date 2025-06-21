
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#--------------------------------------------------------
# Define Constants - Possibility of Future Customizations 
#--------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
UPDATE_INTERVAL = 15  # seconds
MIN_TIMER = 30
MAX_TIMER = 120
BASE_TIMER = 90
DELTA_SECONDS_PER_CAR = 5
LANE_DIVIDER_X = FRAME_WIDTH // 2

#------------------------
# Define Global Variables
#------------------------
frame_lock = threading.Lock()
current_frame = None
running = True

#-------------------------------------------
# GPIO Setup - GPIO Programming for the LEDs 
#-------------------------------------------
lane1_green = LED(17)  # BCM 17 = BOARD 11
lane1_red = LED(27)    # BCM 27 = BOARD 13
lane2_green = LED(22)  # BCM 22 = BOARD 15
lane2_red = LED(23)    # BCM 23 = BOARD 16

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
# Traffic Timer and GPIO Controller
# --------------------------------- 
class TrafficTimer:
    def __init__(self):
        self.timer1 = BASE_TIMER
        self.timer2 = BASE_TIMER
        self.last_update_time = time.time()
        self.last_counts = (0, 0)
        self.green_lane = 1
        self.green_end_time = time.time() + self.timer1
        self.set_lights(1)

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

            # Switch light
            self.green_lane = 1 if count1 >= count2 else 2
            self.green_end_time = now + (self.timer1 if self.green_lane == 1 else self.timer2)
            self.set_lights(self.green_lane)

    def set_lights(self, lane):
        if lane == 1:
            lane1_green.on()
            lane1_red.off()
            lane2_green.off()
            lane2_red.on()
        else:
            lane2_green.on()
            lane2_red.off()
            lane1_green.off()
            lane1_red.on()

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

    capture_thread = threading.Thread(target=capture_frames, args=(cap,))
    capture_thread.start()

    try:
        # Continuously perform object detection on the video stream
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

            # Draw Visuals - Annorate the frame with proper information
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

            # Show frame
            cv2.imshow("Traffic Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resource and cleanup
        running = False
        capture_thread.join()
        cap.release()
        cv2.destroyAllWindows()

        # Turn off all LEDs
        lane1_red.off()
        lane1_green.off()
        lane2_red.off()
        lane2_green.off()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------
# Execute Main Function
# -----------------------
if __name__ == "__main__":
    main()
