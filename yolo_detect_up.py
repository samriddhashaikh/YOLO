import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# --- Global Frame and Lock ---
frame_lock = threading.Lock()
current_frame = None
running = True

# --- Colors for Bounding Boxes ---
bbox_colors = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
    (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
]

# --- Frame Capture Thread ---
def capture_frames(cap):
    global current_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            current_frame = frame.copy()

# --- YOLOModel Inference Class ---
class YOLOModel:
    def __init__(self, model_dir, input_size=640):
        self.model = YOLO(model_dir, task='detect')
        self.input_size = input_size
        self.labels = self.model.names

    def detect_objects(self, frame, min_thresh=0.5):
        results = self.model(frame, verbose=False)
        detections_raw = results[0].boxes
        object_count_lane1 = 0
        object_count_lane2 = 0

        for i in range(len(detections_raw)):
            xyxy = detections_raw[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            classidx = int(detections_raw[i].cls.item())
            classname = self.labels[classidx]
            conf = detections_raw[i].conf.item()

            if conf > min_thresh:
                x_center = (xmin + xmax) // 2
                if x_center < 320:
                    object_count_lane1 += 1
                else:
                    object_count_lane2 += 1

                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{classname}: {int(conf * 100)}%'
                label_size, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                              (xmin + label_size[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame, object_count_lane1, object_count_lane2

# --- Main Execution ---
def main():
    global running

    detector = YOLOModel(model_dir="yolo11n_ncnn_model", input_size=640)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot open USB camera.")
        return

    capture_thread = threading.Thread(target=capture_frames, args=(cap,))
    capture_thread.start()

    frame_rate_buffer = []
    fps_avg_len = 100

    # --- Initialize Counters and Timers ---
    last_lane1_count = last_lane2_count = -1
    timer1 = timer2 = 90
    last_update_time = time.time()

    try:
        while True:
            t_start = time.perf_counter()

            with frame_lock:
                if current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = current_frame.copy()

            # Detect objects and count in lanes
            frame, lane1_count, lane2_count = detector.detect_objects(frame)

            # Check if re-evaluation is needed
            current_time = time.time()
            counts_changed = (lane1_count != last_lane1_count or lane2_count != last_lane2_count)
            if counts_changed or (current_time - last_update_time) >= 15:
                diff = lane1_count - lane2_count
                shift = min(abs(diff) * 5, 30)  # limit shift to avoid extreme changes
                if diff > 0:
                    # Lane 1 has more cars
                    timer1 = max(30, 90 - shift)
                    timer2 = min(150, 90 + shift)
                elif diff < 0:
                    timer2 = max(30, 90 - shift)
                    timer1 = min(150, 90 + shift)
                else:
                    timer1 = timer2 = 90

                last_update_time = current_time
                last_lane1_count = lane1_count
                last_lane2_count = lane2_count

            # FPS calculation
            t_end = time.perf_counter()
            fps = 1.0 / (t_end - t_start)
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(fps)
            avg_fps = np.mean(frame_rate_buffer)

            # Annotate frame
            cv2.line(frame, (320, 0), (320, 480), (255, 255, 255), 2)
            cv2.putText(frame, f"Lane1 Cars: {lane1_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Lane2 Cars: {lane2_count}", (340, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, f"Timer-1: {timer1}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Timer-2: {timer2}s", (340, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Display
            cv2.imshow("YOLOv8n - Traffic Lanes", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        running = False
        capture_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Average pipeline FPS: {np.mean(frame_rate_buffer):.2f}")

if __name__ == "__main__":
    main()
