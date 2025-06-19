from ultralytics import YOLO

# Load a YOLO11n PyTorch model
# model = YOLO("yolov8n.pt")
model = YOLO("yolo11n.pt")


# Export the model to NCNN format
# model.export(format="ncnn", imgsz=640)  # creates 'yolov8n_ncnn_model'
model.export(format="ncnn", imgsz=640)  # creates 'yolo11n_ncnn_model'
