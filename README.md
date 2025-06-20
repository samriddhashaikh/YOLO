# YOLO

# Object Detection with YOLO 
YOLO (You Only Look Once) is a Deep Convolutional Neural Network that performs object detection from images.

It solves two problems - 
1. A Classification problem - to identify the type of object (e.g. car, person, auto, traffic lights, etc.) - Predicts class labels with a confidence score.
2. A Regression problem - it localizes the detected object with a bounding box in the image (e.g. finds the location of the object in the entire image). - Predicts the location of the bounding box - (x,y) coordinate of the centre of the bounding box along with the width and height of the bounding box.


# Create a Custom Dataset
1. Collect new images (car, auto, bus, traffic lights, person, etc.)
2. Annotate images using a tool like Label Studio.

# Model Training
3. Download a pre-trained YOLO model (e.g. Yolov11) from Ultralytics.
4. Train the Yolo model on Google Colab with GPU (Graphics Processing Unit, T4 GPU)
5. Save and download the newly trained YOLO model.

# Deployment
6. Download the model on Raspberry-Pi-5.
7. Convert the .pt model to NCNN format (optimized for the ARM CPUs like the one Raspberry-Pi has).
8. Run the optimized model on the Raspberry-Pi.


---
## 🗂️ Project Structure

```
YOLO/
├── requirements.txt            # Required Python libraries/packages to be installed.
├── camera_capture.py           # Script to capture image/video for model training. REquired during data collection.
├── YOLO_Train.ipynb            # IPython notebook to train the YOLO model. This will be run on a Google Colab environment having GPU accesss (e.g. T4 GPU).
├── yolo11n.pt                  # Trained YOLO model in pytorch format PyTorch format (.pt).
├── my_model.zip                # Trained model with all artifacts.
├── pt2ncnn.py                  # Convert a .pt YOLO model to the NCNN format. NCNN format is suitable for ARM processsors (as found in a Raspberry-Pi board).
├── yolo_detect.py              # Main script to run the YOLO object detection application.
├── ReadMe.md                   # Project Description.

```

---

## 🚀 Quickstart

### Run the following commands in the root project folder.

### 1. Create a Virtual Environment

```bash
python -m venv yolo
```

### 2. Activate the Virtual Environment

```bash
# On your Laptop (Windows)
yolo/Scripts/activate
```
```bash
# On your Raspberry-Pi (running Raspberry Pi OS)
source yolo/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run The Application

```bash
python yolo_detect.py
```

---

---
## 🖼 Data Creation/Annotation in Label Studio [Screenshot]
![Label Studio Interface](assets/Image%20Annotation_Label_Studio.png)
![Image Annotation - Example-1](assets/Image%20Annotation_Label_Studio_1.png)
![Image Annotation - Example-2](assets/Image%20Annotation_Label_Studio_2.png)
![Image Annotation - Example-3](assets/Image%20Annotation_Label_Studio_3.png)
![Image Annotation - Example-4](assets/Image%20Annotation_Label_Studio_4.png)
![Image Annotation - Example-5](assets/Image%20Annotation_Label_Studio_5.png)
![Image Annotation - Example-6](assets/Image%20Annotation_Label_Studio_6.png)

---
## Organizing Data 
## Unzip the YOLO_Data.zip file in `custom_data` directory.
![Orgamizing Data](assets/1.png)

## Organize data in `data` directory having two separate subfolders named `train` and `validation`.
![Orgamizing Data](assets/2.png)

## Create `data.yaml` file.
![Orgamizing Data](assets/3.png)

---
## Installations
![Install Ultralytics](assets/4.png)

---
## Model Architecture
![YOLO Model Architecture](assets/5.png)

---
## Model Training
![Training](assets/7.png)

---
## 📊 Sample Metrics
![Training Metrics](assets/8.png)

## Training/Validation Loss/Accuracy/mAP/Precision/Recall
![Training Metrics](assets/Training_Metrics_Screenshot.png)

---
## Save the Trained Model
![Best Performing Model](assets/10.png)

![Trained Model Files](assets/11.png)
---

## Test the Model
![Test the Model](assets/12.png)

## Model Inference Results
![Test Results  - Example-1](assets/13.png)
![Test Results - Example-2](assets/14.png)


---
## 📄 License
This project is open-sourced under the [MIT License](LICENSE).
---

## 🧠 Author
Ms. Samriddha Shaikh  
Student, Class-10, Amity International School, Sector-46, Gurgaon

---
## 📬 Contact
For questions or suggestions, reach out to me at samriddha.shaikh@gmail.com 
---

 
 
