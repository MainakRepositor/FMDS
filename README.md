# YOLO NAS - StreamLit Object Detection Web App


## Description:

This web application uses the YOLOv8 object detection model to identify faces and classify them as "With Mask" or "No Mask" or "Improper Mask". It's built with Streamlit for an interactive and user-friendly experience. The app supports both images and videos in various formats.

### Key Features:

* Real-time face detection and mask classification
* User-friendly interface for easy uploads
* Efficient processing with YOLOv8
* Visual feedback with bounding boxes and labels

### Use Cases:

* Health and safety compliance monitoring
* Crowd management in public spaces
* Monitoring face mask usage in educational settings
* Enhanced security and surveillance


## Installation

Clone the repository.<br>
Install the required dependencies.<br>
Run the Streamlit app locally.<br>

### Follow below steps : <br>

* Create conda environment
    ```
    conda create -p venv python=3.10 -y

* Install pytorch and torchvision as per your system 
    ```
    https://pytorch.org/get-started/locally/
    ```

* Install required packages
    ```
    pip install -r requirements.txt
    ```

* Run StreamLit Webapp
    ```
    streamlit run app.py
    ```


