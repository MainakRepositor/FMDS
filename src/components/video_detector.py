import cv2
from ultralytics import YOLO
import time

def load_yolonas_process_each_frame(video_name, kpi_text  ,kpi2_text,kpi3_text, stframe, conf):
    cap = cv2.VideoCapture(video_name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO('models/best_3.pt')

    previous_time = 0
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame , conf=conf)
            

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame , channels='BGR', use_column_width = True)
    
            current_time = time.time()
            
            fps = 1/(current_time - previous_time)
            previous_time = current_time
            kpi_text.write(f'<h1 style= "text-align:center"; color:red> {"{:.1f}".format(fps)}</h1>', unsafe_allow_html=True)
            kpi2_text.write(f'<h1 style= "text-align:center"; color:red> {"{:.1f}".format(width)}</h1>', unsafe_allow_html=True)
            kpi3_text.write(f'<h1 style= "text-align:center"; color:red> {"{:.1f}".format(height)}</h1>', unsafe_allow_html=True)
        else: 
            break
    cap.release()