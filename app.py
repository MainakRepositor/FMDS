import streamlit as st
import cv2 , tempfile
import numpy as np
from PIL import Image
from src.components.image_detector import load_yolonas_process_each_image
from src.components.video_detector import load_yolonas_process_each_frame 


def main():
    st.title('Face Mask Detection with YOLOv8')
    st.sidebar.title('Options')
    st.sidebar.image('mona.jpg')
    st.sidebar.markdown('---')
    st.sidebar.subheader('')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left:-300px;
        }
        </style>
        """,
        unsafe_allow_html=True
        )
    
    app_mode = st.sidebar.selectbox(' Choose the App Mode ',['About App', 'Run on Image','Run on Video']) #,'Output/Processed Video'
    
    if app_mode =='About App':
        st.markdown('In this project I am using **YOLO-V8** model to do Face Mask Detection on Images and Videos and we are using ***StreamLit*** to create web application and GUI.')
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left:-300px;
        }
        </style>
        """,
        unsafe_allow_html=True
        )
        # st.video('')

        st.image('mona.jpg')
        
        st.markdown('''Real-time, image-based, and video-based face mask detection using YOLO (You Only Look Once) is a cutting-edge application of deep learning and computer vision technology that has gained significant importance in the wake of the COVID-19 pandemic. YOLO, a real-time object detection system, is particularly well-suited for this task due to its speed and accuracy. This innovative approach leverages convolutional neural networks to detect whether individuals are wearing face masks in real-time, enabling quick and efficient enforcement of mask mandates and ensuring public safety.\n\n The YOLO model's key strength lies in its ability to process images and video frames at an incredible speed, making it ideal for applications where real-time detection is critical. Face mask detection using YOLO is particularly effective in various scenarios, including airports, public transportation, schools, and healthcare facilities. The model can accurately identify individuals who are not wearing masks, providing a valuable tool for law enforcement and healthcare professionals.\n\nThe YOLO model operates by dividing an image or video frame into a grid of cells and assigning each cell the responsibility of predicting objects within its boundaries. In the context of face mask detection, YOLO is trained on a dataset containing labeled images of people with and without masks. The network learns to identify facial features and the presence or absence of a mask. YOLO can simultaneously detect multiple faces and their corresponding mask status within a single frame, allowing it to process crowded scenes efficiently.\n\nOne of the primary advantages of YOLO-based face mask detection is its real-time capabilities. The model can process frames at a speed of up to 60 frames per second, ensuring rapid and accurate detection. This real-time performance is crucial in applications such as security cameras, where immediate action may be required if a person without a mask is detected.\n\n''')

    elif app_mode=='Run on Image':
        

        confidence = st.sidebar.slider('Confidence', min_value=0.15, max_value=1.0)
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left:-300px;
        }
        </style>
        """,
        unsafe_allow_html=True
        )
        img_file_buffer = st.sidebar.file_uploader('Upload an Iamage', type=['jpg', 'jpeg', 'png'])
        
        Demo_image = 'sample_dataset/demo.jpg'
        
        if img_file_buffer is not None:
            img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8),1)
            image = np.array(Image.open(img_file_buffer))
        else:
            img = cv2.imread(Demo_image)
            image = np.array(Image.open(Demo_image))
        
        st.sidebar.text('Original Image')
        st.sidebar.image(image)
        
        load_yolonas_process_each_image(img, confidence, st)
        # logging.info(f"Run on image mode completed successfully")
    elif app_mode=='Run on Video':
        conf = st.sidebar.slider('Confidence', min_value=0.25, max_value=1.0)
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left:-300px;
        }
        </style>
        """,
        unsafe_allow_html=True
        )

        use_webcam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')
        video_file_buffer = st.sidebar.file_uploader('Upload a Video', type=["mp4","avi","mov","asf"])
        
        Demo_video = 'sample_dataset/demo.mp4'
        
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        st.markdown(
            """ Detection performance may vary as per your system configuration
            """)
        if not video_file_buffer:
            if use_webcam:
                tffile.name = 0
            else:
                tffile.name = Demo_video
                demo_vid = open(tffile.name , 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name , 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
        
        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html=True)
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Width**")
            kpi2_text = st.markdown("0")   
        with kpi3:
            st.markdown("**Height**")
            kpi3_text = st.markdown("0")
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        load_yolonas_process_each_frame(tffile.name, kpi1_text, kpi2_text, kpi3_text, stframe, conf)

    st.sidebar.info("Made by: Mainak")
  
            
if __name__=='__main__':
    try:
        main()
    except Exception as e:
        raise Exception(e)

        