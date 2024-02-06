import os
import cv2
import streamlit as st
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
def main():
    st.set_page_config(page_title="Face Registration", page_icon="👤")
    st.title("Face Registration")

    # Prompt for the person's name
    person_name = st.text_input("Enter the person's name:")

    if not person_name:
        st.warning("Please enter a valid name.")
        return

    # Set the base directory for storing the database
    base_directory = "D:/attendance/database/"
    folder_path = os.path.join(base_directory, person_name)

    if os.path.exists(folder_path):

        st.sidebar.header("Face Registration")

        # Choose between capturing images or uploading images
        capture_option = st.sidebar.radio("Choose an option:", ("Capture from Webcam", "Upload Images"))

        # Initialize MTCNN face detector
        mtcnn_detector = MTCNN()

        
        if capture_option == "Upload Images":
            # Upload images from the user's device
            uploaded_files = st.file_uploader("Upload face images (JPG or PNG)", type=["jpg", "png"], accept_multiple_files=True)

            if uploaded_files:
                for i, uploaded_file in enumerate(uploaded_files):
                    # Convert uploaded file to OpenCV format
                    image = Image.open(uploaded_file)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # Face detection using MTCNN
                    faces = mtcnn_detector.detect_faces(image_cv)

                    if len(faces) == 1:
                        # Only one face detected
                        x, y, w, h = faces[0]['box']
                        face_roi = image_cv[y:y + h, x:x + w]

                        # Save the face image
                        image_filename = f"registration_face_{i + 1}.jpg"
                        image_path = os.path.join(folder_path, image_filename)
                        cv2.imwrite(image_path, cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        st.success(f"Image {i + 1} saved from uploaded file")
        elif capture_option == "Capture from Webcam":
        # Number of pictures to be taken
            num_pictures = st.sidebar.slider("Number of Pictures", min_value=1, max_value=10, value=5)

            # Capture and save face images
            captured_count = 0
            cap = cv2.VideoCapture(0)

            while captured_count < num_pictures:
                # Display the webcam feed
                ret, frame = cap.read()

                # Face detection using MTCNN
                faces = mtcnn_detector.detect_faces(frame)
                st.image(frame, channels="BGR", use_column_width=True)

                if len(faces) == 1:
                    # Only one face detected
                    x, y, w, h = faces[0]['box']
                    face_roi = frame[y:y + h, x:x + w]

                    # Display the face region in the sidebar
                    st.sidebar.image(face_roi, channels="BGR", caption="Detected Face", use_column_width=True)

                    image_filename = f"registration_face_{captured_count + 1}.jpg"
                    image_path = os.path.join(folder_path, image_filename)
                    cv2.imwrite(image_path, cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    st.success(f"Image {captured_count + 1} saved")
                    captured_count += 1

            # Release the webcam
            cap.release()
    

if __name__ == "__main__":
    main()
