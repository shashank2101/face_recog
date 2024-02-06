import streamlit as st
import cv2
import os
from mtcnn import MTCNN
from PIL import Image
import numpy as np
def main():
    st.set_page_config(page_title="New Registration", page_icon="👤")
    st.title("New Registration")
    person_name = st.text_input("Enter the person's name:")

    option = st.radio("Choose an option:", ("Upload Image", "Capture from Camera"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload image:", type=["jpg", "png"])

        if uploaded_file is not None:
            st.text("Processing image. Please wait...")
            image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            detector = MTCNN()

            # Detect faces using MTCNN
            faces = detector.detect_faces(image_cv)

            if len(faces) == 0:
                st.warning("No faces detected in the uploaded image.")
            else:
                # Only process the first detected face
                x, y, w, h = faces[0]['box']
                face_roi = image_cv[y:y+h, x:x+w]

                # Save the uploaded image
                if not person_name:
                    st.warning("Please enter a valid name.")
                    return

                # Set the base directory for storing the database
                base_directory = "D:/attendance/database"

                # Check if the folder already exists
                folder_path = os.path.join(base_directory, person_name)
                if os.path.exists(folder_path):
                    st.warning(f"A folder with the name '{person_name}' already exists.")
                else:
                    # Create a new folder
                    os.makedirs(folder_path)
                    st.success(f"Folder '{person_name}' created successfully.")

                    # Save the processed face image
                    image_path = os.path.join(folder_path, "registration_face.jpg")
                    cv2.imwrite(image_path, face_roi)
                    st.success("Image saved successfully.")

    elif option == "Capture from Camera":
        if st.button("Start Camera"):
            if not person_name:
                st.warning("Please enter a valid name.")
                return

            # Set the base directory for storing the database
            base_directory = "D:/attendance/database"

            # Check if the folder already exists
            folder_path = os.path.join(base_directory, person_name)
            if os.path.exists(folder_path):
                st.warning(f"A folder with the name '{person_name}' already exists.")
            else:
                # Create a new folder
                os.makedirs(folder_path)
                st.success(f"Folder '{person_name}' created successfully.")

                # Initialize MTCNN face detector
                detector = MTCNN()

                # Capture and save face image
                cap = cv2.VideoCapture(0)
                while True:
                    # Capture a frame from the webcam
                    ret, frame = cap.read()

                    # Detect faces using MTCNN
                    faces = detector.detect_faces(frame)

                    if faces:
                        # Only process the first detected face
                        x, y, w, h = faces[0]['box']
                        face_roi = frame[y:y+h, x:x+w]

                        # Save the captured image
                        image_path = os.path.join(folder_path, "registration_face.jpg")
                        cv2.imwrite(image_path, face_roi)
                        st.success("Face captured successfully.")
                        break

                    # Display the webcam feed
                    st.image(frame, channels="BGR", use_column_width=True)

    if st.button("Recapture", key="recapture"):
        if not person_name:
            st.warning("Please enter a valid name.")
            return
        base_directory = "D:/attendance/database"
        folder_path = os.path.join(base_directory, person_name)

        if os.path.exists(folder_path):
            # Remove the folder and its contents
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
            os.rmdir(folder_path)
            st.success("Previous registration deleted. Ready for recapture.")
        else:
            st.warning("Folder does not exist.")

if __name__ == "__main__":
    main()
