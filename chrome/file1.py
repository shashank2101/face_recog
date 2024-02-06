import streamlit as st
import cv2
import os
from mtcnn import MTCNN

def main():
    st.set_page_config(page_title="New Registration", page_icon="👤")
    st.title("New Registration")

    # Prompt for the person's name
    person_name = st.text_input("Enter the person's name:")
    if st.button("Start Registration", key="start_registration"):
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
            captured_image_placeholder = st.sidebar.empty()
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
                    image_filename = f"registration_face.jpg"
                    image_path = os.path.join(folder_path, image_filename)
                    cv2.imwrite(image_path, face_roi)
                    st.image(frame, "captured image")
                    # st.sidebar().image(face_roi, channels="BGR", use_column_width=True)
                    captured_image_placeholder.image(face_roi, channels="RGB", caption="Captured Image", use_column_width=True)
                    st.success("Face captured successfully.")
                    break

                # Display the webcam feed

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
