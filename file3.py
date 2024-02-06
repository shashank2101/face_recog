import streamlit as st
import cv2
import os
from mtcnn import MTCNN
from PIL import Image
import numpy as np

def main():
    st.set_page_config(page_title="Multiple Images Registration", page_icon="📸")
    st.title("Multiple Images Registration")

    # Prompt for the person's name
    person_name = st.text_input("Enter the person's name:")

    # Prompt for the number of images to be taken
    num_images = st.number_input("Enter the number of images to be taken:", min_value=1, step=1)

    # Option to choose between uploading images or capturing from camera
    option = st.radio("Choose an option:", ("Upload Images", "Capture from Camera"))

    if option == "Upload Images":
        uploaded_files = st.file_uploader("Upload face images (JPG or PNG)", type=["jpg", "png"], accept_multiple_files=True)

        if uploaded_files is not None and len(uploaded_files) >= num_images:
            # Check if the folder exists
            folder_path = os.path.join("database", person_name)
            if os.path.exists(folder_path):
                # Save uploaded images
                for i, uploaded_file in enumerate(uploaded_files[:num_images]):
                    image = Image.open(uploaded_file)
                    image.save(os.path.join(folder_path, f"{person_name}_{i+1}.jpg"))
                st.success(f"{num_images} images saved successfully.")
            else:
                st.warning(f"Folder '{person_name}' does not exist.")

    elif option == "Capture from Camera":
        if st.button("Start Camera"):
            if not person_name:
                st.warning("Please enter a valid name.")
                return

            # Check if the folder exists
            folder_path = os.path.join("database", person_name)
            if os.path.exists(folder_path):
                # Initialize MTCNN face detector
                detector = MTCNN()

                # Capture and save face images
                cap = cv2.VideoCapture(0)
                img_count = 0
                while img_count < num_images:
                    # Capture a frame from the webcam
                    ret, frame = cap.read()

                    # Detect faces using MTCNN
                    faces = detector.detect_faces(frame)

                    if faces:
                        # Only process the first detected face
                        x, y, w, h = faces[0]['box']
                        face_roi = frame[y:y+h, x:x+w]

                        # Save the captured image
                        img_count += 1
                        cv2.imwrite(os.path.join(folder_path, f"{person_name}_{img_count}.jpg"), face_roi)
                        st.success(f"Image {img_count} captured.")

                    # Display the webcam feed
                    st.image(frame, channels="BGR", use_column_width=True)

                # Release the camera
                cap.release()
            else:
                st.warning(f"Folder '{person_name}' does not exist.")

    # Button to delete the folder and its contents
    if st.button("Delete Folder"):
        if not person_name:
            st.warning("Please enter a valid name.")
            return
        folder_path = os.path.join("database", person_name)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
            os.rmdir(folder_path)
            st.success(f"Folder '{person_name}' deleted successfully.")
        else:
            st.warning(f"Folder '{person_name}' does not exist.")

if __name__ == "__main__":
    main()
