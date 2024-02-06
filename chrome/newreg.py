import streamlit as st
import cv2
from mtcnn.mtcnn import MTCNN
import os

def preprocess_image(face):
    # Add any necessary preprocessing steps here
    return face

def main():
    st.set_page_config(page_title="Face Registration", page_icon="👤")
    st.title("Face Registration with Streamlit and Webcam")

    folder_name = st.text_input("Enter the new student's folder name:")
    if st.button("Start Registration"):
        if folder_name:
            st.sidebar.text("Captured Image:")
            captured_image_placeholder = st.sidebar.empty()
            # video_capture=cv2.VideoCapture(0)
            # Open the webcam
            capture = cv2.VideoCapture(0)

            # Initialize MTCNN for face detection
            mtcnn_detector = MTCNN(min_face_size=30)

            # Flag to control the capture process
            capturing = True

            while capturing:
                # Read a frame from the webcam
                ret, frame = capture.read()

                # Face detection using MTCNN
                faces = mtcnn_detector.detect_faces(frame)

                if len(faces) == 1:
                    # Only one face detected
                    x, y, w, h = faces[0]['box']
                    face_roi = frame[y:y+h, x:x+w]
                    print("face detected")
                    # Resize the face image to match the input size of your model
                    face_resized = cv2.resize(face_roi, (220, 220) )

                    # Preprocess the face image
                    preprocessed_face = preprocess_image(face_resized)
                    folder_path = f"D:/attendance/database/{folder_name}/"
                    image_filename = "new_registration_face.jpg"
                    image_path = os.path.join(folder_path, image_filename)
                    flag=1
                    if not (os.path.exists(folder_path)):
                        os.makedirs(folder_path)
                        if cv2.imwrite(image_path, cv2.cvtColor(preprocessed_face, cv2.COLOR_RGB2BGR)):
                            st.text(image_path)
                            captured_image_placeholder.image(preprocessed_face, channels="BGR", caption="Captured Image", use_column_width=True)
                            st.success("Face captured successfully.")
                            cv2.putText(frame, "Captured successfully", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            # cv2.imshow("Attendance System", frame)
                            capturing = False
                            flag=0
                        else:
                            st.error("Error saving the image.")
                    else:
                        st.error(f"Folder '{folder_name}' already exists. Choose a different folder name.")
                        break
                # Display the result on the frame
                st.image(frame,"Captured image",width=200, channels="BGR", use_column_width=True)
                if not capturing:
                    recapture_button = st.button("Recapture")
                    if recapture_button:
                        print("image removed")
                        os.remove(image_path)   
                        st.success("Previous capture deleted. Ready for recapture.")
                        capturing = True
                        flag=1

            # Close the webcam
            capture.release()

if __name__ == "__main__":
    main()
