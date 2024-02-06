import streamlit as st

def main():
    st.title("Image Capture from Camera")

    # Open a webcam capture with a unique label
    camera_input = st.camera_input(label="my_camera_label")

    # Capture button
    if st.button("Capture Photo"):
        # Display the captured image
        st.image(camera_input, caption="Captured Image", use_column_width=True)

if __name__ == "__main__":
    main()
