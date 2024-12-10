import streamlit as st
import os
from streamlit_option_menu import option_menu
from pain_recognition import process_video
from pose_estimation import run_pose_estimation
from plantar_pressure import run_plantar_pressure_analysis
from report_generation import generate_report
import tempfile

# Set page configuration
st.set_page_config(page_title="OrthoSynergy", layout="wide", initial_sidebar_state="expanded")

# Function to create patient folder
def create_patient_folder(patient_name, patient_age):
    patient_folder = os.path.join("data", "patient_data", f"{patient_name}_{patient_age}")
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    return patient_folder

# Function to save patient photo in the patient folder
def save_patient_photo(photo, patient_folder, patient_name, patient_age):
    if photo:
        # Define the file path for saving the photo as 'name_age.jpg'
        photo_path = os.path.join(patient_folder, f"{patient_name}_{patient_age}.jpg")
        
        # Save the photo to the defined path
        with open(photo_path, "wb") as f:
            f.write(photo.getvalue())
        return photo_path
    return None

def main():
    # Apply custom styling
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&display=swap');

    body {
         font-family: 'Merriweather', serif;
         background-color: #e6f7ff;
         color: black;
    }

    h1, h2, h3, h4, h5, p {
        font-family: 'Merriweather', serif;
        color: #0a246e;
    }

    .css-1d391kg {
        background: linear-gradient(to bottom, #f2f2f2, #d9d9d9);
        color: black;
    }

    .stApp {
        background-image: url("https://fmsportsortho.com/wp-content/uploads/2019/08/medical-bg-1.jpg");
        background-size: cover;
    }

    .stButton > button {
        background-color: #a2ccd3;
        color: black;
        border: 2px solid black;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 20px;
        cursor: pointer;
    }

    .stButton > button:hover {
        background-color: #5f9ea0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for patient details
    with st.sidebar:
        st.image(r"E:\Final Year Project\ortho_project\src\transparent_ortho_synergy_logo.png", use_container_width=True)
        st.title("Patient Details")
        patient_name = st.text_input("Patient Name")
        patient_age = st.text_input("Patient Age")

    # Initialize session state
    if "create_new_patient" not in st.session_state:
        st.session_state.create_new_patient = False
        st.session_state.patient_name = ""
        st.session_state.patient_age = ""
        st.session_state.patient_photo = None
        st.session_state.patient_folder = ""

    # Main page content
    if not st.session_state.create_new_patient:
        # Photo capture section
        st.subheader("Capture Patient Photo")
        if not st.session_state.patient_photo:
            patient_photo = st.camera_input("Take a picture")
            if patient_photo:
                # Create the patient folder
                patient_folder = create_patient_folder(patient_name, patient_age)
                # Save the photo in the patient folder
                photo_path = save_patient_photo(patient_photo, patient_folder, patient_name, patient_age)
                if photo_path:
                    st.session_state.patient_photo = photo_path
                    st.image(photo_path, caption="Captured Patient Photo", use_container_width=True)
                    st.success("Photo saved successfully!")
                else:
                    st.error("Failed to save the photo.")
        else:
            st.image(st.session_state.patient_photo, caption="Patient Photo", use_container_width=True)

        # Create new patient
        if st.button("Create New Patient"):
            if patient_name and patient_age and st.session_state.patient_photo:
                st.session_state.patient_name = patient_name
                st.session_state.patient_age = patient_age
                st.session_state.patient_folder = create_patient_folder(patient_name, patient_age)
                st.session_state.create_new_patient = True
                st.success("Patient created successfully!")
            else:
                st.warning("Please fill out patient details and capture a photo.")
    else:
        # Display the option menu
        selected_module = option_menu(
            menu_title=None,
            options=["Pain Recognition", "Plantar Pressure", "Pose Estimation", "Generate Report"],
            icons=["camera", "footprints", "person", "file-earmark-text"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )

        # Pain Recognition
        if selected_module == "Pain Recognition":
            process_video(st.session_state.patient_folder)

        # Plantar Pressure
        elif selected_module == "Plantar Pressure":
            run_plantar_pressure_analysis(st.session_state.patient_folder)

        # Pose Estimation
        elif selected_module == "Pose Estimation":
            run_pose_estimation(st.session_state.patient_folder)

        # Report Generation
        elif selected_module == "Generate Report":
            st.header("Generate Patient Report")
            if st.button("Generate Report"):
                pdf_path = generate_report(
                    st.session_state.patient_folder,
                    st.session_state.patient_name,
                    st.session_state.patient_age,
                    st.session_state.patient_photo,
                )
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="Download Report",
                        data=pdf_bytes,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                    )

            if st.button("Reset for New Patient"):
                st.session_state.create_new_patient = False
                st.session_state.patient_name = ""
                st.session_state.patient_age = ""
                st.session_state.patient_folder = ""
                st.session_state.patient_photo = None

if __name__ == "__main__":
    main()
