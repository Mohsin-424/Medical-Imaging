import streamlit as st
import os
import tempfile
from streamlit_option_menu import option_menu
from pain_recognition import process_video
from pose_estimation import run_pose_estimation
from plantar_pressure import run_plantar_pressure_analysis
from report_generation import generate_report

# Set page configuration
st.set_page_config(page_title="OrthoSynergy", layout="wide", initial_sidebar_state="expanded")

# Function to create patient folder
def create_patient_folder(patient_name, patient_age):
    patient_folder = os.path.join("data", "patient_data", f"{patient_name}_{patient_age}")
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    return patient_folder

def main():
    # Custom styling for fonts, sidebar, and font colors
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&display=swap');

    body {
         font-family: 'Merriweather', serif;
         background-color: #e6f7ff;
         color: black;
    }

    h1, h2, h3, h4, h5, p {
        font-family: 'Merriweather', serif;
        color: #0a246e;  /* Set headings and paragraph text color to black */
    }

    .css-1d391kg {
        background: linear-gradient(to bottom, #f2f2f2, #d9d9d9);  /* Sidebar gradient color - light gray */
        color: black;  /* Sidebar text color */
    }

    .stApp {
        background-image: url("https://fmsportsortho.com/wp-content/uploads/2019/08/medical-bg-1.jpg"); /* Optional background image */
        background-size: cover;
    }

    .css-17eq0hr {
        color: black;  /* Sidebar text color */
    }

    .css-1n543e5 {
        color: #ffcc00;  /* Sidebar title color*/
    }

    .css-1e5imcs {
        background-color: #f0f8ff;  /* Light blue background for main container */
    }

    /* Option menu customization */
    .stOptionMenu {
        background-color: #ffffff;  /* Option menu background color */
        border-radius: 5px;  /* Optional: rounded corners */
    }

    .stOptionMenu .css-1e5imcs {
        color: black;  /* Menu item text color */
    }

    .stOptionMenu .css-1e5imcs:hover {
        background-color: #146075;  /* Hover background color */
        color: white;  /* Hover text color */
    }

    /* Button customization */
    .stButton > button {
        background-color: #a2ccd3;  /* Light blue color */
        color: black;  /* Button text color */
        border: 2px solid black;  /* Remove border */
        border-radius: 5px;  /* Rounded corners for buttons */
        padding: 10px 20px;  /* Padding for buttons */
        font-size: 20px;  /* Font size for button text */
        cursor: pointer;  /* Cursor changes to pointer on hover */
    }

    /* Button hover effect */
    .stButton > button:hover {
        background-color: #5f9ea0;  /* Darker shade of light blue on hover */
    }
    </style>
""", unsafe_allow_html=True)  

    # Optional logo - local image
    st.sidebar.image(r"E:\Final Year Project\ortho_project\src\transparent_ortho_synergy_logo.png", use_column_width=True)

    # Initialize session state for new patients
    if 'create_new_patient' not in st.session_state:
        st.session_state.create_new_patient = False
        st.session_state.patient_name = ""
        st.session_state.patient_age = ""
        st.session_state.patient_folder = ""

    # Sidebar menu for new patient creation
    st.sidebar.title("Enter Patient Details")
    st.session_state.patient_name = st.sidebar.text_input("Patient Name")
    st.session_state.patient_age = st.sidebar.text_input("Patient Age")

    if st.sidebar.button("Create New Patient"):
        if st.session_state.patient_name and st.session_state.patient_age:
            st.session_state.patient_folder = create_patient_folder(
                st.session_state.patient_name, st.session_state.patient_age
            )
            st.session_state.create_new_patient = True
            st.sidebar.empty()
        else:
            st.sidebar.warning("Please enter both patient name and age.")

    if st.session_state.create_new_patient:
        # Option menu for module navigation
        selected_module = option_menu(
            menu_title=None,
            options=["Pain Recognition", "Plantar Pressure", "Pose Estimation", "Generate Report"],
            icons=["camera", "footprints", "person", "file-earmark-text"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal"
        )

        # Pain Recognition
        if selected_module == "Pain Recognition":
            process_video(st.session_state.patient_folder)

        # Plantar Pressure
        if selected_module == "Plantar Pressure":
            run_plantar_pressure_analysis(st.session_state.patient_folder)

        # Pose Estimation
        if selected_module == "Pose Estimation":
            run_pose_estimation(st.session_state.patient_folder)

        # Report Generation
        if selected_module == "Generate Report":
            st.header("Generate Patient Report")
            if st.button("Generate Report"):
                if st.session_state.patient_folder:
                    # Generate the report and store it temporarily
                    pdf_path = generate_report(
                        st.session_state.patient_folder,
                        st.session_state.patient_name,
                        st.session_state.patient_age
                    )
                    # Provide a download link for the report
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        st.download_button(
                            label="Download Report",
                            data=pdf_bytes,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf"
                        )
                else:
                    st.warning("Patient folder not found.")
                
            if st.button("Reset for New Patient"):
                st.session_state.create_new_patient = False
                st.session_state.patient_name = ""
                st.session_state.patient_age = ""
                st.session_state.patient_folder = ""

if __name__ == "__main__":
    main()
