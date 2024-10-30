import streamlit as st
import serial
import numpy as np
import plotly.express as px
import os
import time
import datetime
import threading
import cv2  # Import OpenCV for screenshots

# Global variables to track serial connections
ser_left = None
ser_right = None
stop_event = None
pressure_matrix_left = None
pressure_matrix_right = None

def run_plantar_pressure_analysis(patient_folder):
    global ser_left, ser_right, stop_event, pressure_matrix_left, pressure_matrix_right

    # Initialize session state variables
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
        st.session_state.show_save_button = False
        st.session_state.left_pressure_data = []
        st.session_state.right_pressure_data = []
        st.session_state.frame_for_screenshot = None  # Initialize frame for screenshots
        st.session_state.screenshot_counter = 0  # Initialize screenshot counter

    # Display Start/Stop buttons based on the running state
    if st.session_state.is_running:
        st.button('Stop Plantar Pressure Analysis', key='stop_analysis', on_click=stop_plantar_pressure)
        st.button('Take Screenshot', key='screenshot_button', on_click=take_screenshot, args=(patient_folder,))
    else:
        st.button('Start Plantar Pressure Analysis', key='start_analysis', on_click=start_plantar_pressure)
        st.button('Reset State', key='reset_state', on_click=reset_state)

    # Create empty containers for side-by-side heatmaps (outside the loop)
    col1, col2 = st.columns(2)  # Two columns for left and right foot heatmaps
    left_placeholder = col1.empty()  # Placeholder for left foot heatmap
    right_placeholder = col2.empty()  # Placeholder for right foot heatmap

    # Display the heatmaps if the analysis is running
    if st.session_state.is_running:
        while not stop_event.is_set():
            if ser_left and pressure_matrix_left is not None:
                fig_left = px.imshow(pressure_matrix_left, color_continuous_scale='hot',
                                     labels={'color': 'Pressure'},
                                     title="Left Foot Pressure",
                                     zmin=1, zmax=100, aspect='equal')
                fig_left.update_traces(zsmooth='best')
                left_placeholder.plotly_chart(fig_left, use_container_width=True)  # Update the left heatmap

            if ser_right and pressure_matrix_right is not None:
                fig_right = px.imshow(pressure_matrix_right, color_continuous_scale='hot',
                                      labels={'color': 'Pressure'},
                                      title="Right Foot Pressure",
                                      zmin=1, zmax=100, aspect='equal')
                fig_right.update_traces(zsmooth='best')
                right_placeholder.plotly_chart(fig_right, use_container_width=True)  # Update the right heatmap

            # Collect data for screenshots
            st.session_state.left_pressure_data.append(pressure_matrix_left)
            st.session_state.right_pressure_data.append(pressure_matrix_right)

            # Save the current frame for screenshot
            st.session_state.frame_for_screenshot = pressure_matrix_left  # Assign current left matrix to screenshot
            time.sleep(0.1)  # Adjust update rate

def start_plantar_pressure():
    global ser_left, ser_right, stop_event, pressure_matrix_left, pressure_matrix_right

    try:
        ser_left = serial.Serial('COM5', 9600, timeout=1)
    except serial.SerialException:
        st.error("Left foot COM port not connected.")
    
    try:
        ser_right = serial.Serial('COM6', 9600, timeout=1)
    except serial.SerialException:
        st.error("Right foot COM port not connected.")

    pressure_matrix_left = np.zeros((6, 3)) if ser_left else None
    pressure_matrix_right = np.zeros((6, 3)) if ser_right else None

    stop_event = threading.Event()  # Create the stop event
    thread = threading.Thread(target=read_pressure_matrices, daemon=True)
    thread.start()

    st.session_state.is_running = True

def stop_plantar_pressure():
    global stop_event, ser_left, ser_right

    stop_event.set()  # Stop the reading loop
    if ser_left:
        ser_left.close()
    if ser_right:
        ser_right.close()

    st.success("Plantar pressure analysis stopped.")
    st.session_state.is_running = False
    st.session_state.show_save_button = True  # Enable the Save Report button

def read_pressure_matrices():
    global pressure_matrix_left, pressure_matrix_right, stop_event
    while not stop_event.is_set():
        if ser_left:
            new_matrix_left = np.zeros((6, 3))
            for i in range(6):
                try:
                    line_left = ser_left.readline().decode().strip()
                    if line_left:
                        pressure_row_left = list(map(int, line_left.split(',')))
                        if len(pressure_row_left) == 3:
                            new_matrix_left[i] = pressure_row_left
                except Exception as e:
                    print("Error reading from left foot Arduino:", e)
            pressure_matrix_left[:] = new_matrix_left

        if ser_right:
            new_matrix_right = np.zeros((6, 3))
            for i in range(6):
                try:
                    line_right = ser_right.readline().decode().strip()
                    if line_right:
                        pressure_row_right = list(map(int, line_right.split(',')))
                        if len(pressure_row_right) == 3:
                            new_matrix_right[i] = pressure_row_right
                except Exception as e:
                    print("Error reading from right foot Arduino:", e)
            pressure_matrix_right[:] = new_matrix_right

        time.sleep(0.01)  # Control update rate

def reset_state():
    st.session_state.is_running = False
    st.session_state.show_save_button = False
    st.session_state.left_pressure_data = []
    st.session_state.right_pressure_data = []
    st.session_state.frame_for_screenshot = None  # Reset frame for screenshots
    st.session_state.screenshot_counter = 0  # Reset screenshot counter
    st.success('Session state reset.')

def take_screenshot(patient_folder):
    # Check if there's a valid frame to save
    if st.session_state.frame_for_screenshot is not None and st.session_state.frame_for_screenshot.size > 0:
        # Create a timestamp for the screenshot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_name = st.session_state.get('patient_name', 'Unknown')
        patient_age = st.session_state.get('patient_age', 'Unknown')

        # Save left foot screenshot
        left_image_path = os.path.join(patient_folder, f'left_foot_screenshot_{patient_name}_{patient_age}_{timestamp}.png')

        # Convert to color image
        left_image_color = cv2.applyColorMap((st.session_state.frame_for_screenshot * 255).astype(np.uint8), cv2.COLORMAP_JET)
        left_image_color = cv2.resize(left_image_color, (600, 400))  # Resize as needed
        cv2.imwrite(left_image_path, left_image_color)

        # Save right foot screenshot if it exists
        if pressure_matrix_right is not None and pressure_matrix_right.size > 0:
            right_image_path = os.path.join(patient_folder, f'right_foot_screenshot_{patient_name}_{patient_age}_{timestamp}.png')
            
            # Convert to color image
            right_image_color = cv2.applyColorMap((pressure_matrix_right * 255).astype(np.uint8), cv2.COLORMAP_JET)
            right_image_color = cv2.resize(right_image_color, (600, 400))  # Resize as needed
            cv2.imwrite(right_image_path, right_image_color)

        st.success(f"Screenshots saved in {patient_folder}.")
        st.session_state.screenshot_counter += 1
    else:
        st.warning('No frame available to save a screenshot.')
