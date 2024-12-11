import numpy as np
import plotly.express as px
import os
import time
import datetime
import threading
import cv2
import tempfile  # Import tempfile for temporary file storage
from scipy.interpolate import griddata
import streamlit as st
import pandas as pd
import tempfile
import serial

import numpy as np
import plotly.express as px
import os
import time
import datetime
import threading
import cv2
import tempfile
from scipy.interpolate import griddata
import streamlit as st
import pandas as pd
import serial

# Global variables to track serial connections
ser_left = None
ser_right = None
stop_event = None
pressure_matrix_left = None
pressure_matrix_right = None

# Custom sensor positions for left foot
left_foot_sensor_positions = [
    (183, 41), (220, 42), (220, 42),
    (164, 85), (196, 84), (225, 84),
    (160, 134), (192, 132), (224, 130),
    (165, 185), (193, 184), (220, 182),
    (171, 238), (191, 237), (215, 235),
    (176, 301), (208, 299), (208, 299)
]

# Custom sensor positions for right foot with X-axis inversion
right_foot_sensor_positions = [
    (220, 42), (220, 42), (183, 41),
    (225, 84), (196, 84), (164, 85),
    (224, 130), (192, 132), (160, 134),
    (220, 182), (193, 184), (165, 185),
    (215, 235), (191, 237), (171, 238),
    (208, 299), (208, 299), (176, 301)
]
def stop_plantar_pressure():
    global stop_event, ser_left, ser_right

    # Signal the thread to stop
    stop_event.set()

    # Close the left foot serial port
    if ser_left:
        try:
            ser_left.close()
            ser_left = None  # Clear the reference
            st.success("Left foot COM port closed.")
        except Exception as e:
            st.error(f"Error closing left foot COM port: {e}")

    # Close the right foot serial port
    if ser_right:
        try:
            ser_right.close()
            ser_right = None  # Clear the reference
            st.success("Right foot COM port closed.")
        except Exception as e:
            st.error(f"Error closing right foot COM port: {e}")

    # Update session state
    st.success("Plantar pressure analysis stopped.")
    st.session_state.is_running = False
    st.session_state.show_save_button = True

def start_plantar_pressure():
    global ser_left, ser_right, stop_event, pressure_matrix_left, pressure_matrix_right

    try:
        ser_left = serial.Serial('COM7', 9600, timeout=1)
    except serial.SerialException:
        st.error("Left foot COM port not connected.")
    
    try:
        ser_right = serial.Serial('COM5', 9600, timeout=1)
    except serial.SerialException:
        st.error("Right foot COM port not connected.")

    pressure_matrix_left = np.zeros((6, 3)) if ser_left else None
    pressure_matrix_right = np.zeros((6, 3)) if ser_right else None

    stop_event = threading.Event()
    thread = threading.Thread(target=read_pressure_matrices, daemon=True)
    thread.start()

    st.session_state.is_running = True
# Function to scale the sensor positions by a factor
def scale_sensor_positions(sensor_positions, scale_factor_x=1.0, scale_factor_y=1.0):
    return [(x * scale_factor_x, y * scale_factor_y) for x, y in sensor_positions]

# Scale the left and right foot sensor positions
scale_factor_x = 2.5 # For narrowing the foot
scale_factor_y = 2.0 # For elongating the foot

left_foot_sensor_positions = scale_sensor_positions(left_foot_sensor_positions, scale_factor_x, scale_factor_y)
right_foot_sensor_positions = scale_sensor_positions(right_foot_sensor_positions, scale_factor_x, scale_factor_y)

# Function to plot interpolated pressure map
def plot_interpolated_pressure_map(pressure_matrix, sensor_positions, grid_resolution=(25, 25), is_right_foot=False):
    # Create a meshgrid for interpolation
    x = np.array([pos[0] for pos in sensor_positions])
    y = np.array([pos[1] for pos in sensor_positions])
    z = pressure_matrix.flatten()

    # If it's the right foot, invert the X-axis
    if is_right_foot:
        x = max(x) - x  # Invert the X values

    # Invert the Y-axis by subtracting from the maximum Y value
    max_y = max(y)
    y_inverted = max_y - y  # Inverting the Y-axis

    # Generate a regular grid for interpolation
    grid_x, grid_y = np.mgrid[min(x):max(x):grid_resolution[0]*1j, min(y_inverted):max(y_inverted):grid_resolution[1]*1j]

    # Perform 2D interpolation on the pressure data
    grid_z = griddata((x, y_inverted), z, (grid_x, grid_y), method='linear')

    # Create a DataFrame for the interpolated heatmap
    df = pd.DataFrame({
        'x': grid_x.flatten(),
        'y': grid_y.flatten(),
        'pressure': grid_z.flatten()
    })

    # Explicitly set color scale and range
    color_scale = 'Blackbody'
    color_range = [1, 100]  # You can adjust this based on your data

    # Plot the interpolated pressure map using Plotly
    fig = px.scatter(df, x='x', y='y', color='pressure', color_continuous_scale=color_scale,
                     title="Interpolated Plantar Pressure Map", labels={'pressure': 'Pressure'},
                     range_color=color_range)

    # Update trace and layout to control the appearance
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='black')))

    # Set the aspect ratio to reflect the desired foot shape
    fig.update_layout(
        xaxis=dict(scaleanchor="y"),  # This links the x and y axes
        yaxis=dict(scaleanchor="x"),
        autosize=True,
        coloraxis_colorbar=dict(title="Pressure")
    )

    return fig


# Adjust the plotting logic for the left and right feet
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
        st.button('Take Screenshot', key='screenshot_button', on_click=lambda: take_screenshot(patient_folder))
    else:
        st.button('Start Plantar Pressure Analysis', key='start_analysis', on_click=start_plantar_pressure)
        st.button('Reset State', key='reset_state', on_click=reset_state)

    # Create empty containers for side-by-side heatmaps
    col1, col2 = st.columns(2)
    left_placeholder = col1.empty()
    right_placeholder = col2.empty()

    # Display the heatmaps if the analysis is running
    if st.session_state.is_running:
        while not stop_event.is_set():
            if ser_left and pressure_matrix_left is not None:
                fig_left = plot_interpolated_pressure_map(pressure_matrix_left, left_foot_sensor_positions, is_right_foot=False)
                left_placeholder.plotly_chart(fig_left, use_container_width=True, key=f"left_foot_{st.session_state.screenshot_counter}")

            if ser_right and pressure_matrix_right is not None:
                fig_right = plot_interpolated_pressure_map(pressure_matrix_right, right_foot_sensor_positions, is_right_foot=True)
                right_placeholder.plotly_chart(fig_right, use_container_width=True, key=f"right_foot_{st.session_state.screenshot_counter}")

            # Collect data for screenshots
            st.session_state.left_pressure_data.append(pressure_matrix_left)
            st.session_state.right_pressure_data.append(pressure_matrix_right)

            # Save the current frame for screenshot
            st.session_state.frame_for_screenshot = pressure_matrix_left
            time.sleep(0.05)  # Adjust update rate
            st.session_state.screenshot_counter += 1


def take_screenshot(patient_folder):
    # Check if there's valid data for both feet to save
    if (st.session_state.frame_for_screenshot is not None and 
        st.session_state.frame_for_screenshot.size > 0 and 
        pressure_matrix_left is not None and 
        pressure_matrix_right is not None):

        # Create a timestamp for the screenshot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define screenshot filenames
        screenshot_filename_left = f"plantar_pressure_left_{timestamp}.png"
        screenshot_filename_right = f"plantar_pressure_right_{timestamp}.png"

        # Ensure the patient folder exists
        os.makedirs(patient_folder, exist_ok=True)

        # Define the full path for saving screenshots
        screenshot_path_left = os.path.join(patient_folder, screenshot_filename_left)
        screenshot_path_right = os.path.join(patient_folder, screenshot_filename_right)

        # Generate and save the heatmap figures for both left and right feet
        fig_left = plot_interpolated_pressure_map(st.session_state.frame_for_screenshot, left_foot_sensor_positions)
        fig_right = plot_interpolated_pressure_map(pressure_matrix_right, right_foot_sensor_positions)

        # Save the screenshots as PNG files
        fig_left.write_image(screenshot_path_left)
        fig_right.write_image(screenshot_path_right)

        st.success(f"Screenshots saved to {patient_folder}")

# Other functions remain the same...



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

        time.sleep(0.05)

def reset_state():
    st.session_state.is_running = False
    st.session_state.show_save_button = False
    st.session_state.left_pressure_data = []
    st.session_state.right_pressure_data = []
    st.session_state.frame_for_screenshot = None
    st.session_state.screenshot_counter = 0
    st.success('Session state reset.')