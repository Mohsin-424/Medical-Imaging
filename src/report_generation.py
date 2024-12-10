import tempfile
import os
from fpdf import FPDF
from PIL import Image
import datetime

class StyledPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "Patient Report", 0, 1, "C")
        self.set_draw_color(0, 0, 0)
        self.line(10, 20, 200, 20)  # Horizontal line

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

import os

def generate_report(patient_folder, patient_name, patient_age, patient_photo):
    # Create a persistent folder for reports if it doesn't exist
    reports_folder = "reports"
    if not os.path.exists(reports_folder):
        os.makedirs(reports_folder)

    # Create a PDF document
    pdf = StyledPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Unique number based on the number of reports already generated
    report_number = len([name for name in os.listdir(reports_folder) if name.startswith(f"{patient_name}_{patient_age}")]) + 1

    # Report filename
    report_filename = f"{patient_name}_{patient_age}_{report_number}.pdf"
    report_path = os.path.join(reports_folder, report_filename)

    # Add a page for the report
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title with patient name and age
    pdf.set_fill_color(220, 220, 220)  # Light gray background
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}, Age: {patient_age}", ln=True, align='C', fill=True)
    pdf.ln(5)  # Add a line break

    # Add the patient's photo
    patient_photo_path = os.path.join(patient_folder, f"{patient_name}_{patient_age}.jpg")  # Assuming the photo is named after the patient
    if os.path.exists(patient_photo_path):
        try:
            pdf.image(patient_photo_path, x=85, y=pdf.get_y(), w=40, h=40)  # Adjust size as needed
            pdf.ln(45)  # Add a line break after the photo
        except Exception as e:
            print(f"Error adding photo: {e}")
            pdf.cell(0, 10, txt="Patient photo not available.", ln=True, align='C')  # Display error if photo fails to load
    else:
        pdf.cell(0, 10, txt="Patient photo not available.", ln=True, align='C')  # Display message if photo file is missing

    # Title for the static pressure distribution
    pdf.cell(0, 10, txt="Static Foot Pressure Distribution", ln=True, align='C', fill=True)
    pdf.ln(10)  # Add a line break

    # Add images from the patient folder to the PDF
    image_files = [f for f in os.listdir(patient_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

    # Add foot heatmaps side by side
    left_foot_image_added = False
    for image_file in image_files:
        image_path = os.path.join(patient_folder, image_file)

        # Check if the image file exists
        if os.path.exists(image_path):
            if "left" in image_file.lower() and not left_foot_image_added:
                # Add left foot heatmap
                pdf.image(image_path, x=10, y=pdf.get_y() + 5, w=90, h=100)  # Adjust width and height as needed
                left_foot_image_added = True
            elif "right" in image_file.lower() and left_foot_image_added:
                # Add right foot heatmap
                pdf.image(image_path, x=110, y=pdf.get_y() + 5, w=90, h=100)  # Adjust width and height as needed
                pdf.ln(95)  # Add space after both images
                break  # Break after adding both images

    # Add pain spike graph image on a new page
    pdf.add_page()
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}, Age: {patient_age}", ln=True, align='C')
    pdf.ln(5)  # Add a line break
    pdf.cell(0, 10, txt="Pain Spike Graph", ln=True, align='C', fill=True)
    pdf.ln(10)  # Add a line break

    pain_spike_images = [img for img in image_files if "pain_level" in img.lower()]
    if pain_spike_images:
        pain_spike_image = pain_spike_images[0]
        pain_spike_path = os.path.join(patient_folder, pain_spike_image)
        pdf.image(pain_spike_path, x=10, y=pdf.get_y() + 5, w=180, h=90)  # Adjust width and height as needed

    # Add body pose and limb angles image on a new page
    pdf.add_page()
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}, Age: {patient_age}", ln=True, align='C')
    pdf.ln(5)  # Add a line break
    pdf.cell(0, 10, txt="Body Pose and Limb Angles", ln=True, align='C', fill=True)
    pdf.ln(10)  # Add a line break

    pose_images = [img for img in image_files if "pose" in img.lower() or "limb_angles" in img.lower()]
    if pose_images:
        pose_image = pose_images[0]
        pose_path = os.path.join(patient_folder, pose_image)
        pdf.image(pose_path, x=10, y=pdf.get_y() + 5, w=180, h=220)  # Adjust width and height as needed

    # Save the PDF to the persistent directory
    pdf.output(report_path)

    # Return the path of the generated report
    return report_path
