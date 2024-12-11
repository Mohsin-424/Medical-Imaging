import cv2
import mediapipe as mp
import math
import streamlit as st
import os
import datetime
import tempfile


class poseDetector:
    def __init__(self, mode=False, modelComp=1, smoothlm=True, segment=False, smoothsegment=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.modelComp = modelComp
        self.smoothlm = smoothlm
        self.segment = segment
        self.smoothsegment = smoothsegment
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComp, self.smoothlm, self.segment, self.smoothsegment, self.detectionCon, self.trackingCon)

    def findPose(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return frame

    def findPosition(self, frame, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return self.lmList

    def findAngle(self, frame, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        radians = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        angle = math.degrees(radians)
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv2.circle(frame, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, str(int(angle)), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        return angle

def run_pose_estimation(patient_folder):
    if 'start_pose' not in st.session_state:
        st.session_state.start_pose = False
    if 'frame_for_screenshot' not in st.session_state:
        st.session_state.frame_for_screenshot = None
    if 'screenshot_counter' not in st.session_state:
        st.session_state.screenshot_counter = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Pose Estimation", key="start_pose_estimation"):
            st.session_state.start_pose = True

    with col2:
        if st.button("Stop Pose Estimation", key="stop_pose_estimation"):
            st.session_state.start_pose = False

    with col3:
        if st.button("Take Screenshot", key="screenshot_button"):
            if st.session_state.frame_for_screenshot is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Save screenshot in patient folder
                screenshot_filename = os.path.join(patient_folder, f"pose_screenshot_{timestamp}.png")
                cv2.imwrite(screenshot_filename, st.session_state.frame_for_screenshot)
                st.write(f"Screenshot saved as {screenshot_filename}")
                st.session_state.screenshot_counter += 1
                st.session_state.latest_screenshot = screenshot_filename
            else:
                st.write("No frame available to capture.")

    if st.session_state.start_pose:
        stframe = st.empty()
        cap = cv2.VideoCapture(1)
        detector = poseDetector()

        while st.session_state.start_pose:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break

            resized = cv2.resize(frame, (500, 600))
            frame_with_pose = detector.findPose(resized)
            lmList = detector.findPosition(frame_with_pose, draw=False)

            if len(lmList) != 0:
                detector.findAngle(frame_with_pose, 23, 11, 13)
                detector.findAngle(frame_with_pose, 24, 12, 14)
                detector.findAngle(frame_with_pose, 23, 25, 27)
                detector.findAngle(frame_with_pose, 24, 26, 28)

            stframe.image(frame_with_pose, channels="BGR")
            st.session_state.frame_for_screenshot = frame_with_pose

        cap.release()
        cv2.destroyAllWindows()

    if 'latest_screenshot' in st.session_state:
        with open(st.session_state.latest_screenshot, "rb") as file:
            btn = st.download_button(
                label="Download Latest Screenshot",
                data=file,
                file_name=f"pose_screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
