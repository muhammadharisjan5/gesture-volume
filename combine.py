from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock

import cv2
import mediapipe as mp

class HandFingerprintRecognitionApp(App):
    def build(self):
        # Initialize MediaPipe Hands model
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands()

        # Create UI
        layout = BoxLayout(orientation='vertical')
        
        self.camera_feed = Image()

        layout.add_widget(self.camera_feed)

        return layout
    
    def calculate_hand_bbox(self, hand_landmarks, frame_shape):
        min_x, max_x = frame_shape[1], 0
        min_y, max_y = frame_shape[0], 0

        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        return min_x, min_y, max_x, max_y


    def update_camera(self, *args):
        # Capture frame from webcam
        ret, frame = self.cap.read()

        # Convert frame to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = self.hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand bounding box
                bbox = self.calculate_hand_bbox(hand_landmarks, frame.shape)

                # Draw hand bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Display frame with hand detection
        self.camera_feed.texture = cv2.flip(frame, 0)

    def on_start(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Update camera feed
        self.update_camera_event = Clock.schedule_interval(self.update_camera, 1.0 / 30.0)

    def on_stop(self):
        # Release webcam
        self.cap.release()
        self.update_camera_event.cancel()

if __name__ == '__main__':
    HandFingerprintRecognitionApp().run()
