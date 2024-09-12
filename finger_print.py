import cv2
import os
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize SIFT feature extractor
sift = cv2.xfeatures2d.SIFT_create()

def calculate_hand_bbox(hand_landmarks, frame_shape):
    min_x, max_x = frame_shape[1], 0
    min_y, max_y = frame_shape[0], 0

    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0])
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    return min_x, min_y, max_x, max_y


while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Convert frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand bounding box
            bbox = calculate_hand_bbox(hand_landmarks, frame.shape)

            # Extract hand region from the frame
            hand_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Convert hand region to grayscale
            gray_hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and compute descriptors for the hand region
            keypoints_1, descriptors_1 = sift.detectAndCompute(gray_hand_region, None)

            # Iterate through the database images
            for file in os.listdir("database"):
                # Load the database fingerprint image
                database_image_path = os.path.join("database", file)
                fingerprint_database_image = cv2.imread(database_image_path)

                # Convert database image to grayscale
                gray_database = cv2.cvtColor(fingerprint_database_image, cv2.COLOR_BGR2GRAY)

                # Detect keypoints and compute descriptors for the database image
                keypoints_2, descriptors_2 = sift.detectAndCompute(gray_database, None)

                # Initialize FLANN-based matcher
                flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict())

                # Perform keypoint matching
                matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

                # Apply Lowe's ratio test to filter good matches
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

                        # Calculate match percentage
                        match_percentage = len(good_matches) / len(keypoints_1) * 100

                        # Display match percentage
                        cv2.putText(frame, f"Match: {match_percentage:.2f}%", (bbox[0] + 20, bbox[1] + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Overwrite the area where the text was drawn with a black rectangle
    frame[0:70, 0:frame.shape[1]] = (0, 0, 0)

    # Display frame with hand detection and fingerprint recognition
    cv2.imshow("Hand and Fingerprint Recognition", frame)

    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

