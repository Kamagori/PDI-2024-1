import cv2
import numpy as np

# Access the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a circle animation closing towards the detected face
    for (x, y, w, h) in faces:
        # Calculate the center of the face
        face_center = (x + w // 2, y + h // 2)

        # Calculate the maximum radius from the corners of the frame to the face center
        max_radius = max(face_center[0], frame.shape[1] - face_center[0],
                         face_center[1], frame.shape[0] - face_center[1])

        # Draw the circle animation
        for radius in range(max_radius, 0, -5):
            # Create a black mask with the same dimensions as the frame
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            # Draw a white circle on the mask
            cv2.circle(mask, face_center, radius, 255, -1)

            # Apply the mask to the frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Display the resulting frame
            cv2.imshow('Circle Animation', masked_frame)

            # Check if the circle is close to the face
            if radius < min(w, h) / 2:
                break  # Break out of the animation loop if the circle is close enough to the face

            # Wait for a short duration (you can adjust this value for the animation speed)
            cv2.waitKey(30)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
