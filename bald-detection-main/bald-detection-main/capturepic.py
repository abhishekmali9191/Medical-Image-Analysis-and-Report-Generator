import cv2


def capture():
    # Create a VideoCapture object to capture from the default camera
    cap = cv2.VideoCapture(0)

    # Loop until the user presses the spacebar to capture an image
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Display the frame on the screen
        
        face_cascade = cv2.CascadeClassifier("frontalface.xml")

    # Detect faces in the image
        faces = face_cascade.detectMultiScale(
        frame, 
        scaleFactor=1.3, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x-50, y-50), (x+w+20,y+h+20), (0, 255, 0), 2)
        cv2.imshow("Camera", frame)

        # Wait for a key press
        key = cv2.waitKey(1)

        # Check if the spacebar key was pressed to capture the image
        if key == ord(' '):
            # Save the captured frame to a file
            cv2.imwrite("image.jpg", frame)

            # Break out of the loop to end the program
            break
    image = cv2.imread("E:\Bald detection\image.jpg")

    # Create a Haar cascade classifier
    face_cascade = cv2.CascadeClassifier("frontalface.xml")

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        image, 
        scaleFactor=1.3, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x-50, y-50), (x+w+20,y+h+20), (0, 255, 0), 2)
        crop = image[y-50:y+h+20, x-50:x+w+20]
        cv2.imshow('Image', crop)
        cv2.imwrite('croped.jpg',crop)

    # Display the resulting image
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

