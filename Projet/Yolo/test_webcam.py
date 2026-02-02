import cv2
 
# Open the default webcam (0)
cap = cv2.VideoCapture(0)
 
# Check if the webcam was opened successfully
if not cap.isOpened():
    print("Error: Could not access the webcam.")
else:
    print("Webcam accessed successfully!")
 
while True :
    # Read the first frame to confirm capturing
    ret, frame = cap.read()
    
    if ret:
        # Display the frame using imshow
        cv2.imshow("Captured Frame", frame)
        
    
    # Release the webcam
cv2.destroyAllWindows()  # Close the window
cap.release()