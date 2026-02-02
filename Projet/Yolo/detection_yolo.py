# Import the YOLO module from the ultralytics library.
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

# Create an instance of the YOLO model
# Initialize it with the pre-trained weights file 'best.pt' located at the specified path
model = YOLO('best.pt')



# Open the default webcam (0)(G)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
 
# Check if the webcam was opened successfully (G)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
else:
    print("Webcam accessed successfully!")


if(ret):
    # Run inference on the source
    results = model(frame, stream=True)

    # On récupère la première frame
    first_result = next(results)
    first_frame = first_result.plot()

    h, w, _ = first_frame.shape
    first_frame.shape = 640 , 480 , _
    h, w, _ = first_frame.shape

    

else:
    print("error in reading of webcam")




# Iterate over each 'result' in the 'results' collection
while(True):

    # Read the first frame to confirm capturing
    ret, frame = cap.read()
    nb_G = 0
    nb_D = 0
    
    if ret:
        
        # Run inference on the source
        results = model(frame, stream=True)
        result = next(results)

        # Extract bounding box information from the current 'result'
        boxes = result.boxes
        for box in boxes:
            box_size = box.xyxy
            size = box_size[0]
           
            x_start_tens = size[0]
            y_start_tens = size[1]
            x_end_tens = size[2]
            y_end_tens = size[3]
            
            x_start = x_start_tens.item()
            # print( "x start = " + str(x_start) )
            y_start =  y_start_tens.item()
            # print( "y start = " + str(y_start) )
            x_end =  x_end_tens.item()
            # print( "x end = " + str(x_end) )
            y_end = y_end_tens.item()
            #print( "y end = " + str(y_end) )

            box_mid = x_start + ( (x_end - x_start) /2 )

            if(box_mid < 320):
                nb_G += 1
            elif(box_mid > 320):
                nb_D +=1



        # Extract mask information from the current 'result'
        masks = result.masks
        # Extract keypoints information from the current 'result'
        keypoints = result.keypoints
        # Extract probability scores from the current 'result'
        probs = result.probs
        # Image annotée (numpy array BGR)
        # annotated_frame = result.plot()

        print("Nombre de personnes à gauche " + str(nb_G))
        print("Nombre de personnes à droite " + str(nb_D) )
        print("\n")

        frame = result.plot()
        cv2.imshow("Captured Frame", frame)
        cv2.waitKey(10)  # Wait for a key press to close

    else:
        print("Error: Could not capture a frame.")

cv2.destroyAllWindows()  # Close the window
    

# Now you can use 'boxes', 'masks', 'keypoints', and 'probs' as needed for further processing or analysis.
# These variables hold the relevant information related to the object detection results for the current iteration
