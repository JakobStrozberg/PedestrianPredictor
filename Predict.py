import cv2
import numpy as np
from collections import deque
from External import perform_transformer_prediction  # FOR ILLUSTRATIVE PURPOSES
from Kalman import perform_kalman_filtering # Replace with the correct module name

# Assuming the import path for YOLO is correct
from ultralytics import YOLO  

# Load a pretrained YOLOv8 model with pose estimation
model = YOLO('yolov8m-pose.pt')

# Define the path to the video file
video_source = '/Users/jakobstrozberg/Desktop/TRAIN20fps.mp4'  # Update to the correct path
# video_source = 0 

# Initialize video capture
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_number = 0
previous_centers = deque(maxlen=50)  # Store centers up to the last 50 frames

# Define colors and radius for drawing
prediction_color = (0, 0, 205)  # Green for future predictions
prediction_radius = 3
current_center_color = (0, 0, 255)  # Red for current bounding box center
previous_center_color = (0, 255, 255)  # Yellow for historical centers
previous_center_radius = 2

PredictionfromTransformer = None
future_predictions = None  

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model(frame, show=True, max_det=1, classes=0)

        # Check that results have the attributes you expect
        for res in results:
            if hasattr(res, 'keypoints') and hasattr(res, 'boxes'):
                keypoints = res.keypoints.xy.numpy().flatten() if res.keypoints.xy is not None and res.keypoints.xy.numel() > 0 else np.array([])
                boxes = res.boxes.xyxy[0].numpy() if len(res.boxes.xyxy) > 0 else None

                if keypoints.size > 0 and boxes is not None:
                    x1, y1, x2, y2 = boxes[:4]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    previous_centers.append((cx, cy))  # Append the current center to the deque

                    # Draw a dot on the current bounding box center
                    cv2.circle(frame, (int(cx), int(cy)), radius=5, color=current_center_color, thickness=-1)
                    bounding_box_center = np.array([cx, cy])

                    frame_data = {
                        'keypoints': keypoints,
                        'frame_number': frame_number,
                        'bounding_box': boxes,
                        'bounding_box_center': bounding_box_center,
                    }
                    #print(frame_data)
                    print(keypoints)
                    print(boxes)
                    print(bounding_box_center)
                    print(frame_number)


                    PredictionfromTransformer = perform_transformer_prediction(frame_data) #Should be sending in 50 predictions, FOR ILLUSTRATIVE PURPOSES ONLY
                    print(f"Prediction from Transformer: {PredictionfromTransformer}") #Returns predictions 25 frames into the future, FOR ILLUSTRATIVE PURPOSES ONLY


                    
                else:
                    print(f"No adequate detections in frame {frame_number}")
            else:
                print(f"No detections in frame {frame_number}")



        
        if PredictionfromTransformer is not None:
            transformer_prediction_color = (255, 20, 147)  # DeepPink color for transformer predictions
            transformer_prediction_radius = 1
            # Iterate through the prediction list
            #for pred in PredictionfromTransformer:
                #pred_x, pred_y = pred
                # Draw transformer predictions
                #cv2.circle(frame, (int(pred_x), int(pred_y)), radius=transformer_prediction_radius, color=transformer_prediction_color, thickness=-1) Commenting out transformer predictions

        # Add visualization of historical centers
        #for previous_center in previous_centers:
            #cv2.circle(frame, (int(previous_center[0]), int(previous_center[1])), radius=previous_center_radius, color=previous_center_color, thickness=-1) Commenting out printing previous centers.


        # Perform prediction and draw future trajectory
        if len(previous_centers) >= 50:
            motion_data = [(frame_number - 50 + i,) + center for i, center in enumerate(previous_centers)]
            future_predictions = perform_kalman_filtering(motion_data, num_predictions=25)
            
            # Draw future predictions on the frame ,COMMENTING OUT KALMAN FILTER PREDICTIONS
            #for prediction in future_predictions:
            #    pred_x, pred_y = prediction
            #    cv2.circle(frame, (int(pred_x), int(pred_y)), radius=prediction_radius, color=prediction_color, thickness=-1)


        transformer_weight = 0.5
        kalman_weight = 0.5
        number_of_future_points = 25 

        if future_predictions is not None:
            future_predictions = future_predictions[-number_of_future_points:]

        # Ensure both predictions are available
        if PredictionfromTransformer is not None and future_predictions is not None:
            combined_predictions = []
            for transformer_pred, kalman_pred in zip(PredictionfromTransformer, future_predictions):
                # Compute weighted average of the predictions
                combined_pred_x = transformer_weight * transformer_pred[0] + kalman_weight * kalman_pred[0]
                combined_pred_y = transformer_weight * transformer_pred[1] + kalman_weight * kalman_pred[1]
                combined_predictions.append((combined_pred_x, combined_pred_y))

                # Draw combined predictions
                cv2.circle(frame, (int(combined_pred_x), int(combined_pred_y)), radius=prediction_radius, color=(255, 255, 0), thickness=-1)  # Cyan for combined predictions

                #ADDING HEAT MAP.
                # Assuming combined_predictions is a list of (x, y) tuples
                heatmap_img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

                # Increase the value around each prediction point
                for (x, y) in combined_predictions:
                    cv2.circle(heatmap_img, (int(x), int(y)), radius=20, color=(1), thickness=-1)  # Adjust radius and color as needed

                # Normalize the heatmap image to ensure values are between 0 and 255
                heatmap_img = cv2.normalize(heatmap_img, None, 0, 255, cv2.NORM_MINMAX)

                # Apply a colormap
                heatmap_colored = cv2.applyColorMap(heatmap_img.astype(np.uint8), cv2.COLORMAP_JET)

                # Overlay the heatmap on the original frame
                # Adjust alpha (transparency) to your preference
                alpha = 0.4
                overlayed_img = cv2.addWeighted(frame, alpha, heatmap_colored, 1 - alpha, 0)

                # Show the frame with the heatmap overlay
                cv2.imshow('Detection and Prediction Overlay with Heatmap', overlayed_img)


        
        # Show the frame with bounding box center and future predictions
        cv2.imshow('Detection and Prediction Overlay', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1



finally:
    cap.release()
    cv2.destroyAllWindows()

print('Finished processing and printing frame data.')