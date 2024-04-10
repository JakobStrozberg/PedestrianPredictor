
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
from collections import deque
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Lambda

# Only the required imports are listed here as the code implementation details for data processing and model building are not shown.


# Necessary custom function
@register_keras_serializable()
def slice_last_steps(inputs, prediction_length):
    return inputs[:, -prediction_length:, :]

# Load the scalers and the pre-trained transformer model
scaler_frames = load('scaler_frames.bin')
scaler_bboxes = load('scaler_bboxes.bin')
transformer_model = load_model('transformer_pedestrian_movement.keras', custom_objects={'slice_last_steps': slice_last_steps})

# Ensure that deque is initialized outside the function
previous_data_window = deque(maxlen=50)


def perform_transformer_prediction(frame_data):
    global previous_data_window

    keypoints = frame_data['keypoints'] #THIS REPRESENTS THE DATA COMING IN FROM THE COMPUTER VISION
    frame_number = frame_data['frame_number'] #THIS REPRESENTS THE DATA COMING IN FROM THE COMPUTER VISION
    bounding_box = frame_data['bounding_box'] #THIS REPRESENTS THE DATA COMING IN FROM THE COMPUTER VISION
    bounding_box_center = frame_data['bounding_box_center'] #THIS REPRESENTS THE DATA COMING IN FROM THE COMPUTER VISION

    # Normalize the frame number and bounding box center using scalers
    frame_number_normalized = scaler_frames.transform([[frame_number]])[0]
    bounding_box_normalized = scaler_bboxes.transform([bounding_box])[0]  # Scale full bounding box


    current_features = np.hstack((keypoints, frame_number_normalized, bounding_box_normalized))

    # Ensure you are adding a 1D array with 39 features to the deque
    previous_data_window.append(current_features.flatten())

    # Check if we have collected 50 frames
    if len(previous_data_window) == 50:
        #print("50 Reached")  # Debug print
        # Prepare the input to match the shape expected by the model
        input_sequence = np.array(previous_data_window)  # Shape should be (50, 39)
        input_sequence = input_sequence.reshape(1, -1, 39)  # Reshape to (1, 50, 39)
        #print(f"Input sequence shape before prediction: {input_sequence.shape}")

        # Perform prediction
        predicted_centers_sequence = transformer_model.predict(input_sequence)
        #print(f"Raw prediction output: {predicted_centers_sequence}") For debugging
        return predicted_centers_sequence[0].tolist()
    else:
        return None  # Not enough data to make a prediction