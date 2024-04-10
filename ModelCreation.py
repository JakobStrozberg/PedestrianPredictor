import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, MultiHeadAttention, Dropout, TimeDistributed
from joblib import dump, load
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import register_keras_serializable



# Load data from a .npy file
data = np.load('FILLED_TRAININGDATASET.npy', allow_pickle=True)

# Extract elements from the data
posture_key_points = np.array([item['keypoints'] for item in data])
frame_numbers = np.asarray([item['frame_number'] for item in data], dtype=np.float64)
bounding_box_locations = np.asarray([item['bounding_box'] for item in data], dtype=np.float64)
bounding_box_centers = np.array([item['bounding_box_center'] for item in data], dtype=np.float64)

# Sort the data by frame number to maintain chronological order
sorted_indices = np.argsort(frame_numbers)
posture_key_points = posture_key_points[sorted_indices]
frame_numbers = frame_numbers[sorted_indices]
bounding_box_locations = bounding_box_locations[sorted_indices]
bounding_box_centers = bounding_box_centers[sorted_indices]

# Normalize the frame numbers and bounding box locations
scaler_frames = MinMaxScaler()
frame_numbers_normalized = scaler_frames.fit_transform(frame_numbers.reshape(-1, 1)).flatten()
dump(scaler_frames, 'scaler_frames.bin')

scaler_bboxes = MinMaxScaler()
bounding_box_locations_normalized = scaler_bboxes.fit_transform(bounding_box_locations)
dump(scaler_bboxes, 'scaler_bboxes.bin')

# Concatenate all features together
features = np.hstack((posture_key_points, frame_numbers_normalized[:, None], bounding_box_locations_normalized))




def create_sequences(features, targets, sequence_length, prediction_distance, prediction_length):
    input_sequences = []
    future_centers_sequence = []

    for i in range(len(features) - sequence_length - prediction_distance - prediction_length + 1):
        input_seq = features[i:i + sequence_length]
        future_center_seq = targets[i + sequence_length + prediction_distance - 1:i + sequence_length + prediction_distance + prediction_length - 1]
        input_sequences.append(input_seq)
        future_centers_sequence.append(future_center_seq)

    return np.array(input_sequences), np.array(future_centers_sequence)

# Create sequences with 25 predictions
sequence_length = 50
prediction_distance = 25
prediction_length = 25
input_sequences, future_centers_sequence = create_sequences(
    features, bounding_box_centers, sequence_length, prediction_distance, prediction_length
)

# Verify the shapes
print("Input sequences shape:", input_sequences.shape)
print("Future centers sequences shape:", future_centers_sequence.shape)

# Split the data into 80% training and 20% testing
input_train, input_test, future_train, future_test = train_test_split(
    input_sequences, future_centers_sequence, test_size=0.2, random_state=42
)

def build_transformer(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, prediction_length=25):
    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(num_transformer_blocks):
        x = LayerNormalization(epsilon=1e-6)(x)
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=dropout
        )(x, x)
        x = Dropout(dropout)(tf.keras.layers.Add()([x, attention_output]))
        x = Dropout(dropout)(Dense(ff_dim, activation="relu")(x))

    # To ensure that the model outputs sequences of 25 steps (prediction_length),
    # we take the last `prediction_length` steps from the output sequence.
    x = TimeDistributed(Dense(mlp_units[0], activation="relu"))(x)
    x = Dropout(dropout)(x)
    x = TimeDistributed(Dense(2, activation="linear"))(x)


    # Define a custom slicing function

    @register_keras_serializable()
    def slice_last_steps(inputs, prediction_length):
        return inputs[:, -prediction_length:, :]

    # Apply this function as a Lambda layer
    outputs = Lambda(slice_last_steps, arguments={'prediction_length': prediction_length})(x)
    

    return Model(inputs, outputs)

# Model Hyperparameters
input_shape = input_train.shape[1:]  # Including the sequence length


# Experiment Configuration
head_size = 256 #FINE TUNEING AFTER num_heads
num_heads = 4 # Should divide the head_size evenly for optimal performance
ff_dim = 200 
num_transformer_blocks = 2 #<2 is fire
mlp_units = [256] # Responsible for final prediction
dropout = 0.0005
def scheduler(epoch, lr):
    if epoch < 100:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.0125).numpy()) # <-0.1



lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Instantiate and compile the model
transformer = build_transformer(
    input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, prediction_length
)
transformer.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Display the model structure
transformer.summary()


# Define the Keras TensorBoard callback.
import tensorboard



#USING A LEARNING RATE SCHEDULER 
history = transformer.fit(
    input_train, future_train,
    validation_split=0.2,
    epochs=300,
    batch_size=256,
    callbacks=[lr_scheduler]  # Pass the learning rate scheduler here
)

# Evaluate the model on the test set
test_loss = transformer.evaluate(input_test, future_test)
print(f"Test loss: {test_loss}")

# Save the model if desired
#transformer.save('transformer_pedestrian_movement.keras')
#transformer.save_weights('transformer_pedestrian_movement.weights.h5')
# Save the model if desired
transformer.save('transformer_pedestrian_movement.keras')

# Save the architecture
#with open('transformer_pedestrian_movement_architecture.json', 'w') as f:
#    f.write(transformer.to_json())
# Save the weights

# Perform inference on the test set
predicted_centers_sequence = transformer.predict(input_test)

# To compute mean squared error across all predicted steps, you need to average over both batches and sequence length
mse = np.mean(np.square(future_test - predicted_centers_sequence))
print(f"Mean Squared Error on Test Data: {mse}")

# Optionally, visualize a few predictions vs actual centers
num_samples_to_visualize = 5
samples_indices = np.random.choice(len(input_test), num_samples_to_visualize, replace=False)

for idx in samples_indices:
    # Retrieve the actual and predicted future bounding box centers sequence for the current index
    actual_sequence = future_test[idx]
    predicted_sequence = predicted_centers_sequence[idx].squeeze()  # Assuming model returns with batch dim

    # Print the shapes and the sequences to inspect
    print("Actual sequence:\n", actual_sequence)
    print("Predicted sequence:\n", predicted_sequence.squeeze())  # Squeeze removes any single-dimensional entries

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(actual_sequence[:, 0], actual_sequence[:, 1], 'ro-', label='Actual Center Sequence')
    plt.plot(predicted_sequence[:, 0], predicted_sequence[:, 1], 'bo-', label='Predicted Center Sequence')
    plt.legend()
    plt.title(f"Comparison of Actual vs Predicted Center Sequences for Sample {idx}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()