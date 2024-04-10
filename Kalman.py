import numpy as np
from numpy.linalg import inv

def fill_diag(diag):
    return np.diag(diag)

def model_CV(X0, Ts=1):
    A = np.array([[1, 0, Ts, 0], [0, 1, 0, Ts], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.zeros((4, 1))
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    D = np.zeros((2, 1))
    return [X0, A, B, C, D]

class KalmanFilter:
    def __init__(self, state_space, P, Q, R):
        self.X = state_space[0]
        self.A, self.B, self.C, self.D = state_space[1:]
        self.P = P
        self.Q = Q
        self.R = R

    def predict(self, U):
        self.X = np.dot(self.A, self.X) + np.dot(self.B, U)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.X

    def update(self, U, Y):
        Yh = np.dot(self.C, self.X) + np.dot(self.D, U)
        S = self.R + np.dot(np.dot(self.C, self.P), self.C.T)
        K = np.dot(np.dot(self.P, self.C.T), inv(S))
        self.X += np.dot(K, (Y - Yh))
        self.P -= np.dot(np.dot(K, self.C), self.P)

    def step(self, U, Y=None):
        if Y is not None:
            self.predict(U)
            self.update(U, Y)
        else:
            self.predict(U)
        return self.X.copy()
    

    def multiple_steps(self, U_arr, Y_arr=None, num_predictions=0):
        predictions = []
        # Update and predict based on the measurements
        for idx in range(len(Y_arr)):
            U = U_arr[idx]
            Y = Y_arr[idx]
            self.predict(U)
            self.update(U, Y)
            predictions.append(self.X[:2].flatten().tolist()) # Append each prediction made after update

        # Predict future steps without a corresponding measurement
        for _ in range(num_predictions):
            U = U_arr[-1]  # Use the last control input for prediction, if control input is a constant or can be assumed the same for future predictions
            self.predict(U)
            predictions.append(self.X[:2].flatten().tolist())

        return predictions



def perform_kalman_filtering(motion_data, num_predictions=25):
    if not motion_data:
        return []

    Ts = 1
    X0 = np.array([motion_data[0][1], motion_data[0][2], 0, 0]).reshape(4, 1)
    P0 = fill_diag([1, 1, 1, 1])
    Q = 0.001 * fill_diag([1, 1, 1, 1])  # Process noise covariance, may require tuning
    R = 1 * np.eye(2)  # Measurement noise covariance, may require tuning

    kf = KalmanFilter(model_CV(X0, Ts), P0, Q, R)

    U_arr = np.zeros((len(motion_data) + num_predictions, 1, 1))  # No control input
    Y_arr = np.array([[x, y] for _, x, y in motion_data]).reshape(-1, 2, 1)

    predictions = kf.multiple_steps(U_arr, Y_arr, num_predictions)

    return predictions