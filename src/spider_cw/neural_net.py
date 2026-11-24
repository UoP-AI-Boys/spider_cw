import os

# import matplotlib.pyplot as plt
import numpy as np

from spider_cw.new import (
    JOINT_MAX,
    JOINT_MIN,
    N_FRAMES,
    N_JOINTS,
    decode,
    fitness_breakdown,
    train_ga,
)
from spider_cw.plot import animate_gait

# ------ Activation Functions ----------

# Relu + Variants
def relu(x):
    return max(0, x)

def leaky_relu(x):
    return max(0.01 * x, x)

# Probablity Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)


class DenseLayer:
    """
    creates a single layer collection of neurons taking:
        inputs
        neurons
        and an activation function
    facillitating forward and backward traversal through the nueral net
    """
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = np.random.uniform(-0.5, 0.5, size=(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    """passes inputs through the layer"""
    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation(inputs @ self.weights + self.biases)
        return self.output

    """ """
    def backward(self, d_output, learning_rate):
        dz = d_output * (1.0 - self.output**2)

        d_weights = self.inputs.T @ dz
        d_biases = np.sum(dz, axis=0, keepdims=True)
        d_inputs = dz @ self.weights.T

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_inputs


class SimpleGaitNetwork:
    """
        creates a neural network of {layer_count} depth
    """
    def __init__(self, layers, activation_func, output_func=softmax):
        self.layers = []

        # layer_count = 4; range(layer_count) -> 0, 1, 2, 3
        for i in range(len(layers) - 1):
            prevSize = layers[i]
            currSize = layers[i + 1]

            self.layers.append(DenseLayer(prevSize, currSize, activation_func))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, x_train, y_train, epochs=3000, learning_rate=0.05):
        n_samples = len(x_train)
        train_losses = []

        for epoch in range(1, epochs + 1):
            y_pred = self.forward(x_train)

            error = y_pred - y_train
            loss = np.mean(error**2)
            train_losses.append(loss)

            loss_grad = (2.0 * error) / n_samples

            grad2 = self.layer3.backward(loss_grad, learning_rate)
            grad1 = self.layer2.backward(grad2, learning_rate)
            self.layer1.backward(grad1, learning_rate)

            if epoch % 500 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f}")

        return train_losses


def get_cyclic_inputs(n_frames=N_FRAMES): # Generate a position on a circle for each frame (each position the spider gait is in the walking cycle)
    angles = 2.0 * np.pi * np.arange(n_frames) / n_frames
    return np.column_stack((np.sin(angles), np.cos(angles))) # tuple output of sin, cos values correlating to a frames position on the circle


def scale_to_unit(gait):
    center = 0.5 * (JOINT_MAX + JOINT_MIN)
    half_range = 0.5 * (JOINT_MAX - JOINT_MIN)
    return (gait - center) / half_range


def scale_to_joints(scaled_gait):
    center = 0.5 * (JOINT_MAX + JOINT_MIN)
    half_range = 0.5 * (JOINT_MAX - JOINT_MIN)
    gait = scaled_gait * half_range + center
    return np.clip(gait, JOINT_MIN, JOINT_MAX)

def main():
    from spider_cw.visualization import (
        plot_foot_paths,
        plot_footfall,
        plot_gait,
        plot_joint_trajectories,
    )

    np.random.seed(42)
    os.makedirs("gaits", exist_ok=True)

    best_chromosome, _ = train_ga()
    target_gait = decode(best_chromosome)
    ga_scores = fitness_breakdown(best_chromosome)

    X = get_cyclic_inputs(N_FRAMES)
    Y = scale_to_unit(target_gait)

    X_train, Y_train = X[0::2], Y[0::2]
    X_val, Y_val = X[1::2], Y[1::2]

    nn = SimpleGaitNetwork(layers=[2, 48, 48, 24], activation_func=tanh)
    nn.train(X_train, Y_train, epochs=3000, learning_rate=0.05)

    val_loss = np.mean((nn.forward(X_val) - Y_val) ** 2)
    print(f"\nValidation Loss: {val_loss:.6f}")

    nn_gait = scale_to_joints(nn.forward(X))
    nn_scores = fitness_breakdown(nn_gait)

    print("\n--- Fitness Comparison ---")
    for key in ["stepping", "coordination", "lift", "smoothness", "total"]:
        print(
            f"{key.capitalize():14s} | GA: {ga_scores[key]:.4f} | NN: {nn_scores[key]:.4f}"
        )

    print("\nGenerating plots and animation...")
    animate_gait(chromosome=nn_gait, title="Neural Network Spider Gait", save="gaits/nn_spider_gait.gif")
    plot_joint_trajectories(target_gait, nn_gait)
    plot_gait(nn_gait)
    plot_footfall(nn_gait)
    plot_foot_paths(nn_gait)


if __name__ == "__main__":
    main()
