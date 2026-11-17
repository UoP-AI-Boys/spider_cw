import os
import matplotlib.pyplot as plt
import numpy as np
from spider_cw.plot import animate_gait

from spider_cw.new import (
    COXA_FORWARD_SIGN,
    FEMUR_UP_SIGN,
    JOINT_MAX,
    JOINT_MIN,
    N_FRAMES,
    N_JOINTS,
    WEIGHTS,
    coordination_fitness,
    decode,
    fitness_breakdown,
    lift_fitness,
    smoothness_fitness,
    stepping_fitness,
    train_ga,
)

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(-0.5, 0.5, size=(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs @ self.weights + self.biases)
        return self.output

    def backward(self, d_output, learning_rate):
        dz = d_output * (1.0 - self.output**2)

        d_weights = self.inputs.T @ dz
        d_biases = np.sum(dz, axis=0, keepdims=True)
        d_inputs = dz @ self.weights.T

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_inputs


class SimpleGaitNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.layer1 = DenseLayer(n_inputs, n_hidden)
        self.layer2 = DenseLayer(n_hidden, n_hidden)
        self.layer3 = DenseLayer(n_hidden, n_outputs)

    def forward(self, x):
        h1 = self.layer1.forward(x)
        h2 = self.layer2.forward(h1)
        return self.layer3.forward(h2)

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


def get_cyclic_inputs(n_frames=N_FRAMES):
    angles = 2.0 * np.pi * np.arange(n_frames) / n_frames
    return np.column_stack((np.sin(angles), np.cos(angles)))


def scale_to_unit(gait):
    center = 0.5 * (JOINT_MAX + JOINT_MIN)
    half_range = 0.5 * (JOINT_MAX - JOINT_MIN)
    return (gait - center) / half_range


def scale_to_joints(scaled_gait):
    center = 0.5 * (JOINT_MAX + JOINT_MIN)
    half_range = 0.5 * (JOINT_MAX - JOINT_MIN)
    gait = scaled_gait * half_range + center
    return np.clip(gait, JOINT_MIN, JOINT_MAX)


def evaluate_gait(gait):
    coxa = gait[:, 0::3] * COXA_FORWARD_SIGN
    femur = gait[:, 1::3] * FEMUR_UP_SIGN

    scores = {
        "stepping": stepping_fitness(coxa),
        "coordination": coordination_fitness(coxa),
        "lift": lift_fitness(coxa, femur),
        "smoothness": smoothness_fitness(gait),
    }
    scores["total"] = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    return scores


def plot_joint_trajectories(ga_gait, nn_gait, save_path="gaits/nn_joint_trajectories.png"):
    frames = np.arange(N_FRAMES)
    joint_names = ["Coxa (Swing)", "Femur (Lift)", "Tibia (Extension)"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Leg 1 Joint Trajectories: GA Keyframe Interpolation vs. Neural Network", fontsize=13)

    for j in range(3):
        axes[j].plot(frames, ga_gait[:, j], "--", label="GA Teacher (Linear Interp)", alpha=0.8)
        axes[j].plot(frames, nn_gait[:, j], "-", label="Neural Network Student", linewidth=2)
        axes[j].set_ylabel(f"{joint_names[j]}\n(rad)")
        axes[j].grid(True, alpha=0.3)
        if j == 0:
            axes[j].legend(loc="upper right")

    axes[-1].set_xlabel("Frame (0 - 299)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved trajectory plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs("gaits", exist_ok=True)

    best_chromosome, _ = train_ga()
    target_gait = decode(best_chromosome)
    ga_scores = fitness_breakdown(best_chromosome)

    X = get_cyclic_inputs(N_FRAMES)
    Y = scale_to_unit(target_gait)

    X_train, Y_train = X[0::2], Y[0::2]
    X_val, Y_val = X[1::2], Y[1::2]

    nn = SimpleGaitNetwork(n_inputs=2, n_hidden=48, n_outputs=N_JOINTS)
    nn.train(X_train, Y_train, epochs=3000, learning_rate=0.05)

    val_loss = np.mean((nn.forward(X_val) - Y_val) ** 2)
    print(f"\nValidation Loss: {val_loss:.6f}")

    nn_gait = scale_to_joints(nn.forward(X))
    nn_scores = evaluate_gait(nn_gait)

    print("\n--- Fitness Comparison ---")
    for key in ["stepping", "coordination", "lift", "smoothness", "total"]:
        print(
            f"{key.capitalize():14s} | GA: {ga_scores[key]:.4f} | NN: {nn_scores[key]:.4f}"
        )

    print("\nGenerating plots and animation...")
    plot_joint_trajectories(target_gait, nn_gait, save_path="gaits/nn_joint_trajectories.png")
    animate_gait(chromosome=nn_gait, title="Neural Network Spider Gait", save="gaits/nn_spider_gait.gif")
