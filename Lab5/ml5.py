import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def squared_euclidean_distance(datapoint, prototype):
    return np.sum((datapoint - prototype) ** 2)


def linear_vector_quantization(num_prototypes, learning_rate, max_epoch):
    data = pd.read_csv("data_lvq.csv", header=None).values
    labels = np.array([1] * 50 + [2] * 50)
    P = data.shape[0]

    prototypes = []
    prototype_labels = []
    for class_label in [1, 2]:
        class_indices = np.where(labels == class_label)[0]
        selected_indices = np.random.choice(class_indices, num_prototypes, replace=False)
        prototypes.extend(data[selected_indices])
        prototype_labels.extend([class_label] * num_prototypes)
    prototypes = np.array(prototypes)
    prototype_labels = np.array(prototype_labels)

    prototype_trace = [prototypes.copy()]
    error_rates = []

    for epoch in range(max_epoch):
        indices = np.random.permutation(P)
        data_shuffled = data[indices]
        labels_shuffled = labels[indices]

        for datapoint, label in zip(data_shuffled, labels_shuffled):
            distances = np.array([squared_euclidean_distance(datapoint, proto) for proto in prototypes])
            winner_idx = np.argmin(distances)
            winner_label = prototype_labels[winner_idx]

            if winner_label == label:
                prototypes[winner_idx] += learning_rate * (datapoint - prototypes[winner_idx])
            else:
                prototypes[winner_idx] -= learning_rate * (datapoint - prototypes[winner_idx])

        prototype_trace.append(prototypes.copy())
        prototype_trace_array = np.stack(prototype_trace)

        predicted_labels = []
        for datapoint in data:
            distances = np.array([squared_euclidean_distance(datapoint, proto) for proto in prototypes])
            winner_idx = np.argmin(distances)
            predicted_labels.append(prototype_labels[winner_idx])
        predicted_labels = np.array(predicted_labels)
        error_rate = np.sum(predicted_labels != labels)
        error_rates.append(error_rate)

    return prototype_trace_array, predicted_labels, np.array(error_rates)


def plot_trajectory(num_prototypes, learning_rate, max_epoch):
    prototype_trace, predicted_labels, _ = linear_vector_quantization(num_prototypes, learning_rate, max_epoch)

    data = pd.read_csv("data_lvq.csv", header=None).values

    plt.figure()
    plt.scatter(data[predicted_labels == 1, 0], data[predicted_labels == 1, 1], color='red')
    plt.scatter(data[predicted_labels == 2, 0], data[predicted_labels == 2, 1], color='blue')

    for i in range(num_prototypes * 2):
        trajectory = np.array([epoch[i] for epoch in prototype_trace])
        color = 'red' if i < num_prototypes else 'blue'
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, marker='*', s=200)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Trajectory Of Prototypes')
    plt.savefig(sys.stdout.buffer)
    plt.close()


def plot_error_rate(num_prototypes, learning_rate, max_epoch):
    _, _, error_rates = linear_vector_quantization(num_prototypes, learning_rate, max_epoch)
    error_rates = [(rate / 100) for rate in error_rates]
    plt.figure()
    plt.plot(range(max_epoch), error_rates)
    plt.xlabel('Epoch')
    plt.ylabel('The error rate in %')
    plt.title('Learning curve')
    plt.savefig(sys.stdout.buffer)
    plt.close()


def plot_data():
    data = pd.read_csv("data_lvq.csv", header=None).values
    labels = np.array([1] * 50 + [2] * 50)

    plt.figure()
    plt.scatter(data[labels == 1, 0], data[labels == 1, 1], color='red')
    plt.scatter(data[labels == 2, 0], data[labels == 2, 1], color='blue')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Trajectory Of Prototypes')
    plt.savefig(sys.stdout.buffer)
    plt.close()


def main():
    np.random.seed(1740844038)
    np.set_printoptions(precision=5, suppress=True, threshold=sys.maxsize)

    K1 = 1
    eta = 0.002
    t_max = 200

    plot_data()

    plot_error_rate(num_prototypes=K1, learning_rate=eta, max_epoch=t_max)
    plot_trajectory(num_prototypes=K1, learning_rate=eta, max_epoch=t_max)

    K2 = 2

    plot_error_rate(num_prototypes=K2, learning_rate=eta, max_epoch=t_max)
    plot_trajectory(num_prototypes=K2, learning_rate=eta, max_epoch=t_max)

if __name__ == "__main__":
    main()