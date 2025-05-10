import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.io import loadmat
import matplotlib.lines as mlines


data = loadmat('COIL20.mat')
x_coords = data['X']
y_coords = data['Y']

def pca(x, d):
    """
    Perform Principal Component Analysis (PCA) on the dataset X.
    :param x: The input data matrix (each column is a data point).
    :param d: The number of principal components to retain.
    :return: U_d : The first d principal components.
    eigenvalues: The eigenvalues corresponding to the principal components.
    Z_d: The reduced dataset.
    """

    # Standardize the dataset
    mean_X = np.mean(x, axis=0)
    std_X = np.std(x, axis=0)
    Z = (x - mean_X) / std_X

    # Compute the covariance matrix
    covariance_matrix = np.cov(Z, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    #  Select the first d principal components
    U_d = eigenvectors[:, :d]

    # Reduce the dimensionality of the data
    Z_d = np.dot(Z, U_d)

    # Return the first d principal components, the eigenvalues and the reduced dataset
    return U_d, eigenvalues, Z_d

def eigen_value_profile(x, d):
    """
    Plot the eigenvalue profile to visualize variance distribution

    :param x: input data matrix
    :param d: the number of principal components to consider
    """

    # Extract the eigenvalue from PCA
    _, eigenvalues, _ = pca(x, d)

    # Plot the figure
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, color='purple')
    plt.xlabel('Index eigen-value')
    plt.ylabel('Eigen-value')
    plt.title('Eigen-value Profile of the Dataset')
    plt.savefig(sys.stdout.buffer, format='png')


def dimension_reduced_data(x, y, d, perplexity, random_state):
    """
    Perform PCA and t-SNE to reduce dimensionality and plot t-SNE figure.
    :param x: input data matrix
    :param y: labels for each data point.
    :param d: dimensionality
    :param perplexity: perplexity parameter for t-SNE
    :param random_state: random seed for reproducibility
    """
    # Apply PCA to reduce dimensionality
    _, _, Z_d = pca(x, d)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced_data = tsne.fit_transform(Z_d)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(20)]  # Create a list of 20 distinct colors

    # Plot figure
    plt.figure(figsize=(6.4, 4.8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y.flatten(), cmap=cmap, s=15)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE visualization of dimension reduced data')

    # Handles for legend
    unique_labels = np.unique(y)
    handles_legend = [
        mlines.Line2D([], [], marker='o', linestyle='None', markersize=4, color=colors[i], label=f'Class {label}')
        for i, label in enumerate(unique_labels)]

    # Create legend and figure
    plt.legend(handles=handles_legend, title="Object ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(sys.stdout.buffer)

def input_data_sample():
    """
    Display a data sample from the input data
    """
    # Load the data
    data = loadmat('COIL20.mat')
    X = data['X']
    # Reshape the first image
    image = np.reshape(X[0, :], (32, 32))

    # Plot the image
    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(image)
    plt.title(f"Input data sample as an image")
    plt.savefig(sys.stdout.buffer)

def calculate_d_for_variance(eigenvalues, variance_thresholds):
    """
    Determine the minimum number of dimensions needed to retain a given percentage of variance
    :param eigenvalues: sorted eigenvalues from PCA
    :param variance_thresholds: list of desired variance thresholds
    :return: minimum number of dimensions needed to reach each variance threshold
    """
    # Calculate cumulative variance
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    # Find number of dimensions needed for each threshold
    d_values = [np.argmax(cumulative_variance >= threshold) + 1 for threshold in variance_thresholds]
    return d_values

def main():

    # Load the dataset
    data = loadmat('COIL20.mat')
    x_coords = data['X']

    # Set the desired dimensionality
    d = 40

    # Perform PCA
    U_d, eigenvalues, Z_d = pca(x_coords, d)
    print(pca(x_coords, d))
    # Compute cumulative variance for d selection
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Calculate the dimensionality d for 0.9, 0.95, and 0.98 variance
    variance_thresholds = [0.9, 0.95, 0.98]
    d_values = calculate_d_for_variance(eigenvalues, variance_thresholds)

    # Print the table
    print("Variance Threshold | Dimensionality (d)")
    print("------------------|-------------------")
    for threshold, d_val in zip(variance_thresholds, d_values):
        print(f"{threshold:.2f}               | {d_val}")

    # Plot the eigenvalue profile
    eigen_value_profile(x_coords, d)

    # Plot the reduced data using t-SNE
    dimension_reduced_data(x_coords, y_coords, d, 4, 42)

    # Plot data sample image
    input_data_sample()

if __name__ == "__main__":
    main()