import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


def squared_euclidean_distance(datapoint, prototype):
    """
    Method to calculate squared eucleidian distance
    :param datapoint: given datapoint
    :param prototype: given prototype
    :return: squared eucleidian distace between them
    """
    return np.sum((datapoint - prototype) ** 2)

def calculate_hvq(data, prototypes):
    """
   Method to calculate quantization error for a dataset
   :param data: input dataset
   :param prototypes: prototype positions
   :return: average quantization error
   """
    error = 0
    for datapoint in data:
        distance = np.array([squared_euclidean_distance(datapoint, proto) for proto in prototypes]) #calculate distances
        min_distance = min(distance) #determine minimal distance
        error += min_distance
    return error #return error of the epoch

def vector_quantization(k, learning_rate, max_epoch):
    """
    Method to implement Winner-Takes-All unsupervised competitive learning
    :param k: Number of prototypes
    :param learning_rate: step size
    :param max_epoch: maximum number of epochs (sweeps through the dataset)
    :return: trace of prototype positions and quantization errors over epochs
    """
    data = read_data("simplevqdata.csv") #load data
    P = data.shape[0] #number of data points
    HVQ_trace = [] #array to store quantization errors

    prototypes = data[np.random.choice(P, k)] #initialize prototypes by selecting random k datapoints

    #sorted_data = data[np.argsort(np.linalg.norm(data, axis=1))]
    #prototypes = sorted_data[-k:] stupid initialization, picking the furthest data points

    prototype_trace = [prototypes.copy()] #store initial prototypes
    for epoch in range(1, max_epoch + 1): #looping through each epoch
        for datapoint in data[np.random.permutation(P)]: #looping through each datapoint in random data order
            distance = np.array([squared_euclidean_distance(datapoint, proto) for proto in prototypes]) #calculate distances
            winner_idx = np.argmin(distance) #the closest prototype becomes the winner
            prototypes[winner_idx] += learning_rate * (datapoint - prototypes[winner_idx]) #update winning prototype

        prototype_trace.append(prototypes.copy()) #store prototype positions for current epoch

        current_HVQ = calculate_hvq(data, prototypes) #evaluate quantization error
        HVQ_trace.append(current_HVQ) #store error for current epoch

    return prototype_trace, HVQ_trace #return prototype positions and errors

def plot_vq(k, learning_rate, max_epoch):
    """
   Method to plot the dataset with prototype trajectories
   :param k: Number of prototypes
   :param learning_rate: step size
   :param max_epoch: maximum number of epochs (sweeps through the dataset)
   """
    data = read_data("simplevqdata.csv")
    colors = ['red', 'blue', 'yellow', 'green'] #prototype colors

    prototype_trace, _ = vector_quantization(k, learning_rate, max_epoch) #train vector quantization to get prototype trajectories

    plt.scatter(data[:, 0], data[:, 1], edgecolors='k')#plot dataset

    for i in range(len(prototype_trace[0])):
        trajectory = np.array([epoch[i] for epoch in prototype_trace]) #get trajectory
        plt.scatter(trajectory[:, 0], trajectory[:, 1], color=colors[i]) #plot the prototypes
        plt.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], linestyle='-', marker='o') #plot the trajectory

    #plotting
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Trajectory Of Prototypes')
    plt.savefig(sys.stdout.buffer)
    plt.close()


def plot_vq_error(HVQerror_k, max_epoch):
    """
    Method to plot quantization error over epochs
    :param HVQerror_k: quantization error list
    :param max_epoch: number of epochs
    :return:
    """
    plt.plot(range(max_epoch), HVQerror_k) #plotting error over epochs
    plt.xlabel("Epoch")
    plt.ylabel("Quantization Error")
    plt.title("Quantization Error Over Epochs")
    plt.savefig(sys.stdout.buffer)
    plt.close()


def read_data(filename):
    """
    Function to read data
    :param filename: given filename of the data file
    :return: data that the file contains
    """
    datadf = pd.read_csv(filename, sep=",", header=None)
    return datadf.values

def main():
    np.random.seed(1740844038)
    np.set_printoptions(precision=5, suppress=True)

    max_epoch = 100
    learning_rates = [0.1, 0.05, 0.01]
    for rate in learning_rates:
        for k in [2, 4]:
            prototype_trace, HVQ_trace = vector_quantization(k, rate, max_epoch)
            plot_vq(k, rate, max_epoch)
            plot_vq_error(HVQ_trace, max_epoch)


if __name__ == "__main__":
    main()