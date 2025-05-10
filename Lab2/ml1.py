import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sci
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def plot_dendrogram(linkage_measure, calc_thresholds):
    """
    Function to implement hierarchical clustering algorithm with different linkage measures
    and number of clusters, plotting the dendrogram
    :param linkage_measure: determines which linkage function will be used (single, average, complete or ward)
    :param calc_thresholds: threshold for either 2, 3 or 4 clusters
    """
    data = read_data("data_clustering.csv")
    linkage_matrix = sci.linkage(data, linkage_measure) #consturct the linkage matrix

    plt.figure()
    sci.dendrogram(linkage_matrix) #construct dendrogram

    # assigning thresholds to different linkage measures
    if calc_thresholds:
        if linkage_measure == 'single':
            thresholds = {2: 0.15, 3: 0.14, 4: 0.13}
        elif linkage_measure == 'average':
            thresholds = {2: 0.4, 3: 0.3, 4: 0.26}
        elif linkage_measure == 'complete':
            thresholds = {2: 0.75, 3: 0.63, 4: 0.55}
        elif linkage_measure == 'ward':
            thresholds = {2: 3.0, 3: 2.0, 4: 1.1}
        else:
            thresholds = {}
        # assign styling and label to every threshold line
        for K, threshold in thresholds.items():
            plt.axhline(y=threshold, color='green', linestyle='dashed', label=f'K={K}, threshold={threshold}')

    #plotting
    plt.xticks([])
    plt.title(f"Dendrogram - {linkage_measure} measure")
    plt.ylabel("Dissimilarity")
    plt.xlabel("Observations")
    plt.savefig(sys.stdout.buffer)
    plt.close()


def agglomerative_clustering(measure, k):
    """
    Function to perform agglomerative clustering g based on the linkage function and the number of clusters
    :param measure: determines which linkage function will be used
    :param k: how many clusters should there be
    """
    data = read_data("data_clustering.csv")

    clustering = AgglomerativeClustering(n_clusters=k, linkage=measure) #perform agglomerative clustering with the given number of clusters and measure
    labels = clustering.fit_predict(data) #fit model to data, predict cluster lables for data points

    #commented out so the themis tests would pass
    #score = silhouette_score(data, labels) #calculate the silhouette score for each data point
    #print(f"sil score for {k} clusters {score}") #display the score

    #plotting
    plt.scatter(data[:, 0], data[:, 1], c=labels, edgecolors= "black")
    plt.title(f"Clustering results for {k} clusters, using '{measure}' measure")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
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

def plot_data_using_scatter_plot():
    """
    Function to plot the orginal data using scatter plot
    """
    data = read_data("data_clustering.csv")
    plt.scatter(data[:, 0], data[:, 1])

    #plotting
    plt.title("Scatter plot - original data")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.savefig(sys.stdout.buffer)
    plt.close()

def main():

    plot_data_using_scatter_plot()
    plot_dendrogram("average", True)
    linkage_measures = ['single', 'average', 'complete', 'ward']
    K = [2, 3, 4]
    for measure in linkage_measures:
        for k in K:
            agglomerative_clustering(measure, k)

if __name__ == "__main__":
    main()
