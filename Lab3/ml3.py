import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import numpy as np



def DBSCAN(D, eps, MinPts):
    """
    Applies DBSCAN algorithm to dataset D with chosen eps and MinPts
    :param D: Dataset
    :param eps: maximum distance between points to be considered in the same cluster
    :param MinPts: minimum number of points to form a dense region
    :return: labels array showing the cluster assignments for each point
    """
    labels = [0] * len(D) #mark all points as uvisited
    C = 0 #cluster index

    for P in range(len(D)): #loop through all the points in the dataset
        if labels[P] != 0: #if a point is visited, skip it
            continue

        NeighborPts = regionQuery(D, P, eps) #find neighbours of the point

        if len(NeighborPts) < MinPts: #if there are fewer neighbours than MinPts, label it as noise
            labels[P] = -1
        else:
            C += 1 #start a new cluster
            expandCluster(D, labels, P, NeighborPts, C, eps, MinPts)

    return np.array(labels)

def expandCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    """
    Expand the current cluster by recursively adding reachable points
    :param D: dataset
    :param labels: list of cluster labels for all points
    :param P: index of xurrent point added to the cluster
    :param NeighborPts: point P list of neighbours
    :param C: current cluster index
    :param eps: maximum distance between points to be considered in the same cluster
    :param MinPts: minimum number of points to form a dense region
    """
    labels[P] = C #assign current point to the cluster C

    i = 0 #index
    while i < len(NeighborPts): #go over the neighbours of the current point
        Pn = NeighborPts[i]

        if labels[Pn] == -1: #change noise to part of current cluster
            labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C #assign point to current cluster
            PnNeighborPts = regionQuery(D, Pn, eps) #find neighbours for new point

            if len(PnNeighborPts) >= MinPts: #if there are enough neighbours, add them to the search are
                NeighborPts += PnNeighborPts

        i += 1

def regionQuery(D, P, eps):
    """
    Query the neighbours of a point with given radius
    :param D: dataset
    :param P: index of point whose neighbour we are searching for
    :param eps: maximum distance between points to be considered in the same cluster
    :return: list of point P neighbours indexes
    """
    neighbors = []
    for Pn in range(len(D)): #check distance between P and every other point
        if np.linalg.norm(D[P] - D[Pn]) <= eps: #if distance is smaller than eps it is a neighbour
            neighbors.append(Pn) #add to neighbour list
    return neighbors

def plot_knn(D, k, y):
    """
    Plot the KNN graph
    :param D: dataset
    :param k: number of nearest neighbours to consider
    :param y: optional, to mark a line parallel to the x axis
    :return: optimal eps calue
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(D) #fit knn model with k neighbours
    distances, indices = nbrs.kneighbors(D)
    distances = np.sort(distances[:, -1]) #sort distances for each point

    x = np.arange(len(distances)) #finding optimal eps
    curvature = np.diff(np.diff(distances))
    elbow_index = np.argmax(curvature) + 1

    #plotting
    plt.plot(distances, label=f'k={k}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-Nearest Neighbor Distance')
    plt.title(f'{k} Nearest Neighbor graph')
    if y is not None: #add horizontal line at given value
        plt.axhline(y, linestyle='--')
    plt.savefig(sys.stdout.buffer)
    plt.close()
    return distances[elbow_index]

def plot_db_scan(D, eps, k):
    """
    Plot DBSCAN clustering results
    :param D: dataset
    :param eps: maximum distance between points to be considered in the same cluster
    :param k: MinPts for DBSCAN
    """
    cluster_labels = DBSCAN(D, eps, k)  #perform DBSCAN clustering

    #plotting
    plt.figure()
    scatter = plt.scatter(D[:, 0], D[:, 1], c=cluster_labels, edgecolors="k")
    plt.title(f"DBSCAN clustering with MinPt={k},eps={eps}")
    plt.xlabel('First feature')
    plt.ylabel('Second feature')
    legend = plt.legend(*scatter.legend_elements(num=sorted(np.unique(cluster_labels))), title="Clusters")
    plt.gca().add_artist(legend)
    plt.savefig(sys.stdout.buffer)
    plt.close()

def read_data(filename):
    """
    Read data from csv file
    :param filename: given file name
    :return: numpy array containing the dataset
    """
    datadf = pd.read_csv(filename, sep=",", header=None)
    return datadf.values

def main():
    eps_values = []
    k_values = [3,4,5]

    for k in k_values:
        eps_estimate = plot_knn(D, k, None)
        eps_values.append(eps_estimate)
    eps_values = [float(eps) for eps in eps_values]
    eps = np.mean(eps_values)

    for MinPts in k_values:
        plot_db_scan(D, eps, MinPts)
        cluster_labels = DBSCAN(D, eps, MinPts)

        if len(set(cluster_labels)) > 1:
            score = silhouette_score(D, cluster_labels)
            print(f"Silhouette score for eps={eps:.3}, MinPts={MinPts}: {score:.3f}")
    print(eps_values)
data = read_data("data_clustering.csv")
D = data

if __name__ == "__main__":
    main()