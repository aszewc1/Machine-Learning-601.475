""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
def single(clust1, clust2, X):
    best_dist = 1000000000
    cluster1 = clust1.tolist()
    cluster2 = clust2.tolist()
    for k in range(len(cluster1)):
        for l in range(len(cluster2)):   
            curr_dist = np.linalg.norm(X[cluster1[k]][:] - X[cluster2[l]][:])
            if curr_dist < best_dist:
                best_dist = curr_dist
    return best_dist

def complete(clust1, clust2, X):
    best_dist = 0
    cluster1 = clust1.tolist()
    cluster2 = clust2.tolist()
    for k in range(len(cluster1)):
        for l in range(len(cluster2)):   
            curr_dist = np.linalg.norm(X[cluster1[k]][:] - X[cluster2[l]][:])
            if curr_dist > best_dist:
                best_dist = curr_dist
    if(len(cluster1) == 0 or len(cluster2)==0):
        best_dist = 10000000
    return best_dist

def average(clust1, clust2, X):
    sum = 0
    cluster1 = clust1.tolist()
    cluster2 = clust2.tolist()
    for k in range(len(cluster1)):
        for l in range(len(cluster2)):
            sum += np.linalg.norm(X[cluster1[k]][:] - X[cluster2[l]][:])
    if(len(cluster1) == 0 or len(cluster2)==0):
        return 10000000
    sum /= (len(cluster1)*len(cluster2))
    return sum

class Model(object):
    """ Abstract model object."""

    def __init__(self):
        raise NotImplementedError()

    def fit_predict(self, X):
        """ Predict

        Args:
            X: A dense matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()



class AgglomerativeClustering(Model):

    def __init__(self, n_clusters = 2, linkage = 'single'):
        """
        Args:
            n_clusters: number of clusters
            linkage: linkage criterion
        """
        # TODO: Initializations etc. go here.
        self.n_clusters = n_clusters
        self.linking = linkage


    def fit_predict(self, X):
        """ Fit and predict.

        Args:
            X: A dense matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        # TODO: Implement this!

        labels = list(range(0, X.shape[0]))
        
        num_clusters = self.n_clusters
        curr_clust_num = (np.unique(np.array(labels))).shape[0]
        while(curr_clust_num > num_clusters):
            print((np.unique(np.array(labels))).shape[0])
            best_dist = 10000000
            best_i_ind = 0
            best_j_ind = 10
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    if i == j or labels[i] == labels[j]:
                        continue

                    #indexes in X where labels = i (cluster num)
                    clust1 = np.where(np.array(labels) == i)[0]
                    clust2 = np.where(np.array(labels) == j)[0]

                    if self.linking == "single":
                        curr_dist = single(clust1, clust2, X)
                    elif self.linking == "complete":
                        curr_dist = complete(clust1, clust2, X)
                    elif self.linking == "average":
                        curr_dist = average(clust1, clust2, X)
                    
                    i_ind = i
                    j_ind = j
                    #update distance
                    if curr_dist < best_dist:
                        best_dist = curr_dist
                        best_i_ind = i_ind
                        best_j_ind = j_ind
                    #tie breaking
                    elif curr_dist == best_dist: 
                        old_clust1 = np.where(np.array(labels) == best_i_ind)[0]
                        old_clust2 = np.where(np.array(labels) == best_j_ind)[0] 
                        new_key = tuple(sorted((clust1.tolist()[0], clust2.tolist()[0])))
                        old_key = tuple(sorted((old_clust1.tolist()[0], old_clust2.tolist()[0])))
                        if new_key < old_key:
                            best_dist = curr_dist
                            best_i_ind = i_ind
                            best_j_ind = j_ind

            new_cluster = labels[best_i_ind]
            old_cluster = labels[best_j_ind]
            labels = [new_cluster if item == old_cluster else item for item in labels]
            
            curr_clust_num = (np.unique(np.array(labels))).shape[0]

        for i in range(num_clusters):
            if i in labels:
                continue
            else:
                labels = [i if item == max(labels) else item for item in labels]
            
        return labels
