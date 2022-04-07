""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np

#calculating single method between clusters
def single(clust1, clust2, X):
    #start with arbitrarily high distance
    best_dist = 1000000000
    #get the indexes of x for each cluster
    cluster1 = clust1.tolist()
    cluster2 = clust2.tolist()
    for k in range(len(cluster1)):
        for l in range(len(cluster2)):   
            #calculate distance and update if it's better
            curr_dist = np.linalg.norm(X[cluster1[k]][:] - X[cluster2[l]][:])
            if curr_dist < best_dist:
                best_dist = curr_dist
    return best_dist

#calculating complete method between clusters
def complete(clust1, clust2, X):
    #set distance arbitrarily low
    best_dist = 0
    cluster1 = clust1.tolist()
    cluster2 = clust2.tolist()
    for k in range(len(cluster1)):
        for l in range(len(cluster2)):   
            #calculate distance and if larger update
            curr_dist = np.linalg.norm(X[cluster1[k]][:] - X[cluster2[l]][:])
            if curr_dist > best_dist:
                best_dist = curr_dist
    #if clusters are empty, return arbitrarily high distance so it won't be kept
    if(len(cluster1) == 0 or len(cluster2)==0):
        best_dist = 10000000
    return best_dist

#calculate average method between clusters
def average(clust1, clust2, X):
    #initialize sum to zero
    sum = 0
    cluster1 = clust1.tolist()
    cluster2 = clust2.tolist()
    for k in range(len(cluster1)):
        for l in range(len(cluster2)):
            #add to total distance
            sum += np.linalg.norm(X[cluster1[k]][:] - X[cluster2[l]][:])
    #if either cluster is empty, return arbitrarily high distance
    if(len(cluster1) == 0 or len(cluster2)==0):
        return 10000000
    #compute average and return
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
        #initialize labels for clusters from 0 to shape of X
        labels = list(range(0, X.shape[0]))
        
        num_clusters = self.n_clusters
        curr_clust_num = (np.unique(np.array(labels))).shape[0]
        #while more clusters exist than desired final number
        while(curr_clust_num > num_clusters):
            #set best parameters arbitrarily high
            best_dist = 10000000
            best_i_ind = 0
            best_j_ind = 10
            #cycle through all possible clusters
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    #if two belong to the same cluster, skip
                    if i == j or labels[i] == labels[j]:
                        continue

                    #indexes in X where labels = i (cluster num)
                    clust1 = np.where(np.array(labels) == i)[0]
                    clust2 = np.where(np.array(labels) == j)[0]

                    #check method
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

            #merge two clusters with the smallest distance
            new_cluster = labels[best_i_ind]
            old_cluster = labels[best_j_ind]
            labels = [new_cluster if item == old_cluster else item for item in labels]
            
            curr_clust_num = (np.unique(np.array(labels))).shape[0]
            
        #get rid of large cluster numbers
        for i in range(num_clusters):
            if i in labels:
                continue
            else:
                labels = [i if item == max(labels) else item for item in labels]
            
        return labels
