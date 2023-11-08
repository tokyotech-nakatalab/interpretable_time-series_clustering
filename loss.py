import numpy as np
from tqdm import tqdm
tqdm.pandas()

def make_cluster_list(X, cluster_labels):
    return [X[cluster_labels.ravel()==i] for i in [0,1]]

def make_prototype(cluster_list):
    return np.stack([np.mean(i, axis=0) for i in cluster_list])

def gyosyuku(X_square, prototype, num):
    output = 2*(X_square.sum()) - 2*num*(np.sum(prototype**2))
    return output/(num-1)

def kairi(X0_square, X1_square, prototype_0, prototype_1, num_0, num_1):
    output = X0_square.sum()/num_0 - 2*(prototype_0@prototype_1) + X1_square.sum()/num_1
    return output

def silho_score(a,b):
    return (b-a)/max(a,b)

def loss(X, square, cluster_labels, stop_amount):
    cluster_list = make_cluster_list(X, cluster_labels)
    cluster_list_square = make_cluster_list(square, cluster_labels)
    prototype = make_prototype(cluster_list)
    num_clus0 = len(cluster_list[0])
    num_clus1 = len(cluster_list[1])
    if (num_clus0==1 or num_clus1==1) or (min(num_clus0, num_clus1)<stop_amount):
        return -np.inf
    a0 = gyosyuku(cluster_list_square[0], prototype[0], num_clus0)
    b = kairi(cluster_list_square[0], cluster_list_square[1], prototype[0], prototype[1], num_clus0, num_clus1)
    a1 = gyosyuku(cluster_list_square[1], prototype[1], num_clus1)
    C0 = silho_score(a0,b)
    C1 = silho_score(a1,b)
    output = (num_clus0/(num_clus0+num_clus1))*C0+(num_clus1/(num_clus0+num_clus1))*C1
    return output