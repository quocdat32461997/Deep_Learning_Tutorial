# import dependencies
from collections import defaultdict
import numpy as np

def cosine(x, y):
    x = np.array(x)
    y = np.array(y)

    top = np.dot(x,y)
    bottom = np.sqrt(np.sum(np.square(x))) *\
            np.sqrt(np.sum(np.square(y)))
    return top / bottom

def seed(docs, k):

    # define diagonal
    D = np.ones(len(list(docs.values())[0]))

    # compute cosines
    cosines = {k:cosine(D, doc) for k,doc in docs.items()}

    # sort cosines
    cosines = sorted(cosines.items(), key = lambda x: x[-1])

    # select initial clusters at index = [3, 5, 7]
    return cosines[3][0], cosines[5][0], cosines[7][0]
    #return np.random.randint(low = 1, high = len(docs), size = k)

def kmean(docs, centroids, epochs):

    def _assign_to_centroids(doc, ctrs):

        # compute cosine angles for every document against centroids
        angles = []
        for cls in ctrs:
            angles.append(cosine(doc, docs[cls]))

        # best centroid
        centroid = np.argmax(angles)
        return ctrs[centroid], angles[centroid]

    def _update_centroids(clusters, angles):
        _centroids = []

        for id, angls in angles.items():
            # find new centroid as average of clusters
            avg_ang = np.average(angls)
            new_id = np.argmin(angls - avg_ang)
            _centroids.append(clusters[id][new_id])

        return _centroids

    # execute kmeans
    for e in range(epochs):
        clusters = {k:[] for k in centroids}
        angles = {k:[] for k in centroids}
        # assign docs to clusters
        for id, doc in docs.items():
            # find closet centroid to each document
            cls, angle = _assign_to_centroids(doc, centroids)
            # assign docs
            clusters[cls].append(id)
            angles[cls].append(angle)

        # update clusters
        centroids = _update_centroids(clusters, angles)

    return centroids, list(clusters.values())

def main():
    # define parameters
    k = 3
    epochs = 5

    # define docs
    docs = [
            [0.22, 0.31, 0.66, 0.45, 0.48, 0.11, 0.33, 0.89, 0.31, 0.66, 0.11, 0.89, 0.0],
            [0.0, 0.75, 0.0, 0.11, 0.0, 0.67, 0.33, 0.0, 0.22, 0.33, 0.0, 0.5, 0.67],
            [0.5, 0.0, 0.11, 0.0, 0.0, 0.66, 0.0, 0.11, 0.0, 0.66, 0.23, 0.0, 0.11],
            [0.22, 0.31, 0.66, 0.45, 0.48, 0.11, 0.33, 0.89, 0.31, 0.66, 0.11, 0.89, 0.0],
            [0.11, 0.31, 0.0, 0.22, 0.11, 0.0, 0.0, 0.5, 0.0, 0.33, 0.0, 0.0, 0.33],
            [0.5, 0.22, 0.11, 0.0, 0.0, 0.15, 0.0, 0.33, 0.0, 0.22, 0.0, 0.66, 0.22],
            [0.33, 0.5, 0.5, 0.0, 0.0, 0.66, 0.0, 0.75, 0.0, 0.22, 0.0, 0.45, 0.45],
            [0.75, 0.14, 0.5, 0.22, 0.48, 0.11, 0.0, 0.25, 0.0, 0.0, 0.0, 0.66, 0.66],
            [0.33, 0.0, 0.22, 0.0, 0.0, 0.11, 0.0, 0.5, 0.0, 0.22, 0.0, 0.0, 0.5],
            [0.22, 0.31, 0.66, 0.45, 0.48, 0.11, 0.33, 0.89, 0.31, 0.66, 0.11, 0.89, 0.0]
    ]

    dists = []
    for doc, idx in zip(docs, range(len(docs))):
            dists.append([])
            for i in range(idx, len(docs)):
                dists[-1].append(cosine(doc, docs[i]))

    print(dists)
    return None

if __name__ == '__main__':
    main()
