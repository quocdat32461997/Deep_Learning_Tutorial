"""
File; assignment_5_dqn170000.py
Author: Dat Quoc Ngo
NET-ID: dqn170000
Date: April 25, 2021
"""

# import dependencies
from matplotlib import pyplot as io
import numpy as np
from PIL import Image
from collections import defaultdict

# set fixed seed
np.random.seed(42)

class ImageCompressor(object):
    def __init__(self, k_cluster):
        """
        Args:
            img : str
                Path to image
        """
        self.img = None
        self.k_cluster = k_cluster
        self.centroids = np.transpose(np.array([
            np.random.randint(low = 0, high = 255, size = k_cluster),
            np.random.randint(low = 0, high = 255, size = k_cluster)
        ]))
        
    def _compute_dist(self, x, y):
        """
        Compute squared Euclidean distance based on x-y coordinates
        """
        return np.sum(np.square(x - y))
    
    def _assign_pixel(self, xs, ys):
        """
        Assign color cluster to pixel
        """
        # find closet centroid
        dists = []
        for centroid in self.centroids:
            dists.append(np.square(xs - centroid[0]) + np.square(ys - centroid[1]))
        dists = np.array(dists)        
        indices = np.argmin(dists, axis = 0)

        # compute total distance
        total_dist = sum([dists[indices[x,y], x, y] for x in range(len(indices)) for y in range(len(indices[0]))])

        return indices, total_dist
    
    def _update_clusters(self, points):
        """
        Update new clusters
        Args:
            points : collections.defaultdict(list)
        Returns:
            None
        """
        
        for cls in range(self.k_cluster):
            pts = np.array(points[cls])
            self.centroids[cls] = [np.floor(np.mean(pts[:, 0])),
                                   np.floor(np.mean(pts[:, 1]))]
        return None

    def compress_img(self, img, epochs, patience = 5, delta = 0.001):
        """
        Compress image using K-Mean Clustering
        Args:
            img : np.array
                Image
            epochs : integer
                Number of epochs
            patience : integer
                Patience of convergence. If the distance changes under delta for patience, then stop
            delta : float
                Margin of difference
        """
        
        # initialize previous total distance
        prev_total_dist = np.inf
        
        # x-y coordinates
        xs = np.array([[x] * img.shape[1] for x in range(img.shape[0])])
        ys = np.array([list(range(img.shape[1]))] * img.shape[0])

        # train K-Mean
        for iter in range(epochs):
            
            # initialize centroid points
            points = defaultdict(list)

            # find closet clusters
            indices, total_dist = self._assign_pixel(xs, ys)
            
            # assign pixels
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    centroid = self.centroids[indices[x,y]]
                    # assign pixel
                    img[x,y] = img[centroid[0], centroid[1]]
                    # add points to clusters
                    points[indices[x,y]].append([x,y])
            
            # update clusters
            self._update_clusters(points)

            # reset points
            del points
            
            print('Epoch {} - Distance Loss {:2e}'.format(iter, total_dist))
            
            # terminate training
            if prev_total_dist is np.inf or delta < (1 - total_dist/prev_total_dist):
                prev_total_dist = total_dist
                patience = 5 # reset patience
            else:
                if patience == 0:
                    break
                else:
                    patience -= 1        
        return img

def main():
    # K-mean clustering parameters
    k_clusters = [2, 5, 10, 15 , 20]
    epochs = 50
    patience = 5
    delta = 0.001

    # compress Koala image
    print('Read Koala Image')
    koala = 'koala.jpg'
    koala_img = io.imread(koala)

    print('Compress Koala Image')
    koala_results = defaultdict()
    for k_c in k_clusters:
        print('Compressing Koala image with K-cluster = {}'.format(k_c))
        compressor = ImageCompressor(k_cluster = k_c)
        koala_results[k_c] = compressor.compress_img(
                img = koala_img.copy(), epochs = epochs,
                patience = patience, delta = delta)
    print('Save compressed Koala images')
    for k_c, img in koala_results.items():
        img = Image.fromarray(img)
        img.save('koala_k_cluster_{}.png'.format(k_c))

    print('Finish compressing Koala image')

    # compress Koala image
    print('Read Penguin Image')
    penguin = 'Penguins.jpg'
    penguin_img = io.imread(penguin)

    print('Compress Penguin Image')
    penguin_results = defaultdict()
    for k_c in k_clusters:
        print('Compressing Penguin image with K-cluster = {}'.format(k_c))
        compressor = ImageCompressor(k_cluster = k_c)
        penguin_results[k_c] = compressor.compress_img(
                img = penguin_img.copy(), epochs = epochs,
                patience = patience, delta = delta)
    print('Save compressed Penguin images')
    for k_c, img in penguin_results.items():
        img = Image.fromarray(img)
        img.save('penguin_k_cluster_{}.png'.format(k_c))

    print('Finish compressing Penguin image')
    return None

if __name__ == '__main__':
    main()
