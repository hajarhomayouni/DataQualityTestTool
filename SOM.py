from sklearn.cluster import KMeans
import pandas as pd
import sklearn
from sklearn import metrics
import collections
import numpy as np
import scipy
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import LabelEncoder
from kmodes.util.dissim import matching_dissim, euclidean_dissim
import sys
import datetime
import matplotlib.pylab as plt
from time import time
import tensorflow as tf
import numpy as np
 
 
import tensorflow as tf

import numpy as np


 

class SOM(object):

    _trained = False
    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m*n, dim]))

            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))
            self._vect_input = tf.placeholder("float", [dim])
            self._iter_input = tf.placeholder("float")
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))           

            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)                                    


            self._sess = tf.Session()
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)


    def _neuron_locations(self, m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])


    def train(self, input_vects):
        for iter_no in range(self._n_iterations):
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
        self._trained = True

    def get_centroids(self):
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
    
    def map_vects(self, input_vects):
        if not self._trained:
            raise ValueError("SOM not trained yet")
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))

            to_return.append(self._locations[min_index])
        return to_return

    
    def clusterFaultyRecords(self, faultyRecordsPreprocessed, faultyRecordsFrame):
        self.train(faultyRecordsPreprocessed.values)
        mapped= self.map_vects(faultyRecordsPreprocessed.values)
        labels_list=[list(x) for x in mapped]
        groupsOfFaultyRecords=[]
        group_index=0
        for i in range(self._m):
            for j in range(self._n):
                indexes_in_cluster=[k for k, x in enumerate(labels_list) if x == [i,j]]
                if len(faultyRecordsFrame.values[indexes_in_cluster]>0):
                    groupsOfFaultyRecords.append(faultyRecordsFrame.iloc[indexes_in_cluster])
                    group_index+=1
        return groupsOfFaultyRecords









    
    
