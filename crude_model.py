from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf
from data import graph_dicts,print_graphs_tuple,generate_data

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import random


tf.reset_default_graph()

#Generates data
data=generate_data(10000,3,3)

#Creates placeholder
graphs_tuple_ph = utils_tf.placeholders_from_data_dicts(data[0:1])

#Creates layer1
graph_network_layer1 = modules.GraphNetwork(
    edge_model_fn=lambda: snt.Linear(output_size=32),
    node_model_fn=lambda: snt.Linear(output_size=16),
    global_model_fn=lambda: snt.Linear(output_size=4))

#Creates layer2
graph_network_layer2 = modules.GraphNetwork(
    edge_model_fn=lambda: snt.Linear(output_size=8),
    node_model_fn=lambda: snt.Linear(output_size=4),
    global_model_fn=lambda: snt.Linear(output_size=1))

#Join the layers to each other
output_graphs_layer1 = graph_network_layer1(graphs_tuple_ph)
output_graphs_layer2 = graph_network_layer2(output_graphs_layer1)

for i in range(1000):
    data_batch_size=random.sample(data,k=128)
    input_graphs = utils_np.data_dicts_to_graphs_tuple(data_batch_size)
    
    #Run the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = utils_tf.get_feed_dict(
          graphs_tuple_ph, input_graphs)
        output = sess.run(output_graphs_layer2, feed_dict)
    
#    print_graphs_tuple(utils_np.get_graph(output, 0))
#    print_graphs_tuple(utils_np.get_graph(input_graphs, 0))
